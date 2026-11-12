import torch
import numpy as np
import pandas as pd
import numbers
from torch.utils.data import Dataset


def map_to_original(feature_matrices, id_maps, node_id_map, graph):
    if len(feature_matrices) != len(id_maps):
        raise ValueError(
            f"Expected one id map per feature matrix, got {len(id_maps)} maps "
            f"for {len(feature_matrices)} matrices."
        )
    max_key = max((node['id'] for node in graph['nodes']), default=-1)
    if max_key < 0:
        raise ValueError("Cannot map layer features because the graph has no nodes.")
    reference_features = next(
        (features for features in feature_matrices if features.size(0) > 0),
        None,
    )
    if reference_features is None:
        raise ValueError("Cannot map layer features because every layer is empty.")
    feature_length = reference_features.size(1)
    device = reference_features.device
    original_mat = torch.zeros(max_key + 1, feature_length, device=device)

    for i in range(len(id_maps)):
        reversed_map = {value: key for key, value in id_maps[i].items()}
        for key, value in reversed_map.items():
            if key >= feature_matrices[i].size(0):
                raise IndexError(
                    f"Layer {i} node index {key} is outside its feature matrix "
                    f"with {feature_matrices[i].size(0)} rows."
                )
            original_mat[value] = feature_matrices[i][key]

    node_df = pd.DataFrame(graph["nodes"])
    node_id_map = node_id_map.reset_index(drop=True)
    merged_nodes = pd.merge(
        node_df,
        node_id_map,
        how='left',
        left_on=['label', 'type'],
        right_on=['id', 'type'],
        validate='many_to_one',
    )
    if merged_nodes['id_y'].isna().any():
        raise ValueError("Graph contains nodes missing from node_id_map.")
    max_layer = merged_nodes['layer'].max()
    num_labels = len(node_id_map)
    entity_index_map = {
        (row['id'], row['type']): idx
        for idx, row in node_id_map.iterrows()
    }
    layered_features = {
        i: torch.zeros(num_labels, feature_length, device=device)
        for i in range(max_layer + 1)
    }

    # Organize features by layer
    for _, row in merged_nodes.iterrows():
        label_idx = entity_index_map[(row['label'], row['type'])]
        layer = row['layer']
        graph_node_id = row['id_x']
        if graph_node_id < 0 or graph_node_id > max_key:
            raise IndexError(f"Graph node ID {graph_node_id} is outside the original feature matrix.")
        layered_features[layer][label_idx, :] = original_mat[graph_node_id]

    return list(layered_features.values())


def map_to_layer(feature_matrix, layer_index, id_maps, node_id_map, graph):
    node_id_map = node_id_map.reset_index(drop=True)
    entity_index_map = {
        (row['id'], row['type']): idx
        for idx, row in node_id_map.iterrows()
    }
    graph_nodes = {node['id']: node for node in graph['nodes']}
    local_to_entity = sorted(
        id_maps[layer_index].items(),
        key=lambda item: item[1],
    )
    entity_indices = []
    for original_id, _ in local_to_entity:
        node = graph_nodes[original_id]
        key = (node['label'], node['type'])
        if key not in entity_index_map:
            raise ValueError(f"Graph node {original_id} is missing from node_id_map.")
        entity_indices.append(entity_index_map[key])

    if not entity_indices:
        return feature_matrix.new_empty((0, feature_matrix.size(-1)))
    return feature_matrix[entity_indices, layer_index, :]


def layer_presence_mask(id_maps, node_id_map, graph, device=None, edge_index=None):
    node_id_map = node_id_map.reset_index(drop=True)
    entity_index_map = {
        key: index
        for index, key in enumerate(zip(node_id_map['id'], node_id_map['type']))
    }
    graph_nodes = {node['id']: node for node in graph['nodes']}
    rows, cols = [], []
    for layer, id_map in enumerate(id_maps):
        if edge_index is None:
            layer_nodes = id_map
        else:
            local_to_original = {local_id: original_id for original_id, local_id in id_map.items()}
            layer_nodes = [
                local_to_original[local_id]
                for local_id in torch.unique(edge_index[layer]).tolist()
            ]
        for original_id in layer_nodes:
            node = graph_nodes[original_id]
            key = (node['label'], node['type'])
            if key not in entity_index_map:
                raise ValueError(f"Graph node {original_id} is missing from node_id_map.")
            rows.append(entity_index_map[key])
            cols.append(layer)
    mask = torch.zeros(len(node_id_map), len(id_maps), dtype=torch.bool)
    if rows:
        mask[rows, cols] = True
    return mask.to(device) if device is not None else mask


class MyDataset(Dataset):
    def __init__(self, data, emb, emd2=None, entity_index_map=None):
        self.data = data.reset_index()
        self.data = self.data.drop(['index'], axis=1)
        self.x = self.data[['drug', 'gene']]
        self.y = self.data.drop(['drug', 'gene'], axis=1)
        self.emb = emb
        self.emb2 = emd2
        self.entity_index_map = entity_index_map

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        drug_id = self._entity_index(self.x.iloc[idx]['drug'], 'drug')
        gene_id = self._entity_index(self.x.iloc[idx]['gene'], 'gene')
        if self.emb2 is None:
            x1 = self._lookup(self.emb, self._embedding_index(drug_id, 'drug'), 'drug')
            x2 = self._lookup(self.emb, self._embedding_index(gene_id, 'gene'), 'gene')
        elif self.emb is None:
            x1 = self._lookup(self.emb2, drug_id, 'drug')
            x2 = self._lookup(self.emb2, gene_id, 'gene')
        else:
            x1 = torch.cat((
                self._lookup(self.emb2, drug_id, 'drug'),
                self._lookup(self.emb, self._embedding_index(drug_id, 'drug'), 'drug'),
            ), dim=0)
            x2 = torch.cat((
                self._lookup(self.emb2, gene_id, 'gene'),
                self._lookup(self.emb, self._embedding_index(gene_id, 'gene'), 'gene'),
            ), dim=0)

        x1 = x1.to(torch.float32)
        x2 = x2.to(torch.float32)

        return x1, x2, torch.tensor(self.y.iloc[idx].values, dtype=torch.float32)

    def _embedding_index(self, entity_id, entity_type):
        if self.entity_index_map is None:
            return entity_id
        key = (entity_id, entity_type)
        if key not in self.entity_index_map:
            raise KeyError(f"{entity_type} ID {entity_id} is missing from the embedding index map.")
        return self.entity_index_map[key]

    @staticmethod
    def _entity_index(value, entity_type):
        if isinstance(value, numbers.Integral):
            return int(value)
        if isinstance(value, numbers.Real) and float(value).is_integer():
            return int(value)
        raise TypeError(f"{entity_type} ID {value!r} must be an integer embedding index.")

    @staticmethod
    def _lookup(embeddings, entity_id, entity_type):
        if entity_id < 0 or entity_id >= embeddings.size(0):
            raise IndexError(
                f"{entity_type} ID {entity_id} is outside the embedding table "
                f"with {embeddings.size(0)} rows."
            )
        return embeddings[entity_id]


def get_llm_features(graph, llm):
    if llm is None:
        raise ValueError("llm embeddings are required to build sequence features.")
    if isinstance(llm, torch.Tensor):
        llm_values = llm.detach().cpu().numpy()
    elif hasattr(llm, 'to_numpy'):
        llm_values = llm.fillna(0).to_numpy()
    else:
        llm_values = np.asarray(llm)
    if llm_values.ndim != 2:
        raise ValueError(f"llm embeddings must be a 2D entity-by-feature array, got {llm_values.shape}.")

    feature_dict = {}
    for node in graph['nodes']:
        entity_id = node['label']
        if not isinstance(entity_id, numbers.Integral):
            raise TypeError(f"Entity ID {entity_id!r} must be an integer index into llm embeddings.")
        if entity_id < 0 or entity_id >= len(llm_values):
            raise IndexError(
                f"Entity ID {entity_id} is outside the llm embedding table with {len(llm_values)} rows."
            )
        feature_dict[node['id']] = llm_values[entity_id]
    return feature_dict


def concatenate_embedding(dict1, dict2):
    result_dict = {}
    for key in dict1:
        result_dict[key] = np.concatenate((dict1[key], dict2[key]))
    return result_dict
