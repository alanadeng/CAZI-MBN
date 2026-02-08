import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


def map_to_original(feature_matrices, id_maps, node_id_map, graph):
    max_key = max(max(d.keys(), default=None) for d in id_maps if d)
    feature_length = len(feature_matrices[0][0])
    device = feature_matrices[0].device
    original_mat = torch.zeros(max_key + 1, feature_length, device=device)


    for i in range(len(id_maps)):
        reversed_map = {value: key for key, value in id_maps[i].items()}
        for key, value in reversed_map.items():
            original_mat[value] = feature_matrices[i][key]

    node_df = pd.DataFrame(graph["nodes"])
    merged_nodes = pd.merge(node_df, node_id_map, how='left', left_on=['label', 'type'], right_on=['id', 'type'])
    max_layer = merged_nodes['layer'].max()
    label_index_map = {label: idx for idx, label in enumerate(merged_nodes['label'].unique())}
    num_labels = len(node_id_map)
    #num_labels = len(label_index_map)
    # layered_features = {i: torch.zeros(num_labels, feature_length) for i in range(max_layer + 1)}
    layered_features = {i: torch.zeros(num_labels, feature_length, device=device) for i in range(max_layer + 1)}


    # Organize features by layer
    for index, row in merged_nodes.iterrows():
        label_idx = label_index_map[row['label']]
        layer = row['layer']
        layered_features[layer][label_idx, :] = original_mat[index]

    return list(layered_features.values())


class MyDataset(Dataset):
    def __init__(self, data, emb, emd2=None):
        self.data = data.reset_index()
        self.data = self.data.drop(['index'], axis=1)
        self.x = self.data[['drug', 'gene']]
        self.y = self.data.drop(['drug', 'gene'], axis=1)
        self.emb = emb
        self.emb2 = emd2

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if (self.emb2 is None):
            x1 = self.emb[self.x['drug'][idx]]
            x2 = self.emb[self.x['gene'][idx]]
        elif (self.emb is None):
            x1 = self.emb2[self.x['drug'][idx]]
            x2 = self.emb2[self.x['gene'][idx]]
        else:
            x1 = torch.cat((self.emb2[self.x['drug'][idx]], self.emb[self.x['drug'][idx]]), dim=0)
            x2 = torch.cat((self.emb2[self.x['gene'][idx]], self.emb[self.x['gene'][idx]]), dim=0)

        x1 = x1.to(torch.float32)
        x2 = x2.to(torch.float32)

        return x1, x2, torch.tensor(self.y.iloc[idx].values, dtype=torch.float32)


def get_llm_features(graph, llm):
    feature_dict = {}
    for node in graph['nodes']:
        feature_dict[node['id']] = llm.iloc[node['label']].values
    return feature_dict


def concatenate_embedding(dict1, dict2):
    result_dict = {}
    for key in dict1:
        result_dict[key] = np.concatenate((dict1[key], dict2[key]))
    return result_dict
