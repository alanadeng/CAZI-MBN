import random
import numbers
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch.utils.data import DataLoader
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from utilities.helpers import map_to_original, map_to_layer, MyDataset, get_llm_features, concatenate_embedding, layer_presence_mask
from utilities.dataSplitter import DataSplitter
from utilities.graphHandler import GraphHandler
from utilities.edgeExtractor import EdgeExtractor
from utilities.layerProcessor import LayerProcessor
from module.topologyAwareProjection import TopologyAwareFeatureProjection
from module.graphTransformer import GraphTransformer
from module.attention import Attention
from module.discriminator import Discriminator
from module.readout import AvgReadout, SAGPoolReadoutEdge
from module.MoEChain import MoE
from module.loss import FocalLoss, KDLoss, MultiLabelSoftMarginLoss
from .MultiLabelClassifier import MultiLabelClassifier


class CAZI:
    def __init__(self, seed, node_id_map, data, device=None, mode='transductive', gnn='transformer', is_supervised=True, n2v=False, batch_size=256,
                 readout='SAGPool', num_layers=6, in_channels=64, latent_channels=32, order=5, reg_coef=1.0, cls_coef=10.0, lr=1e-3, l2_coef=1e-3, llm=None, path="best_model.pth",
                 cls_loss='mls', strategy='topo', nheads=4, sampling=False):
        if device is None:
            self.device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.seed = seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        random.seed(self.seed)

        self.mode = mode
        self.gnn = gnn
        self.is_supervised = is_supervised
        self.batch_size = batch_size
        required_columns = {'drug', 'gene'}
        missing_columns = required_columns.difference(data.columns)
        if missing_columns:
            raise ValueError(f"Interaction data is missing required columns: {sorted(missing_columns)}.")
        self.interaction_columns = [
            column for column in data.columns if column not in required_columns
        ]
        if len(self.interaction_columns) != num_layers:
            raise ValueError(
                f"num_layers={num_layers} must match the number of interaction columns "
                f"({len(self.interaction_columns)})."
            )
        if not self.interaction_columns:
            raise ValueError("Interaction data must contain at least one label column.")
        labels = data[self.interaction_columns]
        if labels.isna().any().any() or not labels.isin([0, 1]).all().all():
            raise ValueError("Interaction label columns must contain only binary 0/1 values.")
        if not isinstance(node_id_map, pd.DataFrame):
            raise TypeError("node_id_map must be a pandas DataFrame with 'id' and 'type' columns.")
        if not {'id', 'type'}.issubset(node_id_map.columns):
            raise ValueError("node_id_map must contain 'id' and 'type' columns.")
        if node_id_map.duplicated(['id', 'type']).any():
            raise ValueError("node_id_map must contain unique (id, type) pairs.")
        self.node_id_map = node_id_map.reset_index(drop=True)
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.reg_coef = reg_coef
        self.lr = lr
        self.l2_coef = l2_coef
        self.path = path
        self.cls_coef = cls_coef
        self.cls_loss = cls_loss
        self.latent_channels = latent_channels
        self.latent_proj = nn.Linear(self.in_channels, self.latent_channels)
        self.data = data
        self.num_nodes = len(self.node_id_map)
        splitter = DataSplitter(data, mode=mode, seed=self.seed)
        self.train_data, self.val_data, self.test_data = splitter.split()

        graph_handler = GraphHandler(
            self.train_data,
            self.val_data,
            test_data=self.test_data,
            num_layer=self.num_layers,
            interaction_columns=self.interaction_columns,
        )
        self.graph = graph_handler.generate_graph()

        edge_extractor = EdgeExtractor(self.graph, seed=self.seed)
        train_edges, val_edges, test_edges = edge_extractor.generate_edges()

        network_extractor = TopologyAwareFeatureProjection(
            train_edges,
            self.in_channels,
            order,
            nodes=[node['id'] for node in self.graph['nodes']],
        )
        self.embeddings = network_extractor.get_node_features()

        if isinstance(llm, torch.Tensor):
            llm_values = llm.detach().cpu().numpy()
        elif hasattr(llm, 'to_numpy'):
            llm_values = llm.to_numpy()
        elif llm is None:
            llm_values = None
        else:
            llm_values = np.asarray(llm)
        if llm_values is not None:
            if llm_values.ndim != 2:
                raise ValueError(
                    f"llm embeddings must be a 2D entity-by-feature matrix, got {llm_values.shape}."
                )
            if not np.isfinite(llm_values).all():
                raise ValueError("llm embeddings must contain only finite values.")
            llm_values = llm_values.astype(np.float32, copy=False)
        self.llm = llm_values

        self.strategy = strategy
        if self.strategy == "llm_only":
            llm_embeddings = get_llm_features(self.graph, self.llm)
            self.embeddings = llm_embeddings
            self.llm = torch.tensor(self.llm, dtype=torch.float32, device=self.device)
            self.in_channels = self.llm.shape[1]
            dim_llm = 0
        elif self.strategy == "llm_topo":
            llm_embeddings = get_llm_features(self.graph, self.llm)
            self.embeddings = concatenate_embedding(self.embeddings, llm_embeddings)
            self.llm = torch.tensor(self.llm, dtype=torch.float32, device=self.device)
            self.in_channels = self.in_channels + self.llm.shape[1]
            dim_llm = 0
        else:
            if self.llm is None:
                dim_llm = 0
            else:
                self.llm = torch.tensor(self.llm, dtype=torch.float32, device=self.device)
                dim_llm = self.llm.shape[1]

        self.readout = readout

        if self.readout=='SAGPool':
            self.readout_func = SAGPoolReadoutEdge(self.in_channels)
        else:
            self.readout_func = AvgReadout().to(self.device)

        self.readout_act_func = nn.Sigmoid().to(self.device)

        layer_processor = LayerProcessor(
            nodes=self.graph["nodes"],
            train_edges=train_edges,
            val_edges=val_edges,
            test_edges=test_edges,
            features_dict=self.embeddings,
            num_layers=self.num_layers
        )
        self.dataset, self.id_map = layer_processor.create_layer_datasets()

        self.model = Model(self.num_nodes, self.num_layers, self.in_channels, self.readout, self.readout_func,
                         self.readout_act_func, gnn=self.gnn, is_supervised=self.is_supervised, dim_llm=dim_llm, nheads=nheads,
                         strategy=self.strategy, sampling=sampling, latent_channels=self.latent_channels).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.l2_coef)

        self.x_list, self.train_edges_list, self.val_edges_list, self.test_edges_list, self.train_edges_neg, self.val_edges_neg, self.test_edges_neg = self.prepare_data()

        self.focal_loss = FocalLoss(gamma=2, alpha=0.25).to(self.device)
        self.mlsml = MultiLabelSoftMarginLoss()

    def negative_sampling(self, edge_list, known_edges=None, candidate_edges=None):
        positive_edges = set(edge_list)
        num_positive = len(positive_edges)

        if num_positive == 0:
            return []
        sampling_edges = candidate_edges if candidate_edges is not None else edge_list
        src_nodes = {edge[0] for edge in sampling_edges}
        dst_nodes = {edge[1] for edge in sampling_edges}
        known_edges = set(known_edges) if known_edges is not None else positive_edges

        possible_pairs = len(src_nodes) * len(dst_nodes) - len(src_nodes.intersection(dst_nodes))
        known_candidate_pairs = sum(
            1 for src, dst in known_edges
            if src in src_nodes and dst in dst_nodes and src != dst
        )
        if possible_pairs - known_candidate_pairs < num_positive:
            raise ValueError("Not enough unique node pairs are available for negative sampling.")

        negative_edges = set()
        max_attempts = max(1000, num_positive * 50)
        attempts = 0

        while len(negative_edges) < num_positive:
            src = random.choice(tuple(src_nodes))
            dst = random.choice(tuple(dst_nodes))
            if src != dst and (src, dst) not in known_edges and (src, dst) not in negative_edges:
                negative_edges.add((src, dst))
            attempts += 1
            if attempts >= max_attempts and len(negative_edges) < num_positive:
                raise RuntimeError("Unable to sample enough unique negative edges.")

        return list(negative_edges)
    
    def prepare_data(self):
        x_list = []
        train_edges_list, val_edges_list, test_edges_list = [], [], []
        train_edges_neg, val_edges_neg, test_edges_neg = [], [], []

        for layer in range(len(self.dataset)):
            feats = self.dataset[layer][0]
            feature_rows = []
            for node_id in sorted(feats):
                feature = feats[node_id]
                if feature is None:
                    feature = np.zeros(self.in_channels, dtype=np.float32)
                feature = np.asarray(feature, dtype=np.float32).reshape(-1)
                if feature.size != self.in_channels:
                    raise ValueError(
                        f"Layer {layer} node {node_id} has {feature.size} features; "
                        f"expected {self.in_channels}."
                    )
                feature_rows.append(feature)
            feature_array = (
                np.stack(feature_rows)
                if feature_rows
                else np.empty((0, self.in_channels), dtype=np.float32)
            )
            x = torch.tensor(feature_array, dtype=torch.float, device=self.device)
            x_list.append(x)

            train_edges = list(set(self.dataset[layer][1]))
            val_edges   = list(set(self.dataset[layer][2]))
            test_edges  = list(set(self.dataset[layer][3]))
            known_edges = self._known_layer_edges(layer)
            known_train_edges = self._known_layer_edges(layer, self.train_data)

            train_edges_list.append(self._edge_tensor(train_edges))
            val_edges_list.append(self._edge_tensor(val_edges))
            test_edges_list.append(self._edge_tensor(test_edges))

            train_edges_neg.append(self._edge_tensor(
                self.negative_sampling(train_edges, known_train_edges, train_edges)
            ))
            val_edges_neg.append(self._edge_tensor(
                self.negative_sampling(val_edges, known_edges, train_edges + val_edges)
            ))
            test_edges_neg.append(self._edge_tensor(
                self.negative_sampling(test_edges, known_edges, train_edges + test_edges)
            ))

        return x_list, train_edges_list, val_edges_list, test_edges_list, train_edges_neg, val_edges_neg, test_edges_neg

    def _known_layer_edges(self, layer, data=None):
        if data is None:
            data = self.data
        node_by_id = {node['id']: node for node in self.graph['nodes']}
        local_to_node = {
            local_id: node_by_id[original_id]
            for original_id, local_id in self.id_map[layer].items()
        }
        drug_ids = {
            node['label']: local_id
            for local_id, node in local_to_node.items()
            if node['type'] == 'drug'
        }
        gene_ids = {
            node['label']: local_id
            for local_id, node in local_to_node.items()
            if node['type'] == 'gene'
        }

        positive_pairs = set()
        for _, row in data.iterrows():
            if any(row[column] == 1 for column in self.interaction_columns):
                drug_id = row['drug']
                gene_id = row['gene']
                if drug_id in drug_ids and gene_id in gene_ids:
                    positive_pairs.add((drug_ids[drug_id], gene_ids[gene_id]))
        return positive_pairs

    def _edge_tensor(self, edges):
        if not edges:
            return torch.empty((2, 0), dtype=torch.long, device=self.device)
        return torch.tensor(edges, dtype=torch.long, device=self.device).t().contiguous()


    # def prepare_data(self):
    #     x_list = []
    #     train_edges_list = []
    #     val_edges_list = []
    #     test_edges_list = []
    #     train_edges_neg = []
    #     val_edges_neg = []
    #     test_edges_neg = []

    #     for layer in range(len(self.dataset)):
    #         x_list = [float(self.dataset[layer][0][i]) for i in sorted(self.dataset[layer][0]) if isinstance(self.dataset[layer][0][i], (int, float))]
    #         x_array = np.array(x_list, dtype=float)
    #         x = torch.tensor(x_array, dtype=torch.float).to(self.device)
    #         x_list.append(x)
    #         train_edges_list.append(torch.tensor(list(set(self.dataset[layer][1])), dtype=torch.long).t().contiguous().to(self.device))
    #         train_edges_neg.append(torch.tensor(self.negative_sampling(self.dataset[layer][1]), dtype=torch.long).t().contiguous().to(self.device))
    #         val_edges_list.append(torch.tensor(list(set(self.dataset[layer][2])), dtype=torch.long).t().contiguous().to(self.device))
    #         val_edges_neg.append(torch.tensor(self.negative_sampling(self.dataset[layer][2]), dtype=torch.long).t().contiguous().to(
    #                 self.device))
    #         test_edges_list.append(torch.tensor(list(set(self.dataset[layer][3])), dtype=torch.long).t().contiguous().to(self.device))
    #         test_edges_neg.append(torch.tensor(self.negative_sampling(self.dataset[layer][3]), dtype=torch.long).t().contiguous().to(
    #                 self.device))

    #     return x_list, train_edges_list, val_edges_list, test_edges_list, train_edges_neg, val_edges_neg, test_edges_neg


    def train(self, epochs=1000, patience_limit=200):
        best_val_auroc = float('-inf')
        best_state = None
        patience_counter = 0

        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=75, gamma=0.75)

        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()

            result = self.model(self.device, self.train_data, self.llm, self.batch_size, self.x_list, self.train_edges_list, self.train_edges_neg,
                                self.id_map, self.node_id_map, self.graph, None, None,
                                encoder_edge_index=self.train_edges_list,
                                encoder_edge_neg_index=self.train_edges_neg)
            xent_loss = self._discriminator_loss(result['logits'])
            loss = xent_loss
            reg_loss = result['reg_loss'].to(self.device)
            loss += self.reg_coef * reg_loss

            if self.is_supervised:
                if self.cls_loss == "mls":
                    supervised_loss = self.mlsml(result['y_pred'], result['y_true'])
                    # supervised_loss = b_xent(result['y_pred'].to(self.device), result['y_true'].to(self.device)).to(self.device)
                else:
                    supervised_loss = self.focal_loss(result['y_pred'].to(self.device), result['y_true'].to(self.device)).to(self.device)
                    self.cls_coef = 100.0
                loss += supervised_loss * self.cls_coef
                loss.backward()
                self.optimizer.step()
                scheduler.step()
                val_auroc, val_auprc = self.evaluate(self.val_data, self.val_edges_list, self.val_edges_neg)


                if val_auroc > best_val_auroc:
                    best_val_auroc = val_auroc
                    patience_counter = 0
                    best_state = {
                        name: value.detach().clone()
                        for name, value in self.model.state_dict().items()
                    }
                    torch.save(self.model.state_dict(), self.path)
                else:
                    patience_counter += 1
                if patience_counter >= patience_limit:
                    print(f"Stopping early at epoch {epoch}")
                    break
                print(f"Epoch {epoch}, Loss:{loss.item():.4f}, Discriminator: {xent_loss.item():.4f}, Consensus: {reg_loss.item():.4f}, Classifier:{supervised_loss.item():.4f}")
            else:
                loss.backward()
                self.optimizer.step()
                scheduler.step()
                print(f"Epoch {epoch}, Loss:{loss.item():.4f}, Discriminator: {xent_loss.item():.4f}, Consensus: {reg_loss.item():.4f}")

        if self.is_supervised and best_state is not None:
            self.model.load_state_dict(best_state)
        elif self.is_supervised and epochs > 0:
            raise RuntimeError("Teacher training completed without a finite validation AUROC.")

    def _discriminator_loss(self, logits):
        b_xent = nn.BCEWithLogitsLoss()
        losses = []
        for logit in logits:
            if logit.numel() == 0:
                continue
            if logit.numel() % 2:
                raise ValueError("Discriminator logits must contain equal positive and negative batches.")
            count = logit.numel() // 2
            labels = torch.cat((
                torch.ones(count, device=logit.device),
                torch.zeros(count, device=logit.device),
            ))
            losses.append(b_xent(logit, labels))
        if not losses:
            raise ValueError("Cannot train the discriminator without positive and negative edges.")
        return torch.stack(losses).sum()


    def evaluate(self, data, edge_list, edge_neg_list):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(
                self.device, data, self.llm, self.batch_size, self.x_list,
                edge_list, edge_neg_list, self.id_map, self.node_id_map,
                self.graph, None, None,
                encoder_edge_index=self.train_edges_list,
                encoder_edge_neg_index=self.train_edges_neg,
            )
        auroc = roc_auc_score(outputs['y_true'].cpu().detach(), outputs['y_pred'].cpu().detach(), average='macro')
        auprc = average_precision_score(outputs['y_true'].cpu().detach(), outputs['y_pred'].cpu().detach(), average='macro')
        return auroc, auprc

    def test(self):
        if self.is_supervised:
            self.model.load_state_dict(torch.load(self.path, map_location=self.device))
            auroc, auprc = self.evaluate(self.test_data, self.test_edges_list, self.test_edges_neg)
            return auroc, auprc
        else:
            ban_model = MultiLabelClassifier(
                self.seed, self.data, device=self.device, num_layers=self.num_layers,
                batch_size=self.batch_size, mode=self.mode, lr=0.01,
                llm=self._entity_embedding_table(self.model.H.detach()),
            )
            ban_model.train_data = self.train_data
            ban_model.val_data = self.val_data
            ban_model.test_data = self.test_data
            if self.mode == 'inductive':
                ban_model.train_data_ind = self.train_data
            ban_model.train(epochs=100, patience_limit=10)
            return ban_model.test()

    def _entity_embedding_table(self, embeddings):
        entity_ids = self.node_id_map['id']
        if not all(isinstance(entity_id, numbers.Integral) and entity_id >= 0 for entity_id in entity_ids):
            raise TypeError("node_id_map ids must be non-negative integers to index entity embeddings.")
        if entity_ids.duplicated().any():
            raise ValueError("node_id_map ids must be unique across entity types to build an embedding table.")
        table = embeddings.new_zeros((int(entity_ids.max()) + 1, embeddings.size(1)))
        table[torch.as_tensor(entity_ids.to_numpy(dtype=np.int64), device=embeddings.device)] = embeddings
        return table

    def co_train(self, is_teacher=False, helper_vectors=None):
        self.model.train()
        self.optimizer.zero_grad()

        result = self.model(self.device, self.train_data, self.llm, self.batch_size,
                            self.x_list, self.train_edges_list, self.train_edges_neg,
                            self.id_map, self.node_id_map, self.graph, None, None,
                            encoder_edge_index=self.train_edges_list,
                            encoder_edge_neg_index=self.train_edges_neg)

        logits = result['logits']
        loss = self._discriminator_loss(logits)

        reg_loss = result['reg_loss'].to(self.device)
        loss += self.reg_coef * reg_loss

        if self.is_supervised:
            if self.cls_loss == "mls":
                supervised_loss = self.mlsml(result['y_pred'], result['y_true'])
            else:
                supervised_loss = self.focal_loss(result['y_pred'], result['y_true'])
                self.cls_coef = 100.0
            loss += supervised_loss * self.cls_coef

        student_latent = result['latent']
        kd_loss = KDLoss()
        if helper_vectors is None:
            total_loss = kd_loss(student_latent, None, loss, is_teacher)
        else:
            total_loss = kd_loss(student_latent, helper_vectors.to(self.device), loss, is_teacher)

        total_loss.backward()
        self.optimizer.step()

        val_auroc, val_auprc = self.evaluate(self.val_data, self.val_edges_list, self.val_edges_neg)
        return student_latent.detach(), val_auroc, val_auprc

    def co_evaluate(self, data, edge_list):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.device, data, self.llm, self.batch_size,
                                 self.x_list, edge_list, self.train_edges_neg,
                                 self.id_map, self.node_id_map, self.graph, None, None,
                                 encoder_edge_index=self.train_edges_list,
                                 encoder_edge_neg_index=self.train_edges_neg)
        auroc = roc_auc_score(outputs['y_true'].cpu().detach(), outputs['y_pred'].cpu().detach(), average='macro')
        auprc = average_precision_score(outputs['y_true'].cpu().detach(), outputs['y_pred'].cpu().detach(),
                                        average='macro')
        return outputs['latent'].cpu().detach(), auroc, auprc

    def supervise(self):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.device, self.train_data, self.llm, self.batch_size,
                                 self.x_list, self.train_edges_list, self.train_edges_neg,
                                 self.id_map, self.node_id_map, self.graph, None, None,
                                 encoder_edge_index=self.train_edges_list,
                                 encoder_edge_neg_index=self.train_edges_neg)
        auroc = roc_auc_score(outputs['y_true'].cpu().detach(), outputs['y_pred'].cpu().detach(), average='macro')
        auprc = average_precision_score(outputs['y_true'].cpu().detach(), outputs['y_pred'].cpu().detach(),
                                        average='macro')
        return outputs['latent'].cpu().detach(), auroc, auprc


class Model(nn.Module):
    def __init__(self, num_nodes, num_layers, in_channels, readout, readout_func, readout_act_func,
                 gnn='transformer', nheads=4, is_supervised=True, dim_llm=0,
                 strategy='topo', sampling=False, latent_channels=32):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.readout = readout
        self.readout_func = readout_func
        self.readout_act_func = readout_act_func
        self.nheads = nheads
        self.is_supervised = is_supervised
        self.strategy = strategy
        self.sampling = sampling
        self.latent_channels = latent_channels
        self.theta = nn.Parameter(torch.ones(self.num_layers, self.in_channels))


        self.gnn = nn.ModuleList([
            GraphTransformer(in_channels, int(in_channels * 2), in_channels, heads=self.nheads)
            for _ in range(self.num_layers)
        ])

        self.disc = Discriminator(in_channels * 2)
        self.H = nn.Parameter(torch.FloatTensor(self.num_nodes, self.in_channels))
        self.latent_proj = nn.Linear(self.in_channels, self.latent_channels)
        self.attn = Attention(self.in_channels, self.num_layers, self.num_nodes)

        # if self.is_supervised:
        #     self.classifier_chain = MoE(self.in_channels + dim_llm, self.in_channels + dim_llm, self.num_layers)
        if self.is_supervised:
            in2_dim = dim_llm if self.strategy == "topo" else self.in_channels

            classifier_input_dim = self.latent_channels + in2_dim if self.strategy == "topo" else self.latent_channels
            self.classifier_chain = MoE(
                in1_dim=classifier_input_dim,
                in2_dim=classifier_input_dim,
                out_dim=self.num_layers,
                num_experts=self.num_layers,
                latent_dim=self.latent_channels,
            )


        self.init_weight()

    def init_weight(self):
        nn.init.xavier_normal_(self.H)

    def node_level_attention(self, H, mask=None):
        N, L, d = H.shape
        if L == 1:
            return H
        if mask is None:
            mask = torch.ones(N, L, dtype=torch.bool, device=H.device)
        mask = mask.to(device=H.device, dtype=torch.bool)
        eps = torch.finfo(H.dtype).eps
        H_new = []

        for p in range(L):
            h_p = H[:, p, :]
            counterpart = mask.clone()
            counterpart[:, p] = False
            has_counterpart = counterpart.any(dim=1, keepdim=True)

            scores = torch.sigmoid((h_p.unsqueeze(1) * H * self.theta[p]).sum(dim=-1)) * counterpart
            normalized_scores = scores / scores.sum(dim=1, keepdim=True).clamp_min(eps)
            normalized_scores = normalized_scores.masked_fill(~counterpart, float('-inf'))
            normalized_scores = torch.where(has_counterpart, normalized_scores, torch.zeros_like(normalized_scores))
            attn_weights = F.softmax(normalized_scores, dim=1) * has_counterpart
            context = (attn_weights.unsqueeze(-1) * H).sum(dim=1)
            H_new.append((h_p + context) * mask[:, p].unsqueeze(-1))

        return torch.stack(H_new, dim=1)

    def forward(self, device, data, llm, batch_size, x, edge_index, edge_neg_index, id_maps,
                node_id_map, graph, samp_bias1, samp_bias2,
                encoder_edge_index=None, encoder_edge_neg_index=None):
        h_1_layers = []
        h_2_layers = []
        logits = []
        result = {}
        if encoder_edge_index is None:
            encoder_edge_index = edge_index
        if encoder_edge_neg_index is None:
            encoder_edge_neg_index = edge_neg_index

        for i in range(self.num_layers):
            if self.sampling == True:
                sampling_data = Data(x=x[i], edge_index=encoder_edge_index[i])
                loader = NeighborLoader(
                    sampling_data,
                    num_neighbors=[15, 10, 5],
                    batch_size=512,
                    shuffle=True,
                    drop_last=False,
                )
                sampling_neg_data = Data(x=x[i], edge_index=encoder_edge_neg_index[i])
                neg_loader = NeighborLoader(
                    sampling_neg_data,
                    num_neighbors=[15, 10, 5],
                    batch_size=512,
                    shuffle=True,
                    drop_last=False,
                )
                h_1 = torch.zeros((x[i].size(0), self.in_channels), device=x[i].device)
                h_2 = torch.zeros((x[i].size(0), self.in_channels), device=x[i].device)
                for batch in loader:
                    h_1[batch.n_id] = self.gnn[i](batch.x, batch.edge_index)
                for neg_batch in neg_loader:
                    h_2[neg_batch.n_id] = self.gnn[i](neg_batch.x, neg_batch.edge_index)
            else:
                h_1 = self.gnn[i](x[i], encoder_edge_index[i])
                h_2 = self.gnn[i](x[i], encoder_edge_neg_index[i])
            h_1_layers.append(h_1)
            h_2_layers.append(h_2)

        presence = layer_presence_mask(id_maps, node_id_map, graph, device=device, edge_index=encoder_edge_index)

        h_1_original = map_to_original(h_1_layers, id_maps, node_id_map, graph)
        h_1_original = torch.stack(h_1_original, dim=1).to(device)
        h_1_context = self.node_level_attention(h_1_original, presence)
        h_1_all, _ = self.attn.aggregate(h_1_context.transpose(0, 1), mask=presence)

        h_2_original = map_to_original(h_2_layers, id_maps, node_id_map, graph)
        h_2_original = torch.stack(h_2_original, dim=1).to(device)
        h_2_all, _ = self.attn.aggregate(h_2_original.transpose(0, 1), mask=presence)

        if h_1_all.shape != self.H.shape or h_2_all.shape != self.H.shape:
            raise ValueError(
                f"Aggregated graph embeddings must match the consensus embedding shape "
                f"{self.H.shape}; got positive {h_1_all.shape} and negative {h_2_all.shape}."
            )

        positive_agreement = F.cosine_similarity(self.H, h_1_all, dim=1).mean()
        negative_agreement = F.cosine_similarity(self.H, h_2_all, dim=1).mean()
        cosine_loss = 1 - positive_agreement + negative_agreement
        result['reg_loss'] = cosine_loss

        for i in range(self.num_layers):
            h_1 = map_to_layer(h_1_context, i, id_maps, node_id_map, graph)
            h_2 = h_2_layers[i]
            pos_feat = torch.cat(
                (h_1[edge_index[i][0]], h_1[edge_index[i][1]]),
                dim=1,
            )
            neg_feat = torch.cat(
                (h_2[edge_neg_index[i][0]], h_2[edge_neg_index[i][1]]),
                dim=1,
            )

            if pos_feat.size(0) == 0:
                c = h_1.new_zeros((1, self.in_channels * 2))
            elif self.readout == 'SAGPool':
                c = self.readout_func(h_1, edge_index[i])
            else:
                c = self.readout_func(pos_feat)
            c = self.readout_act_func(c)
            logits.append(self.disc(c, pos_feat, neg_feat, samp_bias1, samp_bias2))

        result['logits'] = logits
        if self.is_supervised:
            y_pred = []
            y_true = []
            latent = []
            if llm is not None:
                llm = llm.to(device)
            entity_index_map = {
                (row['id'], row['type']): index
                for index, row in node_id_map.reset_index(drop=True).iterrows()
            }
            classifier_embeddings = MyDataset(
                data=data,
                emb=self.latent_proj(self.H),
                emd2=llm if self.strategy == "topo" else None,
                entity_index_map=entity_index_map,
            )
            data_loader = DataLoader(
                classifier_embeddings,
                batch_size=batch_size,
                shuffle=False,
            )
            for x1_batch, x2_batch, y_batch in data_loader:
                x1_batch = x1_batch.to(device)
                x2_batch = x2_batch.to(device)
                y_batch = y_batch.to(device)
                outputs, batch_latent = self.classifier_chain.forward_with_representation(
                    x1_batch,
                    x2_batch,
                )
                if outputs.shape != y_batch.shape:
                    raise ValueError(
                        f"Classifier output shape {outputs.shape} does not match label shape {y_batch.shape}."
                    )
                y_pred.append(outputs)
                y_true.append(y_batch)
                latent.append(batch_latent)

            if not y_pred:
                raise ValueError("Cannot classify an empty interaction dataset.")
            result['y_pred'] = torch.cat(y_pred, dim=0)
            result['y_true'] = torch.cat(y_true, dim=0)
            result['latent'] = torch.cat(latent, dim=0)

        return result