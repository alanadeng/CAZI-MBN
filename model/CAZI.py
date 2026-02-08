import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch.utils.data import DataLoader
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from utilities.helpers import map_to_original, MyDataset, get_llm_features, concatenate_embedding
from utilities.dataSplitter import DataSplitter
from utilities.graphHandler import GraphHandler
from utilities.edgeExtractor import EdgeExtractor
from utilities.layerProcessor import LayerProcessor
from module.topologyAwareProjection import TopologyAwareFeatureProjection
from module.graphTransformer import GraphTransformer
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
        self.node_id_map = node_id_map
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
        self.num_nodes = len(np.unique(self.data['drug'].values)) + len(np.unique(self.data['gene'].values))
        splitter = DataSplitter(data, mode=mode, seed=self.seed)
        self.train_data, self.val_data, self.test_data = splitter.split()

        graph_handler = GraphHandler(self.train_data, self.val_data, test_data=self.test_data, num_layer=self.num_layers)
        self.graph = graph_handler.generate_graph()

        edge_extractor = EdgeExtractor(self.graph, seed=self.seed)
        train_edges, val_edges, test_edges = edge_extractor.generate_edges()

        network_extractor = TopologyAwareFeatureProjection(train_edges, self.in_channels, order)
        self.embeddings = network_extractor.get_node_features()

        self.llm = llm

        self.strategy = strategy
        if self.strategy == "llm_only":
            llm_embeddings = get_llm_features(self.graph, self.llm)
            self.embeddings = llm_embeddings
            self.llm = torch.tensor(self.llm).to(self.device)
            self.in_channels = self.llm.shape[1]
            dim_llm = 0
        elif self.strategy == "llm_topo":
            llm_embeddings = get_llm_features(self.graph, self.llm)
            self.embeddings = concatenate_embedding(self.embeddings, llm_embeddings)
            # self.llm = torch.tensor(self.llm.to_numpy()).to(self.device)
            # self.llm.fillna(0, inplace=True)
            # self.llm = torch.tensor(self.llm.values, dtype=torch.float).to(self.device)
            llm_df = self.llm.copy()
            llm_df = llm_df.fillna(0)
            self.llm = torch.tensor(llm_df.to_numpy(), dtype=torch.float, device=self.device)
            self.in_channels = self.in_channels + self.llm.shape[1]
            dim_llm = 0
        else:
            if self.llm is None:
                dim_llm = 0
            else:
                self.embeddings = self.embeddings
                self.llm = torch.tensor(self.llm.to_numpy()).to(self.device)
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
                         self.readout_act_func, gnn=self.gnn, is_supervised=self.is_supervised, dim_llm=dim_llm, nheads=nheads, strategy=self.strategy, sampling=sampling).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.l2_coef)

        self.x_list, self.train_edges_list, self.val_edges_list, self.test_edges_list, self.train_edges_neg, self.val_edges_neg, self.test_edges_neg = self.prepare_data()

        self.focal_loss = FocalLoss(gamma=2, alpha=0.25).to(self.device)
        self.mlsml = MultiLabelSoftMarginLoss()

    def negative_sampling(self, edge_list):
        src_nodes = [edge[0] for edge in edge_list]
        dst_nodes = [edge[1] for edge in edge_list]
        positive_edges = set(edge_list)
        num_positive = len(positive_edges)
        all_nodes = list(set(src_nodes).union(set(dst_nodes)))

        negative_edges = set()

        while len(negative_edges) < num_positive:
            src = random.choice(all_nodes)
            dst = random.choice(all_nodes)
            if src != dst and (src, dst) not in positive_edges and (src, dst) not in negative_edges:
                negative_edges.add((src, dst))

        return list(negative_edges)
    
    def prepare_data(self):
        x_list = []
        train_edges_list, val_edges_list, test_edges_list = [], [], []
        train_edges_neg, val_edges_neg, test_edges_neg = [], [], []

        for layer in range(len(self.dataset)):
            feats = self.dataset[layer][0]
            x_vals = [float(feats[i]) for i in sorted(feats) if isinstance(feats[i], (int, float))]
            x = torch.tensor(np.array(x_vals, dtype=float), dtype=torch.float).to(self.device)
            x_list.append(x)

            train_edges = list(set(self.dataset[layer][1]))
            val_edges   = list(set(self.dataset[layer][2]))
            test_edges  = list(set(self.dataset[layer][3]))

            train_edges_list.append(torch.tensor(train_edges, dtype=torch.long).t().contiguous().to(self.device))
            val_edges_list.append(torch.tensor(val_edges, dtype=torch.long).t().contiguous().to(self.device))
            test_edges_list.append(torch.tensor(test_edges, dtype=torch.long).t().contiguous().to(self.device))

            train_edges_neg.append(torch.tensor(self.negative_sampling(train_edges), dtype=torch.long).t().contiguous().to(self.device))
            val_edges_neg.append(torch.tensor(self.negative_sampling(val_edges), dtype=torch.long).t().contiguous().to(self.device))
            test_edges_neg.append(torch.tensor(self.negative_sampling(test_edges), dtype=torch.long).t().contiguous().to(self.device))

        return x_list, train_edges_list, val_edges_list, test_edges_list, train_edges_neg, val_edges_neg, test_edges_neg


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
        b_xent = nn.BCEWithLogitsLoss()  # discriminator

        best_val_auroc = 0
        patience_counter = 0

        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=75, gamma=0.75)

        for epoch in range(epochs):
            xent_loss = None
            self.model.train()
            self.optimizer.zero_grad()

            result = self.model(self.device, self.train_data, self.llm, self.batch_size, self.x_list, self.train_edges_list, self.train_edges_neg,
                                self.id_map, self.node_id_map, self.graph, None, None)
            logits = result['logits']

            loss = 0
            for view_idx, logit in enumerate(logits):
                lbl = torch.cat((torch.ones(int(len(logit) / 2)), torch.zeros(int(len(logit) / 2)))).to(self.device)
                if xent_loss is None:
                    xent_loss = b_xent(logit.to(self.device), lbl)
                else:
                    xent_loss += b_xent(logit.to(self.device), lbl)

            loss += xent_loss
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
                    torch.save(self.model.state_dict(), self.path)
                else:
                    patience_counter += 1
                if patience_counter >= patience_limit:
                    print(f"Stopping early at epoch {epoch}")
                    break
                print(f"Epoch {epoch}, Loss:{loss.item():.4f}, Discriminator: {xent_loss.item():.4f}, Consensus: {reg_loss.item():.4f}, Classifier:{supervised_loss.item():.4f}, Val AUROC: {val_auroc:.4f}, Val AUPRC: {val_auprc:.4f}")
            else:
                loss.backward()
                self.optimizer.step()
                scheduler.step()
                print(f"Epoch {epoch}, Loss:{loss.item():.4f}, Discriminator: {xent_loss.item():.4f}, Consensus: {reg_loss.item():.4f}")



    def evaluate(self, data, edge_list, edge_neg_list):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.device, data, self.llm, self.batch_size, self.x_list, edge_list, edge_neg_list, self.id_map, self.node_id_map, self.graph, None, None)
        auroc = roc_auc_score(outputs['y_true'].cpu().detach(), outputs['y_pred'].cpu().detach(), average='samples')
        auprc = average_precision_score(outputs['y_true'].cpu().detach(), outputs['y_pred'].cpu().detach(), average='samples')
        return auroc, auprc

    def test(self):
        if self.is_supervised:
            self.model.load_state_dict(torch.load(self.path))
            auroc, auprc = self.evaluate(self.test_data, self.test_edges_list, self.test_edges_neg)
            print(f"Test AUROC: {auroc}, Test AUPRC: {auprc}")
            return auroc, auprc
        else:
            ban_model = MultiLabelClassifier(self.seed, self.data, mode='transductive', llm=self.model.H, lr=0.01)
            ban_model.train(epochs=100, patience_limit=10)
            ban_model.test()

    def co_train(self, is_teacher=False, helper_vectors=None):
        self.model.train()
        self.optimizer.zero_grad()

        result = self.model(self.device, self.train_data, self.llm, self.batch_size,
                            self.x_list, self.train_edges_list, self.train_edges_neg,
                            self.id_map, self.node_id_map, self.graph, None, None)

        logits = result['logits']
        loss = sum(self.mlsml(logit.to(self.device), torch.cat([
            torch.ones(len(logit) // 2),
            torch.zeros(len(logit) // 2)
        ]).to(self.device)) for logit in logits)

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
                                 self.x_list, edge_list, self.shuf_x_list,
                                 self.id_map, self.node_id_map, self.graph, None, None)
        auroc = roc_auc_score(outputs['y_true'].cpu().detach(), outputs['y_pred'].cpu().detach(), average='samples')
        auprc = average_precision_score(outputs['y_true'].cpu().detach(), outputs['y_pred'].cpu().detach(),
                                        average='samples')
        return outputs['latent'].cpu().detach(), auroc, auprc

    def supervise(self):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.device, self.train_data, self.llm, self.batch_size,
                                 self.x_list, self.train_edges_list, self.shuf_x_list,
                                 self.id_map, self.node_id_map, self.graph, None, None)
        auroc = roc_auc_score(outputs['y_true'].cpu().detach(), outputs['y_pred'].cpu().detach(), average='samples')
        auprc = average_precision_score(outputs['y_true'].cpu().detach(), outputs['y_pred'].cpu().detach(),
                                        average='samples')
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
            GraphTransformer(in_channels, int(in_channels * 2), in_channels)
            for _ in range(self.num_layers)
        ])

        self.disc = Discriminator(in_channels * 2)
        self.H = nn.Parameter(torch.FloatTensor(self.num_nodes, self.in_channels))
        self.latent_proj = nn.Linear(self.in_channels, self.latent_channels)
        self.attn = nn.MultiheadAttention(self.in_channels, self.nheads)

        # if self.is_supervised:
        #     self.classifier_chain = MoE(self.in_channels + dim_llm, self.in_channels + dim_llm, self.num_layers)
        if self.is_supervised:
            in2_dim = dim_llm if self.strategy == "topo" else self.in_channels

            self.classifier_chain = MoE(
                in1_dim=self.in_channels,
                in2_dim=in2_dim,
                out_dim=self.num_layers,
                num_experts=self.num_layers
            )


        self.init_weight()

    def init_weight(self):
        nn.init.xavier_normal_(self.H)

    def node_level_attention(self, H):
        N, L, d = H.shape
        H_new = torch.zeros_like(H)

        for p in range(L):
            h_p = H[:, p, :]
            theta_p = self.theta[p]
            scores = []

            for q in range(L):
                if q == p:
                    scores.append(torch.full((N,), float('-inf'), device=H.device))
                    continue
                h_q = H[:, q, :]
                score = torch.sigmoid((h_p * h_q * theta_p).sum(dim=-1))
                scores.append(score)

            scores_tensor = torch.stack(scores, dim=1)
            attn_weights = F.softmax(scores_tensor, dim=1)

            weighted_sum = sum(
                attn_weights[:, q].unsqueeze(-1) * H[:, q, :]
                for q in range(L) if q != p
            )
            H_new[:, p, :] = weighted_sum

        return H_new

    def forward(self, device, data, llm, batch_size, x, edge_index, edge_neg_index, id_maps, node_id_map, graph, samp_bias1, samp_bias2):
        h_1_all = [];
        h_2_all = [];
        c_all = [];
        logits = []
        result = {}

        for i in range(self.num_layers):
            if self.sampling == True:
                sampling_data = Data(x=x[i], edge_index=edge_index[i])
                # Setup NeighborLoader
                loader = NeighborLoader(
                    sampling_data,
                    num_neighbors=[15] * self.num_layers,
                    batch_size=8,
                    shuffle=True,
                    drop_last=False,
                )
                sampling_neg_data = Data(x=x[i], edge_index=edge_neg_index[i])
                neg_loader = NeighborLoader(
                    sampling_neg_data ,
                    num_neighbors=[15] * self.num_layers,
                    batch_size=8,
                    shuffle=True,
                    drop_last=False,
                )
                h_1 = torch.zeros((x[i].size(0), self.in_channels), device=x[i].device)
                h_2 = torch.zeros((x[i].size(0), self.in_channels), device=x[i].device)
                for batch in loader:
                    emd = self.gnn[i](batch.x, batch.edge_index)
                    h_1[batch.n_id] = emd.detach()
                for neg_batch in neg_loader:
                    emd_neg = self.gnn[i](neg_batch.x, neg_batch.edge_index)
                    h_2[neg_batch.n_id] = emd_neg.detach()
            else:
                h_1 = self.gnn[i](x[i], edge_index[i])
                h_2 = self.gnn[i](x[i], edge_neg_index[i])
            pos_feat = torch.cat((h_1[edge_index[i][0, :], :], h_1[edge_index[i][1, :], :]), dim=1)
            neg_feat = torch.cat((h_2[edge_neg_index[i][0, :], :], h_2[edge_neg_index[i][1, :], :]), dim=1)
    
            if self.readout == 'SAGPool':
                c = self.readout_func(h_1, edge_index[i])
            else:
                c = self.readout_func(pos_feat)
            c = self.readout_act_func(c)
            logit = self.disc(c, pos_feat, neg_feat, samp_bias1, samp_bias2)
            h_1_all.append(h_1)
            h_2_all.append(h_2)
            c_all.append(c)
            logits.append(logit)

        h_1_all = map_to_original(h_1_all, id_maps, node_id_map, graph)
        h_1_all_ = torch.stack(h_1_all, dim=1).to(device)  # (N, L, d)
        h_1_all_ = self.node_level_attention(h_1_all_)

        h_1_attn_output, h_1_attn_output_weights = self.attn(
            h_1_all_.transpose(0, 1), h_1_all_.transpose(0, 1), h_1_all_.transpose(0, 1)
        )
        h_1_weights = h_1_attn_output_weights.mean(dim=1).mean(dim=0)
        h_1_all = (h_1_attn_output * h_1_weights.view(self.num_layers, 1, 1)).sum(dim=0)


        h_2_all = map_to_original(h_2_all, id_maps, node_id_map, graph)
        h_2_all_ = (torch.stack(h_2_all, dim=1)).to(device)
        h_2_attn_output, h_2_attn_output_weights = self.attn(h_2_all_.transpose(0, 1), h_2_all_.transpose(0, 1), h_2_all_.transpose(0, 1))
        h_2_weights = h_2_attn_output_weights.mean(dim=1).mean(dim=0)
        h_2_all = (h_2_attn_output * h_2_weights.view(self.num_layers, 1, 1)).sum(dim=0)

        cosine_loss = F.cosine_similarity(self.H, h_1_all, dim=1).mean() + 1 - F.cosine_similarity(self.H, h_2_all,dim=1).mean()
        reg_loss = cosine_loss

        if self.is_supervised:
            y_pred = []
            y_true = []
            if llm is not None:
                llm = llm.to(device)
            if not(self.strategy == "topo"):
                data_loader = DataLoader(MyDataset(data=data, emb=self.H, emd2=None), batch_size=batch_size, shuffle=True)
            else:
                data_loader = DataLoader(MyDataset(data=data, emb=self.H, emd2=llm), batch_size=batch_size, shuffle=True)
            for x1_batch, x2_batch, y_batch in data_loader:
                x1_batch = x1_batch.to(device)
                x2_batch = x2_batch.to(device)
                y_batch  = y_batch.to(device)
                outputs = self.classifier_chain(x1_batch, x2_batch)
                y_pred.append(outputs)
                y_true.append(y_batch)
            y_pred = torch.cat(y_pred, dim=0)
            y_true = torch.cat(y_true, dim=0)

            result['logits'] = logits
            result['reg_loss'] = reg_loss
            result['y_pred'] = y_pred
            result['y_true'] = y_true
            result['latent'] = self.latent_proj(self.H)

        return result