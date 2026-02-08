import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from utilities.helpers import MyDataset
from utilities.dataSplitter import DataSplitter
from module.loss import FocalLoss, KDLoss
from module.MoEChain import MoE



class MultiLabelClassifier:
    def __init__(self, seed, data, device=None, num_layers=6, batch_size=256, mode='inductive', lr=1e-3, l2_coef=1e-3,
                 path="best_model.pth", cls_loss='bce', llm=None):
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.seed = seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self.mode = mode
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.lr = lr
        self.l2_coef = l2_coef
        self.path = path
        self.cls_loss = cls_loss

        if self.mode == "transductive":
            splitter = DataSplitter(data, mode=mode, seed=self.seed)
            self.train_data, self.val_data, self.test_data = splitter.split()
        else:
            splitter = DataSplitter(data, mode='inductive', seed=self.seed)
            self.train_data_ind, self.val_data, self.test_data = splitter.split()
            trans_splitter = DataSplitter(self.train_data_ind, mode='transductive', seed=self.seed)
            self.train_data, self.val_data_trans, self.test_data_trans = trans_splitter.split()

        if isinstance(llm, torch.Tensor):
            self.llm = llm.to(self.device)
        else:
            self.llm = torch.tensor(llm.to_numpy()).float().to(self.device)

        self.in_channels = self.llm.shape[1]
        self.model = MoE(self.num_layers, self.in_channels).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.l2_coef)
        self.focal_loss = FocalLoss(gamma=2, alpha=0.25).to(self.device)
        self.b_xent = nn.BCEWithLogitsLoss()

    def train(self, epochs=1000, patience_limit=10):
        best_val_auroc = 0
        patience_counter = 0
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.95)

        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            result = self.model(self.device, self.train_data, self.llm, self.batch_size)

            if self.cls_loss == "bce":
                loss = self.b_xent(result['y_pred'], result['y_true'])
            else:
                loss = self.focal_loss(result['y_pred'], result['y_true'])

            loss.backward()
            self.optimizer.step()
            scheduler.step()

            val_auroc, val_auprc = self.evaluate(self.val_data)
            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                patience_counter = 0
                torch.save(self.model.state_dict(), self.path)
            else:
                patience_counter += 1

            if patience_counter >= patience_limit:
                print(f"Early stopping at epoch {epoch}")
                break

            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Val AUROC: {val_auroc:.4f}, Val AUPRC: {val_auprc:.4f}")

    def evaluate(self, data):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.device, data, self.llm, self.batch_size)
        auroc = roc_auc_score(outputs['y_true'].cpu(), outputs['y_pred'].cpu(), average='samples')
        auprc = average_precision_score(outputs['y_true'].cpu(), outputs['y_pred'].cpu(), average='samples')
        return auroc, auprc

    def test(self):
        self.model.load_state_dict(torch.load(self.path))
        auroc, auprc = self.evaluate(self.test_data)
        print(f"Test AUROC: {auroc:.4f}, Test AUPRC: {auprc:.4f}")
        return auroc, auprc

    def co_train(self, is_teacher=False, helper_logits=None):
        self.model.train()
        self.optimizer.zero_grad()
        result = self.model(self.device, self.train_data, self.llm, self.batch_size)

        if self.cls_loss == "bce":
            loss = self.b_xent(result['y_pred'], result['y_true'])
        else:
            loss = self.focal_loss(result['y_pred'], result['y_true'])

        kd_loss = KDLoss()
        if helper_logits is None:
            total_loss = kd_loss(result['y_pred'], helper_logits, loss, is_teacher)
        else:
            total_loss = kd_loss(result['y_pred'], helper_logits.to(self.device), loss, is_teacher)

        total_loss.backward()
        self.optimizer.step()

        val_auroc_trans, val_auprc_trans = self.evaluate(self.val_data_trans)
        val_auroc, val_auprc = self.evaluate(self.val_data)
        logits, _, _ = self.co_evaluate(self.train_data)

        print(f"-- MoE: Loss:{total_loss.item():.4f}, Trans Val AUROC: {val_auroc_trans:.4f}, Trans Val AUPRC: {val_auprc_trans:.4f}, Val AUROC: {val_auroc:.4f}, Val AUPRC: {val_auprc:.4f}")
        return logits, val_auroc, val_auroc

    def co_evaluate(self, data):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.device, data, self.llm, self.batch_size)
        auroc = roc_auc_score(outputs['y_true'].cpu(), outputs['y_pred'].cpu(), average='samples')
        auprc = average_precision_score(outputs['y_true'].cpu(), outputs['y_pred'].cpu(), average='samples')
        return outputs['y_pred'].cpu(), auroc, auprc

    def finetune(self, epochs=150, patience_limit=50):
        best_val_auroc = 0
        patience_counter = 0
        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            result = self.model(self.device, self.val_data, self.llm, self.batch_size)

            if self.cls_loss == "bce":
                loss = self.b_xent(result['y_pred'], result['y_true'])
            else:
                loss = self.focal_loss(result['y_pred'], result['y_true'])

            loss.backward()
            self.optimizer.step()

            val_auroc, val_auprc = self.evaluate(self.val_data)
            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                patience_counter = 0
                torch.save(self.model.state_dict(), self.path)
            else:
                patience_counter += 1

            if patience_counter >= patience_limit:
                print(f"Early stopping at epoch {epoch}")
                break

            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Val AUROC: {val_auroc:.4f}, Val AUPRC: {val_auprc:.4f}")
