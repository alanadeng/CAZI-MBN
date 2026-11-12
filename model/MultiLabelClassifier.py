import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from utilities.helpers import MyDataset
from utilities.dataSplitter import DataSplitter
from module.loss import FocalLoss, KDLoss, MultiLabelSoftMarginLoss
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
        self.output_dim = len([col for col in data.columns if col not in ('drug', 'gene')])

        if self.mode == "transductive":
            splitter = DataSplitter(data, mode=mode, seed=self.seed)
            self.train_data, self.val_data, self.test_data = splitter.split()
        else:
            splitter = DataSplitter(data, mode='inductive', seed=self.seed)
            self.train_data_ind, self.val_data, self.test_data = splitter.split()
            self.train_data = self.train_data_ind

        if llm is None:
            raise ValueError("llm sequence embeddings are required by the student classifier.")
        if isinstance(llm, torch.Tensor):
            self.llm = llm.to(device=self.device, dtype=torch.float32)
        else:
            llm_values = llm.to_numpy() if hasattr(llm, 'to_numpy') else np.asarray(llm)
            self.llm = torch.as_tensor(llm_values, dtype=torch.float32, device=self.device)

        if self.llm.dim() != 2:
            raise ValueError(f"llm embeddings must be a 2D entity-by-feature tensor, got {self.llm.shape}.")
        self.in_channels = self.llm.shape[1]
        self.model = MoE(
            in1_dim=self.in_channels,
            in2_dim=self.in_channels,
            out_dim=self.output_dim,
            num_experts=self.num_layers,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.l2_coef)
        self.focal_loss = FocalLoss(gamma=2, alpha=0.25).to(self.device)
        self.b_xent = nn.BCEWithLogitsLoss()
        self.mlsml = MultiLabelSoftMarginLoss()

    def _classification_loss(self, y_pred, y_true):
        if self.cls_loss == "bce":
            return self.b_xent(y_pred, y_true)
        if self.cls_loss == "mls":
            return self.mlsml(y_pred, y_true)
        return self.focal_loss(y_pred, y_true)

    def _forward(self, data):
        dataset = MyDataset(data=data, emb=None, emd2=self.llm)
        if len(dataset) == 0:
            raise ValueError("Cannot run the classifier on an empty interaction dataset.")

        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        y_pred = []
        y_true = []
        latent = []
        for x1_batch, x2_batch, y_batch in data_loader:
            x1_batch = x1_batch.to(self.device)
            x2_batch = x2_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            logits, representation = self.model.forward_with_representation(x1_batch, x2_batch)
            y_pred.append(logits)
            y_true.append(y_batch)
            latent.append(representation)

        return {
            'y_pred': torch.cat(y_pred, dim=0),
            'y_true': torch.cat(y_true, dim=0),
            'latent': torch.cat(latent, dim=0),
        }

    def train(self, epochs=1000, patience_limit=10):
        best_val_auroc = float('-inf')
        best_state = None
        patience_counter = 0
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.95)

        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            result = self._forward(self.train_data)

            loss = self._classification_loss(result['y_pred'], result['y_true'])

            loss.backward()
            self.optimizer.step()
            scheduler.step()

            val_auroc, val_auprc = self.evaluate(self.val_data)
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
                print(f"Early stopping at epoch {epoch}")
                break

            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        if best_state is not None:
            self.model.load_state_dict(best_state)
        elif epochs > 0:
            raise RuntimeError("Student training completed without a finite validation AUROC.")

    def evaluate(self, data):
        self.model.eval()
        with torch.no_grad():
            outputs = self._forward(data)
        auroc = roc_auc_score(outputs['y_true'].cpu(), outputs['y_pred'].cpu(), average='macro')
        auprc = average_precision_score(outputs['y_true'].cpu(), outputs['y_pred'].cpu(), average='macro')
        return auroc, auprc

    def test(self):
        self.model.load_state_dict(torch.load(self.path, map_location=self.device))
        auroc, auprc = self.evaluate(self.test_data)
        return auroc, auprc

    def co_train(self, is_teacher=False, helper_logits=None):
        self.model.train()
        self.optimizer.zero_grad()
        result = self._forward(self.train_data)

        loss = self._classification_loss(result['y_pred'], result['y_true'])

        if not is_teacher and helper_logits is None:
            raise ValueError("helper_logits are required when training the student.")
        if helper_logits is not None and helper_logits.shape != result['latent'].shape:
            raise ValueError(
                f"Teacher and student representations must have matching shapes; "
                f"got {helper_logits.shape} and {result['latent'].shape}."
            )

        kd_loss = KDLoss()
        if helper_logits is None:
            total_loss = kd_loss(result['latent'], helper_logits, loss, is_teacher)
        else:
            total_loss = kd_loss(result['latent'], helper_logits.to(self.device), loss, is_teacher)

        total_loss.backward()
        self.optimizer.step()

        val_auroc, val_auprc = self.evaluate(self.val_data)

        print(f"-- MoE: Loss:{total_loss.item():.4f}")
        return result['latent'].detach(), val_auroc, val_auprc

    def co_evaluate(self, data):
        self.model.eval()
        with torch.no_grad():
            outputs = self._forward(data)
        auroc = roc_auc_score(outputs['y_true'].cpu(), outputs['y_pred'].cpu(), average='macro')
        auprc = average_precision_score(outputs['y_true'].cpu(), outputs['y_pred'].cpu(), average='macro')
        return outputs['y_pred'].cpu(), auroc, auprc

    def finetune(self, epochs=150, patience_limit=50):
        best_val_auroc = float('-inf')
        best_state = None
        patience_counter = 0
        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            result = self._forward(self.train_data)

            loss = self._classification_loss(result['y_pred'], result['y_true'])

            loss.backward()
            self.optimizer.step()

            val_auroc, val_auprc = self.evaluate(self.val_data)
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
                print(f"Early stopping at epoch {epoch}")
                break

            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        if best_state is not None:
            self.model.load_state_dict(best_state)
        elif epochs > 0:
            raise RuntimeError("Student finetuning completed without a finite validation AUROC.")
