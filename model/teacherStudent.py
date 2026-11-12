import torch
import random
import numpy as np
from .CAZI import CAZI
from .MultiLabelClassifier import MultiLabelClassifier


class TeacherStudent:
    def __init__(self, seed, node_id_map, data, device=None, mode='transductive', gnn='transformer', is_supervised=True,
                 n2v=False, batch_size=256, readout='SAGPool', num_layers=6, in_channels=128, order=5, reg_coef=1.0,
                 cls_coef=1.0, lr=1e-4,
                 l2_coef=1e-4, llm=None, cazi_path="best_cazi_.pth", ban_path="best_ban.pth", cls_loss='mls',
                 inner_runs=2, strategy='topo', nheads=4, sampling=False):

        self.seed = seed
        self.cazi_teacher = True
        self.mode = mode

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        random.seed(seed)

        if device is None:
            self.device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.cazi = CAZI(seed, node_id_map, data, device=self.device, mode=mode, gnn=gnn,
                         is_supervised=is_supervised,
                         n2v=n2v, batch_size=batch_size, readout=readout, num_layers=num_layers,
                         in_channels=in_channels, order=order, reg_coef=reg_coef, cls_coef=cls_coef, lr=lr,
                         l2_coef=l2_coef, llm=llm, path=cazi_path, cls_loss=cls_loss, strategy=strategy,
                         nheads=nheads, sampling=sampling)
        self.mlc = MultiLabelClassifier(seed, data, device=self.device, num_layers=num_layers,
                                        batch_size=batch_size, mode=mode, lr=lr, l2_coef=l2_coef,
                                        path=ban_path, cls_loss=cls_loss, llm=llm)

        # Both models must train and distill on the exact same ordered interactions.
        self.train_data = self.cazi.train_data
        self.val_data = self.cazi.val_data
        self.test_data = self.cazi.test_data
        self.mlc.train_data = self.train_data
        self.mlc.val_data = self.val_data
        self.mlc.test_data = self.test_data
        if self.mode == 'inductive':
            self.mlc.train_data_ind = self.train_data

        self.ban_path = ban_path
        self.cazi_path = cazi_path
        self.inner_runs = inner_runs

    def train(self, epochs=1000):
        cazi_best_val_auroc = float('-inf')
        ban_best_val_auroc = float('-inf')
        cazi_best_state = None
        ban_best_state = None
        for outer_epoch in range(epochs):
            print(f"Outer Epoch {outer_epoch}, Teacher: {'CAZI' if self.cazi_teacher else 'Chain'}")

            for inner_epoch in range(self.inner_runs):
                print(f"- Inner Epoch {inner_epoch}")

                teacher_latents, cazi_auroc, cazi_auprc = self.cazi.co_train(is_teacher=True)
                _, ban_auroc, ban_auprc = self.mlc.co_train(
                    is_teacher=False,
                    helper_logits=teacher_latents,
                )

                if cazi_auroc > cazi_best_val_auroc:
                    cazi_best_val_auroc = cazi_auroc
                    cazi_best_state = {
                        name: value.detach().clone()
                        for name, value in self.cazi.model.state_dict().items()
                    }
                    torch.save(self.cazi.model.state_dict(), self.cazi_path)
                if ban_auroc > ban_best_val_auroc:
                    ban_best_val_auroc = ban_auroc
                    ban_best_state = {
                        name: value.detach().clone()
                        for name, value in self.mlc.model.state_dict().items()
                    }
                    torch.save(self.mlc.model.state_dict(), self.ban_path)

            self.cazi_teacher = True

        if cazi_best_state is not None:
            self.cazi.model.load_state_dict(cazi_best_state)
        if ban_best_state is not None:
            self.mlc.model.load_state_dict(ban_best_state)

        if epochs > 0 and self.inner_runs > 0 and (
            cazi_best_state is None or ban_best_state is None
        ):
            raise RuntimeError(
                "Co-training completed without a finite validation AUROC for both models."
            )

    def offline_train(self, epochs=10000):
        self.cazi_teacher = True
        self.cazi.train(epochs=epochs, patience_limit=30)

        best_val_auroc = float('-inf')
        best_state = None
        patience_counter = 0
        for epoch in range(epochs):
            print(f"Epoch {epoch}")
            teacher_latents, _, _ = self.cazi.supervise()
            _, val_auroc, _ = self.mlc.co_train(
                is_teacher=False,
                helper_logits=teacher_latents,
            )
            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                patience_counter = 0
                best_state = {
                    name: value.detach().clone()
                    for name, value in self.mlc.model.state_dict().items()
                }
                torch.save(self.mlc.model.state_dict(), self.ban_path)
            else:
                patience_counter += 1
                if patience_counter >= 30:
                    break

        if best_state is not None:
            self.mlc.model.load_state_dict(best_state)
        elif epochs > 0:
            raise RuntimeError("Student distillation completed without a finite validation AUROC.")

    def test(self):
        self.cazi.model.load_state_dict(torch.load(self.cazi_path, map_location=self.device))
        self.mlc.model.load_state_dict(torch.load(self.ban_path, map_location=self.device))

        cazi_auroc, cazi_auprc = self.cazi.test()
        ban_auroc, ban_auprc = self.mlc.test()

        return cazi_auroc, cazi_auprc, ban_auroc, ban_auprc