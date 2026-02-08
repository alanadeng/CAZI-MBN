import torch
import random
import numpy as np
from .CAZI import CAZI
from .MultiLabelClassifier import MultiLabelClassifier
from utilities.dataSplitter import DataSplitter


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

        if self.mode == 'transductive':
            self.cazi = CAZI(seed, node_id_map, data, device=self.device, mode=mode, gnn=gnn,
                             is_supervised=is_supervised,
                             n2v=n2v, batch_size=batch_size, readout=readout, num_layers=num_layers,
                             in_channels=in_channels, order=order, reg_coef=reg_coef, cls_coef=cls_coef, lr=lr,
                             l2_coef=l2_coef, llm=llm, path=cazi_path, cls_loss=cls_loss, strategy=strategy,
                             nheads=nheads, sampling=sampling)
            self.mlc = MultiLabelClassifier(seed, data, device=self.device, num_layers=num_layers,
                                            batch_size=batch_size, mode=mode, lr=lr, l2_coef=l2_coef,
                                            path=ban_path, cls_loss=cls_loss, llm=llm)
        else:
            splitter = DataSplitter(data, mode='inductive', seed=self.seed)
            self.train_data, self.val_data, self.test_data = splitter.split()
            self.cazi = CAZI(seed, node_id_map, self.train_data, device=self.device, mode="transductive", gnn=gnn,
                             is_supervised=is_supervised, n2v=n2v, batch_size=batch_size, readout=readout,
                             num_layers=num_layers,
                             in_channels=in_channels, order=order, reg_coef=reg_coef, cls_coef=cls_coef, lr=lr,
                             l2_coef=l2_coef, llm=llm, path=cazi_path, cls_loss=cls_loss, strategy=strategy,
                             nheads=nheads, sampling=sampling)
            self.mlc = MultiLabelClassifier(seed, data, device=self.device, num_layers=num_layers,
                                            batch_size=batch_size, mode='inductive', lr=lr, l2_coef=l2_coef,
                                            path=ban_path, cls_loss=cls_loss, llm=llm)

        self.ban_path = ban_path
        self.cazi_path = cazi_path
        self.inner_runs = inner_runs

    def train(self, epochs=1000):
        for outer_epoch in range(epochs):
            cazi_best_val_auroc = 0
            ban_best_val_auroc = 0

            print(f"Outer Epoch {outer_epoch}, Teacher: {'CAZI' if self.cazi_teacher else 'Chain'}")

            for inner_epoch in range(self.inner_runs):
                print(f"- Inner Epoch {inner_epoch}")

                if self.cazi_teacher:
                    teacher_latents, cazi_auroc, cazi_auprc = self.cazi.co_train(is_teacher=True)
                    _, ban_auroc, ban_auprc = self.mlc.co_train(is_teacher=False, helper_logits=teacher_latents)
                else:
                    teacher_latents, ban_auroc, ban_auprc = self.mlc.co_train(is_teacher=True)
                    _, cazi_auroc, cazi_auprc = self.cazi.co_train(is_teacher=False, helper_logits=teacher_latents)

                if cazi_auroc > cazi_best_val_auroc:
                    cazi_best_val_auroc = cazi_auroc
                    torch.save(self.cazi.model.state_dict(), self.cazi_path)
                if ban_auroc > ban_best_val_auroc:
                    ban_best_val_auroc = ban_auroc
                    torch.save(self.mlc.model.state_dict(), self.ban_path)

            self.cazi_teacher = ban_auroc <= cazi_auroc

        if self.mode == 'inductive':
            self.mlc.finetune()

    def offline_train(self, epochs=10000):
        self.cazi_teacher = True
        self.cazi.train(epochs=epochs, patience_limit=30)

        for epoch in range(epochs):
            print(f"Epoch {epoch}")
            teacher_latents, _, _ = self.cazi.supervise()
            self.mlc.co_train(is_teacher=False, helper_logits=teacher_latents)

        if self.mode == 'inductive':
            self.mlc.model.load_state_dict(torch.load(self.ban_path))
            self.mlc.finetune(epochs=300)

    def test(self):
        self.cazi.model.load_state_dict(torch.load(self.cazi_path))
        self.mlc.model.load_state_dict(torch.load(self.ban_path))

        cazi_auroc, cazi_auprc = self.cazi.test()
        ban_auroc, ban_auprc = self.mlc.test()

        print(f"CAZI: Test AUROC: {cazi_auroc}, Test AUPRC: {cazi_auprc}")
        print(f"MoE: Test AUROC: {ban_auroc}, Test AUPRC: {ban_auprc}")

        return cazi_auroc, cazi_auprc, ban_auroc, ban_auprc