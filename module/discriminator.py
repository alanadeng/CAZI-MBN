import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k_bilinear = nn.Bilinear(n_h, n_h, 1)
        self.norm = nn.LayerNorm(n_h)
        self.activation = nn.PReLU()

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.kaiming_uniform_(m.weight.data, a=0.01)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c.expand_as(h_pl)  # Expand c to match the dimensions of h_pl

        h_pl = self.norm(h_pl)
        h_mi = self.norm(h_mi)
        h_pl = self.activation(h_pl)
        h_mi = self.activation(h_mi)

        sc_1 = torch.squeeze(self.f_k_bilinear(h_pl, c_x), 1)
        sc_2 = torch.squeeze(self.f_k_bilinear(h_mi, c_x), 1)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2
        logits = torch.cat((sc_1, sc_2), 0)

        return logits
