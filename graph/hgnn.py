import torch
from torch import nn
# from torch.nn import Identity     # used in residual connection, but deprecated in the latest version
import torch.nn.functional as F
from typing import Tuple

class GATedge(nn.Module):
    '''
    Machine node embedding
    '''
    def __init__(self,
                 in_feats: Tuple[int, int],
                 out_feats: int,
                 num_head: int,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 activation=None):
        '''
        :param in_feats: tuple, input dimension of (operation node, machine node)
        :param out_feats: Dimension of the output (machine embedding)
        :param num_head: Number of heads
        '''
        super(GATedge, self).__init__()
        self._num_heads = num_head  # single head is used in the actual experiment
        self._in_src_feats = in_feats[0]
        self._in_dst_feats = in_feats[1]
        self._out_feats = out_feats

        self.fc_src = nn.Linear(
            self._in_src_feats, out_feats * num_head, bias=False)
        self.fc_dst = nn.Linear(
            self._in_dst_feats, out_feats * num_head, bias=False)
        self.fc_edge = nn.Linear(
            1, out_feats * num_head, bias=False)

        self.attn_l = nn.Parameter(torch.rand(size=(1, num_head, out_feats), dtype=torch.float))
        self.attn_r = nn.Parameter(torch.rand(size=(1, num_head, out_feats), dtype=torch.float))
        self.attn_e = nn.Parameter(torch.rand(size=(1, num_head, out_feats), dtype=torch.float))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.reset_parameters()
        self.activation = activation        # not used!!!

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_edge.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_e, gain=gain)

    def forward(self, ope_ma_adj_batch: torch.Tensor, batch_idxes: torch.Tensor, feat: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        # Two linear transformations are used for the machine nodes and operation nodes, respective
        # In linear transformation, an W^O (\in R^{d \times 7}) for \mu_{ijk} is equivalent to
        #   W^{O'} (\in R^{d \times 6}) and W^E (\in R^{d \times 1}) for the nodes and edges respectively
        h_src = self.feat_drop(feat[0])
        h_dst = self.feat_drop(feat[1])
        feat_src: torch.Tensor = self.fc_src(h_src)
        feat_dst: torch.Tensor = self.fc_dst(h_dst)
        feat_edge: torch.Tensor = self.fc_edge(feat[2].unsqueeze(-1))

        # Calculate attention coefficients
        el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
        er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
        ee = (feat_edge * self.attn_l).sum(dim=-1).unsqueeze(-1)
        el_add_ee = ope_ma_adj_batch[batch_idxes].unsqueeze(-1) * el.unsqueeze(-2) + ee
        a = el_add_ee + ope_ma_adj_batch[batch_idxes].unsqueeze(-1) * er.unsqueeze(-3)
        eijk: torch.Tensor = self.leaky_relu(a)
        ekk: torch.Tensor = self.leaky_relu(er + er)

        # Normalize attention coefficients
        mask = torch.cat((ope_ma_adj_batch[batch_idxes].unsqueeze(-1)==1,
                          torch.full(size=(ope_ma_adj_batch[batch_idxes].size(0), 1,
                                           ope_ma_adj_batch[batch_idxes].size(2), 1),
                                     dtype=torch.bool, fill_value=True)), dim=-3)
        e = torch.cat((eijk, ekk.unsqueeze(-3)), dim=-3)
        e[~mask] = float('-inf')    # may use -torch.inf; here we have auto-cast
        alpha = F.softmax(e.squeeze(-1), dim=-2)
        alpha_ijk = alpha[..., :-1, :]
        alpha_kk = alpha[..., -1, :].unsqueeze(-2)

        # Calculate an return machine embedding
        Wmu_ijk = feat_edge + feat_src.unsqueeze(-2)
        a = Wmu_ijk * alpha_ijk.unsqueeze(-1)
        b = torch.sum(a, dim=-3)
        c = feat_dst * alpha_kk.squeeze().unsqueeze(-1)
        nu_k_prime = torch.sigmoid(b+c)
        return nu_k_prime

class MLPsim(nn.Module):        # MLP Simple Version?
    '''
    Part of operation node embedding
    '''
    def __init__(self,
                 in_feats,
                 out_feats,
                 hidden_dim,
                 num_head,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2):
        '''
        :param in_feats: Dimension of the input vectors of the MLPs
        :param out_feats: Dimension of the output (operation embedding) of the MLPs
        :param hidden_dim: Hidden dimensions of the MLPs
        :param num_head: Number of heads
        '''
        super(MLPsim, self).__init__()
        self._num_heads = num_head
        self._in_feats = in_feats
        self._out_feats = out_feats

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.project = nn.Sequential(
            nn.Linear(self._in_feats, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, self._out_feats),
        )

    def forward(self, feat: torch.Tensor, adj: torch.Tensor):
        # MLP_{\theta_x}, where x = 1, 2, 3, 4
        # Note that message-passing should along the edge (according to the adjacency matrix)
        a = adj.unsqueeze(-1) * feat.unsqueeze(-3)
        b = torch.sum(a, dim=-2)
        c = self.project(b)
        return c
