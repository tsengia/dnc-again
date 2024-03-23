from typing import List
import torch
import torch.utils.data
from Util import split_tensor, oneplus

_EPS = 1e-6


class TemporalMemoryLinkage(torch.nn.Module):
    def __init__(self):
        super(TemporalMemoryLinkage, self).__init__()
        self.temp_link_mat = None
        self.precedence_weighting = None
        self.diag_mask = None

        self.initial_temp_link_mat = None
        self.initial_precedence_weighting = None
        self.initial_diag_mask = None
        self.initial_shape = None

    def new_sequence(self):
        self.temp_link_mat = None
        self.precedence_weighting = None
        self.diag_mask = None

    def _init_link(self, w_dist):
        s = list(w_dist.size())
        if self.initial_shape is None or s != self.initial_shape:
            self.initial_temp_link_mat = torch.zeros(s[0], s[-1], s[-1]).to(w_dist.device)
            self.initial_precedence_weighting = torch.zeros(s[0], s[-1]).to(w_dist.device)
            self.initial_diag_mask = (1.0 - torch.eye(s[-1]).unsqueeze(0).to(w_dist)).detach()

        self.temp_link_mat = self.initial_temp_link_mat
        self.precedence_weighting = self.initial_precedence_weighting
        self.diag_mask = self.initial_diag_mask

    def _update_precedence(self, w_dist):
        # w_dist shape: [ batch, cell count ]
        self.precedence_weighting = (1.0 - w_dist.sum(-1, keepdim=True)) * self.precedence_weighting + w_dist

    def _update_links(self, w_dist):
        if self.temp_link_mat is None:
            self._init_link(w_dist)

        wt_i = w_dist.unsqueeze(-1)
        wt_j = w_dist.unsqueeze(-2)
        pt_j = self.precedence_weighting.unsqueeze(-2)

        self.temp_link_mat = ((1 - wt_i - wt_j) * self.temp_link_mat + wt_i * pt_j) * self.diag_mask

    def forward(self, w_dist, prev_r_dists, debug = None):
        self._update_links(w_dist)
        self._update_precedence(w_dist)

        # prev_r_dists shape: [ batch, n heads, cell count ]
        # Emulate matrix-vector multiplication by broadcast and sum. This way we don't need to transpose the matrix
        tlm_multi_head = self.temp_link_mat.unsqueeze(1)

        forward_dist = (tlm_multi_head * prev_r_dists.unsqueeze(-2)).sum(-1)
        backward_dist = (tlm_multi_head * prev_r_dists.unsqueeze(-1)).sum(-2)

        # output shapes [ batch, n_heads, cell_count ]
        return forward_dist, backward_dist



class DistSharpnessEnhancer(torch.nn.Module):
    def __init__(self, n_heads: int | List[int]):
        super(DistSharpnessEnhancer, self).__init__()
        self.n_heads = n_heads if isinstance(n_heads, list) else [n_heads]
        self.n_data = sum(self.n_heads)

    def forward(self, nn_input, *dists):
        assert len(dists) == len(self.n_heads)
        nn_input = oneplus(nn_input[..., :self.n_data])
        factors = split_tensor(nn_input, self.n_heads)

        res = []
        for i, d in enumerate(dists):
            ndim = d.dim()
            f  = factors[i]
            if ndim==2:
                assert self.n_heads[i]==1
            elif ndim==3:
                f = f.unsqueeze(-1)
            else:
                assert False

            d += _EPS
            d = d / d.max(dim=-1, keepdim=True)[0]
            d = d.pow(f)
            d = d / d.sum(dim=-1, keepdim=True)
            res.append(d)
        return res

    def get_neural_input_size(self):
        return self.n_data