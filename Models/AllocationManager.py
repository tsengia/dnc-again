
import torch
import torch.utils.data
import torch.nn.functional as F

_EPS = 1e-6

class AllocationManager(torch.nn.Module):
    def __init__(self):
        super(AllocationManager, self).__init__()
        self.usages = None
        self.zero_usages = None
        self.debug_sequ_init = False
        self.one = None

    def _init_sequence(self, prev_read_distributions):
        # prev_read_distributions size is [batch, n_heads, cell count]
        s = prev_read_distributions.size()
        if self.zero_usages is None or list(self.zero_usages.size())!=[s[0],s[-1]]:
            self.zero_usages = torch.zeros(s[0], s[-1], device = prev_read_distributions.device)
            if self.debug_sequ_init:
                self.zero_usages += torch.arange(0, s[-1]).unsqueeze(0) * 1e-10

        self.usages = self.zero_usages

    def _init_consts(self, device):
        if self.one is None:
            self.one = torch.ones(1, device=device)

    def new_sequence(self):
        self.usages = None

    def update_usages(self, prev_write_distribution, prev_read_distributions, free_gates):
        # Read distributions shape: [batch, n_heads, cell count]
        # Free gates shape: [batch, n_heads]

        self._init_consts(prev_read_distributions.device)
        phi = torch.addcmul(self.one, free_gates.unsqueeze(-1), prev_read_distributions, value=-1).prod(-2)
        # Phi is the free tensor, sized [batch, cell count]

        # If memory usage counter if doesn't exists
        if self.usages is None:
            self._init_sequence(prev_read_distributions)
            # in first timestep nothing is written or read yet, so we don't need any further processing
        else:
            self.usages = torch.addcmul(self.usages, prev_write_distribution.detach(), (1 - self.usages), value=1) * phi

        return phi

    def forward(self, prev_write_distribution, prev_read_distributions, free_gates):
        phi = self.update_usages(prev_write_distribution, prev_read_distributions, free_gates)
        sorted_usage, free_list = (self.usages*(1.0-_EPS)+_EPS).sort(-1)

        u_prod = sorted_usage.cumprod(-1)
        one_minus_usage = 1.0 - sorted_usage
        sorted_scores = torch.cat([one_minus_usage[..., 0:1], one_minus_usage[..., 1:] * u_prod[..., :-1]], dim=-1)

        return sorted_scores.clone().scatter_(-1, free_list, sorted_scores), phi


class ContentAddressGenerator(torch.nn.Module):
    def __init__(self, disable_content_norm: bool = False, mask_min: float = 0.0, disable_key_masking: bool = False):
        super(ContentAddressGenerator, self).__init__()
        self.disable_content_norm = disable_content_norm
        self.mask_min = mask_min
        self.disable_key_masking = disable_key_masking

    def forward(self, memory, keys, betas, mask=None):
        # Memory shape [batch, cell count, word length]
        # Key shape [batch, n heads*, word length]
        # Betas shape [batch, n heads]
        if mask is not None and self.mask_min != 0:
            mask = mask * (1.0-self.mask_min) + self.mask_min

        single_head = keys.dim() == 2
        if single_head:
            # Single head
            keys = keys.unsqueeze(1)
            if mask is not None:
                mask = mask.unsqueeze(1)

        memory = memory.unsqueeze(1)
        keys = keys.unsqueeze(-2)

        if mask is not None:
            mask = mask.unsqueeze(-2)
            memory = memory * mask
            if not self.disable_key_masking:
                keys = keys * mask

        # Shape [batch, n heads, cell count]
        norm = keys.norm(dim=-1)
        if not self.disable_content_norm:
            norm = norm * memory.norm(dim=-1)

        scores = (memory * keys).sum(-1) / (norm + _EPS)
        scores *= betas.unsqueeze(-1)

        res = F.softmax(scores, scores.dim()-1)
        return res.squeeze(1) if single_head else res