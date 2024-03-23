
import torch
import torch.utils.data
import torch.nn.functional as F
import torch.nn.init as init
import functools
import math

import lightning as L

def oneplus(t):
    return F.softplus(t, 1, 20) + 1.0

def get_next_tensor_part(src, dims, prev_pos=0):
    if not isinstance(dims, list):
        dims=[dims]
    n = functools.reduce(lambda x, y: x * y, dims)
    data = src.narrow(-1, prev_pos, n)
    return data.contiguous().view(list(data.size())[:-1] + dims) if len(dims)>1 else data, prev_pos + n

def split_tensor(src, shapes):
    pos = 0
    res = []
    for s in shapes:
        d, pos = get_next_tensor_part(src, s, pos)
        res.append(d)
    return res

def dict_get(dict,name):
    return dict.get(name) if dict is not None else None


def dict_append(dict, name, val):
    if dict is not None:
        l = dict.get(name)
        if not l:
            l = []
            dict[name] = l
        l.append(val)


def init_debug(debug, initial):
    if debug is not None and not debug:
        debug.update(initial)

def merge_debug_tensors(d, dim):
    if d is not None:
        for k, v in d.items():
            if isinstance(v, dict):
                merge_debug_tensors(v, dim)
            elif isinstance(v, list):
                d[k] = torch.stack(v, dim)


def linear_reset(module, gain=1.0):
    assert isinstance(module, torch.nn.Linear)
    init.xavier_uniform_(module.weight, gain=gain)
    s = module.weight.size(1)
    if module.bias is not None:
        module.bias.data.zero_()
