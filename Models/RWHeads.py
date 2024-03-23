import torch
import torch.utils.data
import torch.nn.functional as F

from AllocationManager import AllocationManager, ContentAddressGenerator
from Util import split_tensor, oneplus

class WriteHead(torch.nn.Module):
    @staticmethod
    def create_write_archive(write_dist, erase_vector, write_vector, phi):
        return dict(write_dist=write_dist, erase_vector=erase_vector, write_vector=write_vector, phi=phi)

    def __init__(self, 
                 allocation_manager: AllocationManager,
                 write_content_generator: ContentAddressGenerator,
                 dealloc_content: bool = True
                ):
        super(WriteHead, self).__init__()
        self.write_content_generator = write_content_generator
        self.allocation_manager = allocation_manager
        self.last_write = None
        self.dealloc_content = dealloc_content
        self.new_sequence()

    def new_sequence(self):
        self.last_write = None
        self.allocation_manager.new_sequence()

    @staticmethod
    def mem_update(memory, write_dist, erase_vector, write_vector, phi):
        # In original paper the memory content is NOT deallocated, which makes content based addressing basically
        # unusable when multiple similar steps should be done. The reason for this is that the memory contents are
        # still there, so the lookup will find them, unless an allocation clears it before the next search, which is
        # completely random. So I'm arguing that erase matrix should also take in account the free gates (multiply it
        # with phi)
        write_dist = write_dist.unsqueeze(-1)

        erase_matrix = 1.0 - write_dist * erase_vector.unsqueeze(-2)
        if phi is not None:
            erase_matrix = erase_matrix * phi.unsqueeze(-1)

        update_matrix = write_dist * write_vector.unsqueeze(-2)
        return memory * erase_matrix + update_matrix

    def forward(self, memory, write_content_key, write_beta, erase_vector, write_vector, alloc_gate, write_gate,
                free_gates, prev_read_dist, write_mask=None, debug=None):
        last_w_dist = self.last_write["write_dist"] if self.last_write is not None else None

        content_dist = self.write_content_generator(memory, write_content_key, write_beta, mask = write_mask)
        alloc_dist, phi = self.allocation_manager(last_w_dist, prev_read_dist, free_gates)

        # Shape [batch, cell count]
        write_dist = write_gate * (alloc_gate * alloc_dist + (1-alloc_gate)*content_dist)
        self.last_write = WriteHead.create_write_archive(write_dist, erase_vector, write_vector, phi if self.dealloc_content else None)

        return WriteHead.mem_update(memory, **self.last_write)

class RawWriteHead(torch.nn.Module):
    def __init__(self, n_read_heads, word_length, use_mask=False, dealloc_content=True, disable_content_norm=False,
                 mask_min=0.0, disable_key_masking=False):
        super(RawWriteHead, self).__init__()
        self.write_head = WriteHead(dealloc_content = dealloc_content, disable_content_norm = disable_content_norm,
                                    mask_min=mask_min, disable_key_masking=disable_key_masking)
        self.word_length = word_length
        self.n_read_heads = n_read_heads
        self.use_mask = use_mask
        self.input_size = 3*self.word_length + self.n_read_heads + 3 + (self.word_length if use_mask else 0)

    def new_sequence(self):
        self.write_head.new_sequence()

    def get_prev_write(self):
        return self.write_head.last_write

    def forward(self, memory, nn_output, prev_read_dist, debug):
        shapes = [[self.word_length]] * (4 if self.use_mask else 3) + [[self.n_read_heads]] + [[1]] * 3
        tensors = split_tensor(nn_output, shapes)

        if self.use_mask:
            write_mask = torch.sigmoid(tensors[0])
            tensors=tensors[1:]
        else:
            write_mask = None

        write_content_key, erase_vector, write_vector, free_gates, write_beta, alloc_gate, write_gate = tensors

        erase_vector = torch.sigmoid(erase_vector)
        free_gates = torch.sigmoid(free_gates)
        write_beta = oneplus(write_beta)
        alloc_gate = torch.sigmoid(alloc_gate)
        write_gate = torch.sigmoid(write_gate)

        return self.write_head(memory, write_content_key, write_beta, erase_vector, write_vector,
                               alloc_gate, write_gate, free_gates, prev_read_dist, debug=debug, write_mask=write_mask)

    def get_neural_input_size(self):
        return self.input_size


class ReadHead(torch.nn.Module):
    def __init__(self, disable_content_norm=False, mask_min=0.0, disable_key_masking=False):
        super(ReadHead, self).__init__()
        self.content_addr_generator = ContentAddressGenerator(disable_content_norm=disable_content_norm,
                                                              mask_min=mask_min,
                                                              disable_key_masking=disable_key_masking)
        self.read_dist = None
        self.read_data = None
        self.new_sequence()

    def new_sequence(self):
        self.read_dist = None
        self.read_data = None

    def forward(self, memory, read_content_keys, read_betas, forward_dist, backward_dist, gates, read_mask=None, debug=None):
        content_dist = self.content_addr_generator(memory, read_content_keys, read_betas, mask=read_mask)

        self.read_dist = backward_dist * gates[..., 0:1] + content_dist * gates[...,1:2] + forward_dist * gates[..., 2:]

        # memory shape: [ batch, cell count, word_length ]
        # read_dist shape: [ batch, n heads, cell count ]
        # result shape: [ batch, n_heads, word_length ]
        self.read_data = (memory.unsqueeze(1) * self.read_dist.unsqueeze(-1)).sum(-2)

        return self.read_data


class RawReadHead(torch.nn.Module):
    def __init__(self, n_heads, word_length, use_mask=False, disable_content_norm=False, mask_min=0.0,
                 disable_key_masking=False):
        super(RawReadHead, self).__init__()
        self.read_head = ReadHead(disable_content_norm=disable_content_norm, mask_min=mask_min,
                                  disable_key_masking=disable_key_masking)
        self.n_heads = n_heads
        self.word_length = word_length
        self.use_mask = use_mask
        self.input_size = self.n_heads * (self.word_length*(2 if use_mask else 1) + 3 + 1)

    def get_prev_dist(self, memory):
        if self.read_head.read_dist is not None:
            return self.read_head.read_dist
        else:
            m_shape = memory.size()
            return torch.zeros(m_shape[0], self.n_heads, m_shape[1]).to(memory)

    def get_prev_data(self, memory):
        if self.read_head.read_data is not None:
            return self.read_head.read_data
        else:
            m_shape = memory.size()
            return torch.zeros(m_shape[0], self.n_heads, m_shape[-1]).to(memory)

    def new_sequence(self):
        self.read_head.new_sequence()

    def forward(self, memory, nn_output, forward_dist, backward_dist, debug):
        shapes = [[self.n_heads, self.word_length]] * (2 if self.use_mask else 1) + [[self.n_heads], [self.n_heads, 3]]
        tensors = split_tensor(nn_output, shapes)

        if self.use_mask:
            read_mask = torch.sigmoid(tensors[0])
            tensors = tensors[1:]
        else:
            read_mask = None

        keys, betas, gates = tensors

        betas = oneplus(betas)
        gates = F.softmax(gates, gates.dim()-1)

        return self.read_head(memory, keys, betas, forward_dist, backward_dist, gates, debug=debug, read_mask=read_mask)

    def get_neural_input_size(self):
        return self.input_size
