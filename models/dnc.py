# Copyright 2017 Robert Csordas. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================

import torch
import torch.utils.data

import lightning as L

from models.util import linear_reset, split_tensor

from models.rwheads import RawReadHead, RawWriteHead
from models.temporal_memory import TemporalMemoryLinkage, DistSharpnessEnhancer

class DNC(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int, word_length: int, cell_count: int, n_read_heads: int, controller, batch_first: bool=False, clip_controller=20,
                 bias: bool=True, mask: bool=False, dealloc_content: bool=True, link_sharpness_control: bool=True, disable_content_norm: bool=False,
                 mask_min: float=0.0, disable_key_masking: bool=False):
        super(DNC, self).__init__()

        self.clip_controller = clip_controller

        self.read_head = RawReadHead(n_read_heads, word_length, use_mask=mask, disable_content_norm=disable_content_norm,
                                     mask_min=mask_min, disable_key_masking=disable_key_masking)
        self.write_head = RawWriteHead(n_read_heads, word_length, use_mask=mask, dealloc_content=dealloc_content,
                                       disable_content_norm=disable_content_norm, mask_min=mask_min,
                                       disable_key_masking=disable_key_masking)
        self.temporal_link = TemporalMemoryLinkage()
        self.sharpness_control = DistSharpnessEnhancer([n_read_heads, n_read_heads]) if link_sharpness_control else None

        in_size = input_size + n_read_heads * word_length
        control_channels = self.read_head.get_neural_input_size() + self.write_head.get_neural_input_size() +\
                           (self.sharpness_control.get_neural_input_size() if self.sharpness_control is not None else 0)

        self.controller = controller
        controller.init(in_size)
        self.controller_to_controls = torch.nn.Linear(controller.get_output_size(), control_channels, bias=bias)
        self.controller_to_out = torch.nn.Linear(controller.get_output_size(), output_size, bias=bias)
        self.read_to_out = torch.nn.Linear(word_length * n_read_heads, output_size, bias=bias)

        self.cell_count = cell_count
        self.word_length = word_length

        self.memory = None
        self.reset_parameters()

        self.batch_first = batch_first
        self.zero_mem_tensor = None

    def reset_parameters(self):
        linear_reset(self.controller_to_controls)
        linear_reset(self.controller_to_out)
        linear_reset(self.read_to_out)
        self.controller.reset_parameters()

    def _step(self, in_data):

        # input shape: [ batch, channels ]
        batch_size = in_data.size(0)

        # run the controller
        prev_read_data = self.read_head.get_prev_data(self.memory).view([batch_size, -1])

        control_data = self.controller(torch.cat([in_data, prev_read_data], -1))

        # memory ops
        controls = self.controller_to_controls(control_data).contiguous()
        controls = controls.clamp(-self.clip_controller, self.clip_controller) if self.clip_controller is not None else controls

        shapes = [[self.write_head.get_neural_input_size()], [self.read_head.get_neural_input_size()]]
        if self.sharpness_control is not None:
            shapes.append(self.sharpness_control.get_neural_input_size())

        tensors = split_tensor(controls, shapes)

        write_head_control, read_head_control = tensors[:2]
        tensors = tensors[2:]

        prev_read_dist = self.read_head.get_prev_dist(self.memory)

        self.memory = self.write_head(self.memory, write_head_control, prev_read_dist)

        prev_write = self.write_head.get_prev_write()
        forward_dist, backward_dist = self.temporal_link(prev_write["write_dist"] if prev_write is not None else None, prev_read_dist)

        if self.sharpness_control is not None:
            forward_dist, backward_dist = self.sharpness_control(tensors[0], forward_dist, backward_dist)

        read_data = self.read_head(self.memory, read_head_control, forward_dist, backward_dist)

        # output:
        return self.controller_to_out(control_data) + self.read_to_out(read_data.view(batch_size,-1))

    def _mem_init(self, batch_size, device):
        if self.zero_mem_tensor is None or self.zero_mem_tensor.size(0)!=batch_size:
            self.zero_mem_tensor = torch.zeros(batch_size, self.cell_count, self.word_length).to(device)

        self.memory = self.zero_mem_tensor

    def forward(self, in_data):
        self.write_head.new_sequence()
        self.read_head.new_sequence()
        self.temporal_link.new_sequence()
        self.controller.new_sequence()

        self._mem_init(in_data.size(0 if self.batch_first else 1), in_data.device)

        out_tsteps = []

        if self.batch_first:
            # input format: batch, time, channels
            for t in range(in_data.size(1)):
                out_tsteps.append(self._step(in_data[:,t]))
        else:
            # input format: time, batch, channels
            for t in range(in_data.size(0)):
                out_tsteps.append(self._step(in_data[t]))

        return torch.stack(out_tsteps, dim=1 if self.batch_first else 0)


class LitDNC(L.LightningModule):
    def __init__(self, optimizer_type: str, learning_rate: float, weight_decay: float, momentum: float, eps: float = 1e-10):
        super().__init__()
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.eps = eps
        self.model = DNC()
        # TODO: Setup the model
        pass

    def training_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        
        if self.optimizer_type == "sgd":
            return torch.optim.SGD(self.params, self.learning_rate, self.weight_decay, self.momentum)
        elif self.optimizer_type == "adam":
            return torch.optim.Adam(self.params, self.learning_rate, self.weight_decay)
        elif self.optimizer_type == "rmsprop":
            return torch.optim.RMSprop(self.params, self.learning_rate, self.weight_decay, self.momentum, self.eps)
        else:
            assert "Invalid optimizer: %s" % self.optimizer_type