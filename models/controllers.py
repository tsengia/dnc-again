from typing import List
import torch
import torch.utils.data
import torch.nn.init as init
import math

from models.util import linear_reset

class LSTMController(torch.nn.Module):
    def __init__(self, layer_sizes, out_from_all_layers: bool =True):
        super(LSTMController, self).__init__()
        self.out_from_all_layers = out_from_all_layers
        self.layer_sizes = layer_sizes
        self.states = None
        self.outputs = None

    def new_sequence(self):
        self.states = [None] * len(self.layer_sizes)
        self.outputs = [None] * len(self.layer_sizes)

    def reset_parameters(self):
        def init_layer(layer, index):
            size = self.layer_sizes[index]
            # Initialize all matrices to sigmoid, just data input to tanh
            a=math.sqrt(3.0)*self.stdevs[i]
            layer.weight.data[0:-size].uniform_(-a,a)
            a*=init.calculate_gain("tanh")
            layer.weight.data[-size:].uniform_(-a, a)
            if layer.bias is not None:
                layer.bias.data[self.layer_sizes[i]:].fill_(0)
                # init forget gate to large number.
                layer.bias.data[:self.layer_sizes[i]].fill_(1)

        # xavier init merged input weights
        for i in range(len(self.layer_sizes)):
            init_layer(self.in_to_all[i], i)
            init_layer(self.out_to_all[i], i)
            if i>0:
                init_layer(self.prev_to_all[i-1], i)

    def _add_modules(self, name, m_list):
        for i, m in enumerate(m_list):
            self.add_module("%s_%d" % (name,i), m)

    def init(self, input_size):
        self.layer_sizes = self.layer_sizes

        # Xavier init: input to all gates is layers_sizes[i-1] + layer_sizes[i] + input_size -> layer_size big.
        # So use xavier init according to this.
        self.input_sizes = [(self.layer_sizes[i - 1] if i>0 else 0) + self.layer_sizes[i] + input_size
                            for i in range(len(self.layer_sizes))]
        self.stdevs = [math.sqrt(2.0 / (self.layer_sizes[i] + self.input_sizes[i])) for i in range(len(self.layer_sizes))]
        self.in_to_all= [torch.nn.Linear(input_size, 4*self.layer_sizes[i]) for i in range(len(self.layer_sizes))]
        self.out_to_all = [torch.nn.Linear(self.layer_sizes[i], 4 * self.layer_sizes[i], bias=False) for i in range(len(self.layer_sizes))]
        self.prev_to_all = [torch.nn.Linear(self.layer_sizes[i-1], 4 * self.layer_sizes[i], bias=False) for i in range(1,len(self.layer_sizes))]

        self._add_modules("in_to_all", self.in_to_all)
        self._add_modules("out_to_all", self.out_to_all)
        self._add_modules("prev_to_all", self.prev_to_all)

        self.reset_parameters()

    def get_output_size(self):
        return sum(self.layer_sizes) if self.out_from_all_layers else self.layer_sizes[-1]

    def forward(self, data):
        for i, size in enumerate(self.layer_sizes):
            d = self.in_to_all[i](data)
            if self.outputs[i] is not None:
                d+=self.out_to_all[i](self.outputs[i])
            if i>0:
                d+=self.prev_to_all[i-1](self.outputs[i-1])

            input_data = torch.tanh(d[...,-size:])
            forget_gate, input_gate, output_gate = torch.sigmoid(d[...,:-size]).chunk(3,dim=-1)

            state_update = input_gate * input_data

            if self.states[i] is not None:
                self.states[i] = self.states[i]*forget_gate + state_update
            else:
                self.states[i] = state_update

            self.outputs[i] = output_gate * torch.tanh(self.states[i])

        return torch.cat(self.outputs, -1) if self.out_from_all_layers else self.outputs[-1]


class FeedforwardController(torch.nn.Module):
    def __init__(self, layer_sizes: List[int]=[]):
        super(FeedforwardController, self).__init__()
        self.layer_sizes = layer_sizes

    def new_sequence(self):
        pass

    def reset_parameters(self):
        for module in self.model:
            if isinstance(module, torch.nn.Linear):
                linear_reset(module, gain=init.calculate_gain("relu"))

    def get_output_size(self):
        return self.layer_sizes[-1]

    def init(self, input_size):
        self.layer_sizes = self.layer_sizes

        # Xavier init: input to all gates is layers_sizes[i-1] + layer_sizes[i] + input_size -> layer_size big.
        # So use xavier init according to this.
        self.input_sizes = [input_size] + self.layer_sizes[:-1]

        layers = []
        for i, size in enumerate(self.layer_sizes):
            layers.append(torch.nn.Linear(self.input_sizes[i], self.layer_sizes[i]))
            layers.append(torch.nn.ReLU())
        self.model = torch.nn.Sequential(*layers)
        self.reset_parameters()

    def forward(self, data):
        return self.model(data)
