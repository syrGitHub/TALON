# layers/temporal_mlp.py
# Cell
from typing import Callable, Optional
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

class x_encoder(nn.Module):
    def __init__(self, config):
        super(x_encoder, self).__init__()
        input_size = config.token_len
        num_experts = config.num_experts
        encoder_hidden_size = config.hidden_size

        self.distribution_fit = nn.Sequential(nn.Linear(input_size, encoder_hidden_size, bias=False), nn.ReLU(),
                                              nn.Linear(encoder_hidden_size, num_experts, bias=False))

    def forward(self, x):
        # x: [bs * n_vars * token_num x token_len]
        out = self.distribution_fit(x)
        return out

class complexity_encoder(nn.Module):
    def __init__(self, config):
        super(complexity_encoder, self).__init__()
        input_size = config.num_experts
        num_experts = config.num_experts
        encoder_hidden_size = config.hidden_size

        self.distribution_fit = nn.Sequential(nn.Linear(input_size, encoder_hidden_size, bias=False), nn.ReLU(),
                                              nn.Linear(encoder_hidden_size, num_experts, bias=False))

    def forward(self, x):
        # complexity: [bs * n_vars * token_num x expert_num]
        out = self.distribution_fit(x)
        return out