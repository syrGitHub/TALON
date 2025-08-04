import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearExpert(nn.Module):
    """
    Just one Linear layer
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(self, input_size, output_size):
        super(LinearExpert, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.Linear = nn.Linear(self.input_size, self.output_size)

    def forward(self, x):
        # x: [Linear_token_num x token_len]

        if x.size(0) == 0 or x.size(1) == 0:
            # 如果输入为空，直接返回一个占位符（如空Tensor）或跳过计算
            # print(f"LinearExpert Skipping computation due to invalid input shape: {x.shape}")
            return torch.empty((0, self.output_size), device=x.device, dtype=x.dtype) # 返回一个与输出形状一致的空Tensor

        x = self.Linear(x)

        return x  # [Linear_token_num x hidden_dim_of_llama]


class CNNExpert(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size):
        """
        Args:
            patch_len: patch 内的 token 个数
            dim_per_token: 每个 token 的维度（例如 embedding size）
            hidden_dim: 中间隐藏层，用于卷积输出的维度
        """
        super(CNNExpert, self).__init__()
        self.output_size = output_size
        self.conv1 = nn.Conv1d(in_channels=1,
                               out_channels=hidden_dim,
                               kernel_size=3,
                               padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=hidden_dim,
                               out_channels=1,
                               kernel_size=1)
        self.proj = nn.Linear(input_size, output_size)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [CNN_token_num x hidden_dim_of_llama]
        Returns:
            Tensor of shape [CNN_token_num x hidden_dim_of_llama]
        """
        if x.size(0) == 0 or x.size(1) == 0:
            return torch.empty((0, self.output_size), device=x.device, dtype=x.dtype) # 返回一个与输出形状一致的空Tensor

        # x: [CNN_token_num x 1 x token_len] CNN expects [batch, channels, seq_len]
        x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = x.squeeze(1)
        x = self.proj(x)  # [N, token_len] → [N, output_size]
        return x


class LSTMExpert(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size, num_layers=1, bidirectional=False):
        """
        Args:
            input_size: 每个 patch token 的维度（1）
            hidden_dim: LSTM 隐藏层维度
            num_layers: LSTM 层数
            bidirectional: 是否使用双向 LSTM
        """
        super(LSTMExpert, self).__init__()
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.proj = nn.Linear(self.num_directions * hidden_dim, output_size)  # 保证输出维度不变

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [LSTM_token_num x input_size]
        Returns:
            Tensor of shape [LSTM_token_num x input_size]
        """
        if x.size(0) == 0 or x.size(1) == 0:
            return torch.empty((0, self.output_size), device=x.device, dtype=x.dtype) # 返回一个与输出形状一致的空Tensor

        # 需要给 LSTM 一个 "input_size" 维度（这里 batch_size = LSTM_token_num, seq_len = token_len, input_size = 1）
        x = x.unsqueeze(2)  # [N, L, 1]

        lstm_out, _ = self.lstm(x)  # [N, L, hidden_dim * num_directions]
        lstm_last = lstm_out[:, -1, :]  # 取最后时间步的输出
        out = self.proj(lstm_last)  # [N, output_size]
        return out
