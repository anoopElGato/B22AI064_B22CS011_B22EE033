from typing import List

import torch
from torch import nn, Tensor


class GroupRNN(nn.Module):
    def __init__(self, input_size: int,
                 hidden_size: int,
                 groups: int,
                 rnn_type: str,
                 num_layers: int = 1,
                 bidirectional: bool = False,
                 batch_first: bool = True):
        super().__init__()
        assert input_size % groups == 0, \
            f"input_size % groups must be equal to 0, but got {input_size} % {groups} = {input_size % groups}"
        assert hidden_size % groups == 0, \
            f"hidden_size % groups must be equal to 0, but got {hidden_size} % {groups} = {hidden_size % groups}"

        self.groups = groups
        self.rnn_list = nn.ModuleList()
        for _ in range(groups):
            self.rnn_list.append(
                getattr(nn, rnn_type)(input_size=input_size // groups, hidden_size=hidden_size // groups,
                                      num_layers=num_layers, bidirectional=bidirectional, batch_first=batch_first)
            )

    def forward(self, inputs: Tensor, hidden_state: List[Tensor]):
        """
        :param hidden_state: List[state1, state2, ...], len(hidden_state) = groups
        :param inputs: (batch, steps, input_size)
        :return:
        """
        outputs = []
        out_states = []
        batch, steps, _ = inputs.shape

        inputs = torch.reshape(inputs, shape=(batch, steps, self.groups, -1))
        for idx, rnn in enumerate(self.rnn_list):
            out, state = rnn(inputs[:, :, idx, :], hidden_state[idx])
            outputs.append(out)
            out_states.append(state)

        outputs = torch.cat(outputs, dim=2)

        return outputs, out_states


class InterFrameGRUBlock(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, groups: int, rnn_type: str):
        super().__init__()
        assert rnn_type in ["RNN", "GRU", "LSTM"], f"rnn_type should be RNN/GRU/LSTM, but got {rnn_type}!"

        self.inter_frame_rnn = GroupRNN(input_size=input_size, hidden_size=hidden_size,
                                        groups=groups, rnn_type=rnn_type)
        self.output_projection = nn.Linear(in_features=hidden_size, out_features=input_size)
        self.output_norm = nn.LayerNorm(normalized_shape=input_size, elementwise_affine=True)

    def forward(self, inputs: Tensor, hidden_state: List[Tensor]):
        """
        :param hidden_state: List[state1, state2, ...], len(hidden_state) = groups
        :param inputs: (B, F, T, N)
        :return:
        """
        batch, bands, frames, channels = inputs.shape

        inter_out = torch.reshape(inputs, shape=(batch * bands, frames, channels))
        inter_out, hidden_state = self.inter_frame_rnn(inter_out, hidden_state)
        inter_out = self.output_projection(inter_out)
        inter_out = self.output_norm(inter_out)
        inter_out = torch.reshape(inter_out, shape=(batch, bands, frames, channels))

        return inputs + inter_out, hidden_state
