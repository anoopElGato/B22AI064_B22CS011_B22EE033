import torch

from torch import nn, Tensor


class FrequencyTransformerBlock(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()

        self.attn_norm = nn.LayerNorm(input_size)
        self.attn = nn.MultiheadAttention(embed_dim=input_size, num_heads=num_heads,
                                          dropout=dropout, batch_first=True)
        self.gru_norm = nn.LayerNorm(input_size)
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=1, bidirectional=True, batch_first=True)
        self.gru_projection = nn.Linear(hidden_size * 2, input_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: Tensor):
        """
        :param inputs: (B, F, T, N)
        :return:
        """
        batch, bands, frames, channels = inputs.shape

        outputs = torch.permute(inputs, dims=(0, 2, 1, 3)).contiguous()
        outputs = torch.reshape(outputs, shape=(batch * frames, bands, channels))

        attn_inputs = self.attn_norm(outputs)
        attn_out, _ = self.attn(attn_inputs, attn_inputs, attn_inputs, need_weights=False)
        outputs = outputs + self.dropout(attn_out)

        gru_inputs = self.gru_norm(outputs)
        gru_out, _ = self.gru(gru_inputs)
        gru_out = self.gru_projection(gru_out)
        outputs = outputs + self.dropout(gru_out)

        outputs = torch.reshape(outputs, shape=(batch, frames, bands, channels))
        outputs = torch.permute(outputs, dims=(0, 2, 1, 3)).contiguous()

        return outputs
