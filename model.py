import itertools
import math

import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn


class ProtectedAttributeClassifier(nn.Module):
    def __init__(self):
        super(ProtectedAttributeClassifier, self).__init__()
        linear_list = [13, 64, 32, 1]
        self.linear_layers = nn.ModuleList()
        for i in range(len(linear_list) - 2):
            self.linear_layers.append(nn.Linear(linear_list[i], linear_list[i+1]))

        self.final_layer = nn.Linear(linear_list[-2], linear_list[-1])

    def forward(self, x):
        for layer in self.linear_layers:
            x = F.relu(layer(x))

        return self.final_layer(x), x


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


class MultiHeadAttention(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 2 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, q, mask=None, return_attention=True):
        batch_size, seq_length, embed_dim = x.size()
        kv = self.qkv_proj(x)
        # Separate Q, K, V from linear output
        kv = kv.reshape(batch_size, seq_length, self.num_heads, 2 * self.head_dim)
        kv = kv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        k, v = kv.chunk(2, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o


class AttributeClassifier(nn.Module):
    def __init__(self, p_model):
        super(AttributeClassifier, self).__init__()
        linear_list = [14, 64, 32, 1]
        self.p_model = p_model
        self.linear_layers = nn.ModuleList()
        for i in range(len(linear_list) - 2):
            self.linear_layers.append(nn.Linear(linear_list[i], linear_list[i+1]))

        self.final_layer = nn.Linear(linear_list[-2], linear_list[-1])
        self.attention = MultiHeadAttention(1, 1, 1)

    def get_linear_parameters(self):
        return itertools.chain(self.linear_layers.parameters(), self.final_layer.parameters())

    def get_attention_parameters(self):
        return self.attention.parameters()

    def forward(self, x):
        x_p = np.delete(x, 9, 1)
        for layer in self.linear_layers:
            x = F.relu(layer(x))
        with torch.no_grad():
            q = self.p_model(x_p)[1].unsqueeze(dim=2).unsqueeze(dim=1)
        x, attention = self.attention(x.unsqueeze(dim=2), q)
        return torch.sigmoid(self.final_layer(x.view(-1, 32))), attention
