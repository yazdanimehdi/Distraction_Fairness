import itertools
import math

import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn


class ProtectedAttributeClassifier(nn.Module):
    def __init__(self, dataset):
        super(ProtectedAttributeClassifier, self).__init__()
        if dataset == "Adult":
            linear_list = [13, 64, 32, 1]
        else:
            linear_list = [124, 256, 128, 9]
        self.linear_layers = nn.ModuleList()
        for i in range(len(linear_list) - 2):
            self.linear_layers.append(nn.Linear(linear_list[i], linear_list[i+1]))

        self.final_layer = nn.Linear(linear_list[-2], linear_list[-1])

    def forward(self, x):
        for layer in self.linear_layers:
            x = F.relu(layer(x))

        return torch.sigmoid(self.final_layer(x)), x


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
    def __init__(self, p_model, dataset):
        super(AttributeClassifier, self).__init__()
        if dataset == 'Health':
            linear_list = [125, 256, 128, 1]
            self.f_v = 128
            self.p_a = 123
        else:
            linear_list = [14, 64, 32, 1]
            self.f_v = 32
            self.p_a = 9
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
        x_p = np.delete(x, self.p_a, 1)
        for layer in self.linear_layers:
            x = F.relu(layer(x))
            x = F.dropout(x, 0.2)
        with torch.no_grad():
            q = self.p_model(x_p)[1].unsqueeze(dim=2).unsqueeze(dim=1)
        x, attention = self.attention(x.unsqueeze(dim=2), q)
        return torch.sigmoid(self.final_layer(x.view(-1, self.f_v))), attention


class AttributeClassifierAblation(nn.Module):
    def __init__(self, dataset='Adult', accuracy_layers=(14, 64, 32, 1), fairness_layers=(32, 32, 32, 32),
                 fairness_layers_position=3, fairness_layer_mode="linear"):
        super(AttributeClassifierAblation, self).__init__()
        self.mode = fairness_layer_mode
        assert(fairness_layers_position < len(accuracy_layers))
        assert(fairness_layers[0] == accuracy_layers[fairness_layers_position - 1])
        assert(fairness_layers[-1] == accuracy_layers[fairness_layers_position - 1])
        if dataset == 'Adult':
            assert(accuracy_layers[0] == 14)

        elif dataset == 'Health':
            assert(accuracy_layers[0] == 125)

        else:
            raise NotImplementedError()

        self.accuracy_layers_part_1 = nn.ModuleList()
        for i in range(fairness_layers_position - 1):
            self.accuracy_layers_part_1.append(nn.Linear(accuracy_layers[i], accuracy_layers[i + 1]))

        self.accuracy_layers_part_2 = nn.ModuleList()
        for i in range(fairness_layers_position - 1, len(accuracy_layers) - 1):
            self.accuracy_layers_part_2.append(nn.Linear(accuracy_layers[i], accuracy_layers[i + 1]))

        self.fairness_layers = nn.ModuleList()
        if fairness_layer_mode == 'linear':
            for i in range(len(fairness_layers) - 1):
                self.fairness_layers.append(nn.Linear(fairness_layers[i], fairness_layers[i + 1]))
        elif fairness_layer_mode == 'attention':
            self.fairness_layers.append(MultiHeadAttention(1, 1, 1))
        else:
            raise NotImplementedError()

    def get_accuracy_parameters(self):
        return itertools.chain(self.accuracy_layers_part_1.parameters(), self.accuracy_layers_part_2.parameters())

    def get_fairness_parameters(self):
        return self.fairness_layers.parameters()

    def forward(self, x):

        for layer in self.accuracy_layers_part_1:
            x = F.relu(layer(x))

        for layer in self.fairness_layers:
            if self.mode == 'linear':
                x = F.relu(layer(x))
            else:
                x = layer(x)

        for idx, layer in enumerate(self.accuracy_layers_part_2):
            if idx != len(self.accuracy_layers_part_2) - 1:
                x = F.relu(layer(x))
            else:
                x = torch.sigmoid(layer(x))

        return x