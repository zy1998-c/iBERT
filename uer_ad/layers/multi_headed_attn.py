# -*- encoding:utf-8 -*-
import math
import torch
import torch.nn as nn
import pdb

class MultiHeadedAttention(nn.Module):
    """
    Each head is a self-attention operation.
    self-attention refers to https://arxiv.org/pdf/1706.03762.pdf
    """
    def __init__(self, hidden_size, heads_num, dropout):
        super(MultiHeadedAttention, self).__init__()
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        self.per_head_size = hidden_size // heads_num

        self.linear_layers = nn.ModuleList([
                nn.Linear(hidden_size, hidden_size) for _ in range(3)
            ])
        
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(hidden_size, hidden_size)


    def forward(self, args, layer, key, value, query, mask, lena=None, kg=None):
        """
        Args:
            key: [batch_size x seq_length x hidden_size]
            value: [batch_size x seq_length x hidden_size]
            query: [batch_size x seq_length x hidden_size]
            mask: [batch_size x 1 x seq_length x seq_length]

        Returns:
            output: [batch_size x seq_length x hidden_size]
        """

        batch_size, seq_length, hidden_size = key.size()
        heads_num = self.heads_num
        per_head_size = self.per_head_size

        def shape(x):
            return x. \
                   contiguous(). \
                   view(batch_size, seq_length, heads_num, per_head_size). \
                   transpose(1, 2)

        def unshape(x):
            return x.transpose(1, 2).contiguous().view(batch_size, seq_length, hidden_size)

        query, key, value = [l(x). \
                             view(batch_size, -1, heads_num, per_head_size). \
                             transpose(1, 2) \
                             for l, x in zip(self.linear_layers, (query, key, value))
                            ]
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / math.sqrt(float(per_head_size))
        scores = scores + mask
        probs = nn.Softmax(dim=-1)(scores)   # softmax

        large = torch.ones_like(probs).cuda()  # 32,12,128,128
        if lena != None:
            for i in range(len(lena)):  # 32, 2
                large[i, :, 1:int(lena[i][0])-1, int(lena[i][0]):-1] = args.hy_layer[layer][2]
                large[i, :, int(lena[i][0]):-1, 1: int(lena[i][0])-1] = args.hy_layer[layer][2]
        if kg != None:
            kg = kg.permute(1, 0, 2, 3)
            sim = kg[0].unsqueeze(1)
            top = kg[1].unsqueeze(1)
            probs = torch.mul(probs, large)
            probs = probs + args.hy_layer[layer][0]*sim + args.hy_layer[layer][1]*top

        probs = self.dropout(probs)
        output = unshape(torch.matmul(probs, value))
        output = self.final_linear(output)
        
        return output
