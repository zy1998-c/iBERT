# -*- encoding:utf-8 -*-
import torch.nn as nn
import torch
import pdb
from uer_ad.layers.layer_norm import LayerNorm
from uer_ad.layers.position_ffn import PositionwiseFeedForward
from uer_ad.layers.multi_headed_attn import MultiHeadedAttention


class TransformerLayer(nn.Module):
    """
    Transformer layer mainly consists of two parts:
    multi-headed self-attention and feed forward layer.
    """
    def __init__(self, args):
        super(TransformerLayer, self).__init__()

        # Multi-headed self-attention.
        self.self_attn = MultiHeadedAttention(
            args.hidden_size, args.heads_num, args.dropout
        )
        self.dropout_1 = nn.Dropout(args.dropout)
        self.layer_norm_1 = LayerNorm(args.hidden_size)
        # Feed forward layer.
        self.feed_forward = PositionwiseFeedForward(
            args.hidden_size, args.feedforward_size
        )
        self.dropout_2 = nn.Dropout(args.dropout)
        self.layer_norm_2 = LayerNorm(args.hidden_size)
        self.ad_magn = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size), nn.ReLU(),
            nn.Linear(args.hidden_size, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.ad_top = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size), nn.ReLU(),
            nn.Linear(args.hidden_size, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.ad_sim = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size), nn.ReLU(),
            nn.Linear(args.hidden_size, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, args, layer, hidden, mask, lena=None, mul=None):
        """
        Args:
            hidden: [batch_size x seq_length x emb_size]
            mask: [batch_size x 1 x seq_length x seq_length]

        Returns:
            output: [batch_size x seq_length x hidden_size]
        """

        batchs_num = args.instances_num // args.batch_size
        if args.instances_num % args.batch_size != 0:
            batchs_num += 1
        kg_layer = args.kg_layer
        tem = torch.mean(hidden, dim=1)
        if layer in kg_layer[0]:
            hy_sim = torch.exp(self.ad_sim(tem))
            hy_sim = hy_sim.mean(dim=0).cpu()
            hy_sim = float(hy_sim.detach().numpy()[0])
            args.hy_layer[layer][0] = hy_sim
            if args.batch_num == 0:
                args.hy_cal[layer][0] = hy_sim
            else:
                args.hy_cal[layer][0] = hy_sim + args.hy_cal[layer][0]
            if args.batch_num == batchs_num - 1:
                args.hy[layer][0] = args.hy_cal[layer][0] / args.batch_num

        if layer in kg_layer[1]:
            hy_top = torch.exp(self.ad_top(tem))
            hy_top = hy_top.mean(dim=0).cpu()
            hy_top = float(hy_top.detach().numpy()[0])
            args.hy_layer[layer][1] = hy_top
            if args.batch_num == 0:
                args.hy_cal[layer][1] = hy_top
            else:
                args.hy_cal[layer][1] = hy_top + args.hy_cal[layer][1]
            if args.batch_num == batchs_num - 1:
                args.hy[layer][1] = args.hy_cal[layer][1] / args.batch_num

        if layer in kg_layer[2]:
            hy_magn = torch.exp(self.ad_magn(tem))
            hy_magn = hy_magn.mean(dim=0).cpu()
            hy_magn = float(hy_magn.detach().numpy()[0])
            args.hy_layer[layer][2] = hy_magn
            if args.batch_num == 0:
                args.hy_cal[layer][2] = hy_magn
            else:
                args.hy_cal[layer][2] = hy_magn + args.hy_cal[layer][2]
            if args.batch_num == batchs_num - 1:
                args.hy[layer][2] = args.hy_cal[layer][2] / args.batch_num

        if args.batch_num == batchs_num - 1 and layer == 11:
            print(args.hy)
        inter = self.dropout_1(self.self_attn(args, layer, hidden, hidden, hidden, mask, lena, mul))
        inter = self.layer_norm_1(inter + hidden)
        output = self.dropout_2(self.feed_forward(inter))
        output = self.layer_norm_2(output + inter)  
        return output
