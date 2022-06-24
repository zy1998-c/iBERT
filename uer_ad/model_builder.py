# -*- encoding:utf-8 -*-
import torch
from uer_ad.layers.embeddings import BertEmbedding, WordEmbedding
from uer_ad.encoders.bert_encoder import BertEncoder
from uer_ad.encoders.rnn_encoder import LstmEncoder, GruEncoder
from uer_ad.encoders.birnn_encoder import BilstmEncoder
from uer_ad.encoders.cnn_encoder import CnnEncoder, GatedcnnEncoder
from uer_ad.encoders.attn_encoder import AttnEncoder
from uer_ad.encoders.gpt_encoder import GptEncoder
from uer_ad.encoders.mixed_encoder import RcnnEncoder, CrnnEncoder
from uer_ad.encoders.synt_encoder import SyntEncoder
from uer_ad.targets.bert_target import BertTarget
from uer_ad.targets.lm_target import LmTarget
from uer_ad.targets.cls_target import ClsTarget
from uer_ad.targets.mlm_target import MlmTarget
from uer_ad.targets.nsp_target import NspTarget
from uer_ad.targets.s2s_target import S2sTarget
from uer_ad.targets.bilm_target import BilmTarget
from uer_ad.subencoders.avg_subencoder import AvgSubencoder
from uer_ad.subencoders.rnn_subencoder import LstmSubencoder
from uer_ad.subencoders.cnn_subencoder import CnnSubencoder
from uer_ad.models.model import Model


def build_model(args):
    """
    Build universial encoder representations models.
    The combinations of different embedding, encoder, 
    and target layers yield pretrained models of different 
    properties. 
    We could select suitable one for downstream tasks.
    """

    if args.subword_type != "none":
        subencoder = globals()[args.subencoder.capitalize() + "Subencoder"](args, len(args.sub_vocab))
    else:
        subencoder = None

    embedding = globals()[args.embedding.capitalize() + "Embedding"](args, len(args.vocab))
    encoder = globals()[args.encoder.capitalize() + "Encoder"](args)
    target = globals()[args.target.capitalize() + "Target"](args, len(args.vocab))
    model = Model(args, embedding, encoder, target, subencoder)

    return model
