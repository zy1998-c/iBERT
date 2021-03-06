"""
This script provides an exmaple to wrap UER-py for classification.
"""
import torch
import random
import argparse
import string
import time
import collections
import numpy as np
import torch.nn as nn
import pandas as pd
import os
import sys
import pdb
from sklearn.utils import shuffle
from nltk.corpus import wordnet as wn
from uer_ad.utils.vocab import Vocab
from uer_ad.utils.constants import *
from uer_ad.utils.tokenizer import *
from uer_ad.layers.embeddings import *
from uer_ad.encoders.bert_encoder import *
from uer_ad.encoders.rnn_encoder import *
from uer_ad.encoders.birnn_encoder import *
from uer_ad.encoders.cnn_encoder import *
from uer_ad.encoders.attn_encoder import *
from uer_ad.encoders.gpt_encoder import *
from uer_ad.encoders.mixed_encoder import *
from uer_ad.utils.optimizers import *
from uer_ad.utils.config import load_hyperparam
from uer_ad.utils.seed import set_seed
from uer_ad.model_saver import save_model
from scipy.stats import spearmanr, pearsonr
class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.embedding = globals()[args.embedding.capitalize() + "Embedding"](args, len(args.vocab))
        self.encoder = globals()[args.encoder.capitalize() + "Encoder"](args)
        self.labels_num = args.labels_num
        self.pooling = args.pooling
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, self.labels_num)

    def forward(self, args, src, tgt, seg, len_a, mix):
        """
        Args:
            src: [batch_size x seq_length]
            tgt: [batch_size]
            seg: [batch_size x seq_length]
        """
        # Embedding.
        emb = self.embedding(src, seg)
        # Encoder.
        output = self.encoder(args, emb, seg, len_a, mix)
        # Target.
        if self.pooling == "mean":
            output = torch.mean(output, dim=1)
        elif self.pooling == "max":
            output = torch.max(output, dim=1)[0]
        elif self.pooling == "last":
            output = output[:, -1, :]
        else:
            output = output[:, 0, :]
        output = torch.tanh(self.output_layer_1(output))
        logits = self.output_layer_2(output)
        if tgt is not None:
            loss = nn.MSELoss()(logits.squeeze(1), tgt)
            return loss, logits
        else:
            return None, logits


def count_labels_num(path):
    '''labels_set, columns = set(), {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.strip().split("\t")):
                    columns[column_name] = i
                continue
            line = line.strip().split("\t")
            label = int(line[columns["labels"]])
            labels_set.add(label)'''
    data = pd.read_csv(path, sep='\t', encoding='utf-8')
    labels = list(data["labels"])
    labels_set = set(labels)
    print('label num: ', len(labels_set))
    return len(labels_set)


def load_or_initialize_parameters(args, model):
    if args.pretrained_model_path is not None:
        # Initialize with pretrained model.
        model.load_state_dict(torch.load(args.pretrained_model_path), strict=False)
    else:
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if 'gamma' not in n and 'beta' not in n:
                p.data.normal_(0, 0.02)


def build_optimizer(args, model):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.train_steps * args.warmup, t_total=args.train_steps)
    return optimizer, scheduler


def batch_loader(batch_size, src, tgt, seg, len_a, mix):
    instances_num = src.size()[0]
    for i in range(instances_num // batch_size):
        src_batch = src[i * batch_size: (i + 1) * batch_size, :]
        tgt_batch = tgt[i * batch_size: (i + 1) * batch_size]
        seg_batch = seg[i * batch_size: (i + 1) * batch_size, :]
        mix_batch = mix[i * batch_size: (i + 1) * batch_size, :]
        len_batch = len_a[i * batch_size: (i + 1) * batch_size, :]
        yield src_batch, tgt_batch, seg_batch, len_batch, mix_batch
    if instances_num > instances_num // batch_size * batch_size:
        src_batch = src[instances_num // batch_size * batch_size:, :]
        tgt_batch = tgt[instances_num // batch_size * batch_size:]
        seg_batch = seg[instances_num // batch_size * batch_size:, :]
        mix_batch = mix[instances_num // batch_size * batch_size:, :]
        len_batch = len_a[instances_num // batch_size * batch_size:, :]
        yield src_batch, tgt_batch, seg_batch, len_batch, mix_batch

def read_dataset(args, path, isshuffle=False):
    dataset, columns = [], {}
    data = pd.read_csv(path, sep='\t', encoding="utf-8", dtype=str)

    if isshuffle == True:
        data = data.sample(frac=1)
        data.reset_index(drop=True, inplace=True)
        text_a, text_b, label, top, sim = data['text_a'], data['text_b'], data['labels'], data['topic'], data['similarity']
        proportion = args.proportion
        text_a, text_b, label, top, sim = text_a.iloc[:int(len(text_a) * proportion)], text_b.iloc[:int(len(text_b) * proportion)], \
                                            label.iloc[:int(len(label) * proportion)],top.iloc[:int(len(top) * proportion)], \
                                            sim.iloc[:int(len(sim) * proportion)]
    else:
        text_a, text_b, label, top, sim = data['text_a'], data['text_b'], data['labels'], data['topic'], data['similarity']
    print(len(text_a))
    for i in range(len(text_a)):
        topic = np.array(eval(top[i]))
        similarity = np.array(eval(sim[i]))
        src_a = [args.vocab.get(t) for t in args.tokenizer.tokenize(str(text_a[i]))]
        src_b = [args.vocab.get(t) for t in args.tokenizer.tokenize(str(text_b[i]))]
        src_a = [CLS_ID] + src_a + [SEP_ID]
        src_b = src_b + [SEP_ID]
        src = src_a + src_b
        seg = [1] * len(src_a) + [2] * len(src_b)
        kg1 = similarity / len(src)
        kg2 = topic / len(src)
        tgt = float(label[i])
        len_a = [len(src_a), len(src)]
        kg_matrix = np.zeros((2, args.seq_length, args.seq_length))
        if len(src) > args.seq_length:
            src = src[:args.seq_length]
            seg = seg[:args.seq_length]
            kg_matrix[0, :args.seq_length, :args.seq_length] = kg1[:args.seq_length, :args.seq_length]
            kg_matrix[1, :args.seq_length, :args.seq_length] = kg2[:args.seq_length, :args.seq_length]
        else:
            kg_matrix[0, :len(seg), :len(seg)] = kg1
            kg_matrix[1, :len(seg), :len(seg)] = kg2
        while len(src) < args.seq_length:
            src.append(0)
            seg.append(0)

        dataset.append((src, tgt, seg, len_a, kg_matrix))

    return dataset


def train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, len_batch, mix_batch):
    model.zero_grad()
    src_batch = src_batch.to(args.device)
    tgt_batch = tgt_batch.to(args.device)
    seg_batch = seg_batch.to(args.device)
    len_batch = len_batch.to(args.device)
    mix_batch = mix_batch.to(args.device)

    loss, _ = model(args, src_batch, tgt_batch, seg_batch, len_batch, mix_batch)
    if torch.cuda.device_count() > 1:
        loss = torch.mean(loss)

    if args.fp16:
        with args.amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()

    optimizer.step()
    scheduler.step()

    return loss


def evaluate(args, dataset, print_confusion_matrix=False):
    src = torch.LongTensor([sample[0] for sample in dataset])
    tgt = torch.FloatTensor([sample[1] for sample in dataset])
    seg = torch.LongTensor([sample[2] for sample in dataset])
    len_a = torch.LongTensor([sample[3] for sample in dataset])
    mix = torch.FloatTensor([sample[4] for sample in dataset])

    batch_size = args.batch_size
    instances_num = src.size()[0]
    args.instances_num = instances_num
    args.model.eval()
    preds, golds = [], []
    for i, (src_batch, tgt_batch, seg_batch, len_batch, mix_batch) in enumerate(batch_loader(batch_size, src, tgt, seg, len_a, mix)):
        args.batch_num = i
        src_batch = src_batch.to(args.device)
        tgt_batch = tgt_batch.to(args.device)
        seg_batch = seg_batch.to(args.device)
        len_batch = len_batch.to(args.device)
        mix_batch = mix_batch.to(args.device)
        with torch.no_grad():
            loss, logits = args.model(args, src_batch, tgt_batch, seg_batch, len_batch, mix_batch)

        preds.extend(logits.squeeze(1).data.cpu().numpy())
        golds.extend(tgt_batch.data.cpu().numpy())

    p = pearsonr(preds, golds)[0]
    s = spearmanr(preds, golds)[0]
    print("Pearsonr {:.4f}, Spearmanr {:.4f}.".format(p, s))
    if print_confusion_matrix:
        final_pred = pd.DataFrame(columns=['index', 'prediction'])
        final_pred['index'] = range(len(preds))
        final_pred['prediction'] = preds
        final_pred.to_csv(args.pred_path, sep='\t', encoding='utf-8', index=False)
    return p, s


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--pretrained_model_path", default="./models/model_reg_train.bin", type=str,
                        help="Path of the pretrained model.")
    parser.add_argument("--output_model_path", default="./models/classifier_model.bin", type=str,
                        help="Path of the output model.")
    parser.add_argument("--vocab_path", type=str, default="./models/google_uncased_en_vocab.txt",
                        help="Path of the vocabulary file.")
    parser.add_argument("--train_path", type=str,
                        help="Path of the trainset.")
    parser.add_argument("--dev_path", type=str,
                        help="Path of the devset.")
    parser.add_argument("--test_path", type=str,
                        help="Path of the testset.")
    parser.add_argument("--pred_path", type=str, default="./results/pred_model.csv",
                        help="Path of the testset.")
    parser.add_argument("--proportion", type=float, default=1.0,
                        help="Path of the testset.")
    parser.add_argument("--config_path", default="./models/bert_base_config.json", type=str,
                        help="Path of the config file.")

    # Model options.
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size.")
    parser.add_argument("--seq_length", type=int, default=128,
                        help="Sequence length.")
    parser.add_argument("--embedding", choices=["bert", "word"], default="bert",
                        help="Emebdding type.")
    parser.add_argument("--encoder", choices=["bert", "lstm", "gru", "cnn", "gatedcnn", "attn", "synt", "rcnn", "crnn", "gpt", "bilstm"],
                        default="bert", help="Encoder type.")
    parser.add_argument("--bidirectional", action="store_true", help="Specific to recurrent model.")
    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type.")
    parser.add_argument("--factorized_embedding_parameterization", action="store_true",
                        help="Factorized embedding parameterization.")
    parser.add_argument("--parameter_sharing", action="store_true", help="Parameter sharing.")

    # Tokenizer options.
    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="bert",
                        help="Specify the tokenizer."
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Space tokenizer segments sentences into words according to space."
                        )

    # Optimizer options.
    parser.add_argument("--soft_targets", action='store_true',
                        help="Train model with logits.")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate.")
    parser.add_argument("--warmup", type=float, default=0.1,
                        help="Warm up value.")
    parser.add_argument("--fp16", action='store_true', default=True,
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit.")
    parser.add_argument("--fp16_opt_level", choices=["O0", "O1", "O2", "O3"], default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    # Training options.
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout.")
    parser.add_argument("--epochs_num", type=int, default=3,
                        help="Number of epochs.")
    parser.add_argument("--report_steps", type=int, default=1000,
                        help="Specific steps to print prompt.")
    parser.add_argument("--seed", type=int, default=47,
                        help="Random seed.")
    parser.add_argument("--kg_layer", type=str, default=True)
    parser.add_argument("--hy_layer", type=list, default=[[0, 0, 1], [0, 0, 1],[0, 0, 1], [0, 0, 1],[0, 0, 1], [0, 0, 1],[0, 0, 1], [0, 0, 1],[0, 0, 1], [0, 0, 1],[0, 0, 1], [0, 0, 1]])
    parser.add_argument("--hy_cal", type=list,
                        default=[[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1],
                                 [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]])
    parser.add_argument("--hy", type=list,
                        default=[[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1],
                                 [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]])
    parser.add_argument("--batch_num", type=int, default=0)
    parser.add_argument("--instances_num", type=int, default=0)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--data_name", type=str)
    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)
    args.kg_layer = eval(args.kg_layer)
    set_seed(args.seed)
    # Count the number of labels.
    args.labels_num = 1
    print('num_labels: ', args.labels_num)
    # Load vocabulary.
    vocab = Vocab()
    vocab.load(args.vocab_path)
    args.vocab = vocab

    # Build classification model.
    model = Classifier(args)

    # Load or initialize parameters.
    load_or_initialize_parameters(args, model)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(args.device)
    args.model = model
    # freeze(args.model)
    # Build tokenizer.
    args.tokenizer = globals()[args.tokenizer.capitalize() + "Tokenizer"](args)
    args.data_name = '_mix'

    args.train_path = "./datasets/{}/train{}.tsv".format(args.dataset_name, args.data_name)
    args.dev_path = './datasets/{}/dev{}.tsv'.format(args.dataset_name, args.data_name)
    args.test_path = './datasets/{}/test{}.tsv'.format(args.dataset_name, args.data_name)

    # Training phase.
    trainset = read_dataset(args, args.train_path, isshuffle=True)
    devset = read_dataset(args, args.dev_path)
    testset = read_dataset(args, args.test_path)

    instances_num = len(trainset)
    batch_size = args.batch_size

    src = torch.LongTensor([example[0] for example in trainset])
    tgt = torch.FloatTensor([example[1] for example in trainset])
    seg = torch.LongTensor([example[2] for example in trainset])
    len_a = torch.LongTensor([example[3] for example in trainset])
    mix = torch.FloatTensor([example[4] for example in trainset])

    args.train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    print("Batch size: ", batch_size)
    print("The number of training instances:", instances_num)

    optimizer, scheduler = build_optimizer(args, model)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        args.amp = amp

    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    total_loss, result, best_result = 0., 0., 0.

    print("Start training.")
    patience_counter = 0
    patience = 5   #early stopping
    args.instances_num = instances_num
    for epoch in range(1, args.epochs_num + 1):
        model.train()
        time_start = time.time()
        for i, (src_batch, tgt_batch, seg_batch, len_batch, mix_batch) in enumerate(batch_loader(batch_size, src, tgt, seg, len_a, mix)):
            args.batch_num = i
            # print('batch:', args.batch_num)
            loss = train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, len_batch, mix_batch)
            total_loss += loss.item()
            if (i + 1) % args.report_steps == 0:
                print("Epoch id: {}, Training steps: {}, Time: {:.4f}s, Avg loss: {:.3f}".format(epoch, i + 1, time.time()-time_start,
                                                                                  total_loss / args.report_steps))
                total_loss = 0.

        print("Start evaluation on dev dataset.")
        pearsonr_correlation, spearmanr_correlation = evaluate(args, devset)
        if pearsonr_correlation < best_result:
            patience_counter += 1
        else:
            best_result = pearsonr_correlation
            patience_counter = 0
            save_model(model, args.output_model_path)

        evaluate(args, testset)

        if patience_counter >= patience:
            print("-> Early stopping: patience limit reached, stopping...")
            print("Final evaluation on the test dataset.")
            if args.test_path is not None:
                print("Test set evaluation.")
                if torch.cuda.device_count() > 1:
                    model.module.load_state_dict(torch.load(args.output_model_path))
                else:
                    model.load_state_dict(torch.load(args.output_model_path))
                evaluate(args, testset)
            os.remove(args.output_model_path)
            break

        if epoch == args.epochs_num:
            print("Final evaluation on the test dataset.")
            if args.test_path is not None:
                print("Test set evaluation.")
                if torch.cuda.device_count() > 1:
                    model.module.load_state_dict(torch.load(args.output_model_path))
                else:
                    model.load_state_dict(torch.load(args.output_model_path))
                evaluate(args, testset)


if __name__ == "__main__":
    main()
