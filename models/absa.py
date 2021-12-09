# This script includes two simple absa layers: GRU and SAN described in the paper:
#   Exploiting BERT for End-to-End Aspect-Based Sentiment Analysis
# Codes in this file are adopted from:
#   https://github.com/lixin4ever/BERT-E2E-ABSA/blob/master/absa_layer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastNLP import seq_len_to_mask


class SAN(nn.Module):
    def __init__(self, input_size, nhead=12, dropout=0.1):
        super(SAN, self).__init__()
        self.input_size = input_size
        self.nhead = nhead
        self.self_attn = nn.MultiheadAttention(input_size, nhead, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(input_size)

    def forward(self, input, input_mask=None):
        """
        `input`: a seq of vec: (bs, seq_len, encoding_dim)
        `input_mask`: the mask of input of shape (bs, seq_len)
            Note that the PADDING token will be True in the mask
        `output`: a seq of vec: (bs, seq_len, encoding_dim)
        """
        # adapt the input shape for the self attention networks (make bs in 2nd dim)
        src = input.transpose(0, 1)
        attn_output, _ = self.self_attn(src, src, src, key_padding_mask=input_mask)
        src = src + self.dropout(attn_output)
        output = self.layer_norm(src).transpose(0, 1)

        return output


class GRU(nn.Module):
    """ customized GRU with layer normalization """
    def __init__(self, input_size, hidden_size, bidirectional=True):
        super(GRU, self).__init__()
        self.input_size = input_size
        if bidirectional:
            self.hidden_size = hidden_size // 2
        else:
            self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.Wxrz = nn.Linear(in_features=self.input_size, out_features=2*self.hidden_size, bias=True)
        self.Whrz = nn.Linear(in_features=self.hidden_size, out_features=2*self.hidden_size, bias=True)
        self.Wxn = nn.Linear(in_features=self.input_size, out_features=self.hidden_size, bias=True)
        self.Whn = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size, bias=True)
        self.LNx1 = nn.LayerNorm(2*self.hidden_size)
        self.LNh1 = nn.LayerNorm(2*self.hidden_size)
        self.LNx2 = nn.LayerNorm(self.hidden_size)
        self.LNh2 = nn.LayerNorm(self.hidden_size)

    def forward(self, x):
        """
        x: input tensor, shape: (batch_size, seq_len, input_size)
        """
        def recurrence(xt, htm1):
            """
            xt: current input
            htm1: previous hidden state
            """
            gates_rz = torch.sigmoid(self.LNx1(self.Wxrz(xt)) + self.LNh1(self.Whrz(htm1)))
            rt, zt = gates_rz.chunk(2, 1)
            nt = torch.tanh(self.LNx2(self.Wxn(xt))+rt*self.LNh2(self.Whn(htm1)))
            ht = (1.0-zt) * nt + zt * htm1
            return ht

        steps = range(x.size(1))
        bs = x.size(0)
        hidden = self.init_hidden(bs)
        # shape: (seq_len, bsz, input_size)
        input = x.transpose(0, 1)
        output = []
        for t in steps:
            hidden = recurrence(input[t], hidden)
            output.append(hidden)
        # shape: (bsz, seq_len, input_size)
        output = torch.stack(output, 0).transpose(0, 1)

        if self.bidirectional:
            output_b = []
            hidden_b = self.init_hidden(bs)
            for t in steps[::-1]:
                hidden_b = recurrence(input[t], hidden_b)
                output_b.append(hidden_b)
            output_b = output_b[::-1]
            output_b = torch.stack(output_b, 0).transpose(0, 1)
            output = torch.cat([output, output_b], dim=-1)
        return output

    def init_hidden(self, bs):
        if torch.cuda.is_available():
            h_0 = torch.zeros(bs, self.hidden_size).cuda()
        else:
            h_0 = torch.zeros(bs, self.hidden_size)
        return h_0


class ABSATagger(nn.Module):
    def __init__(self, tagger_type, embed, num_labels, bert_dropout=0.1, 
                 hidden_dropout=0.1, hidden_size=768):
        super(ABSATagger, self).__init__()
        self.tagger_type = tagger_type

        self.embedding = embed
        self.embed_dim = self.embedding.embedding_dim
        self.num_labels = num_labels
        self.hidden_size = hidden_size

        if self.tagger_type == 'bert_gru':
            self.tagger = GRU(input_size=self.embed_dim, 
                              hidden_size=self.hidden_size)
        elif self.tagger_type == 'bert_san':
            self.tagger = SAN(input_size=self.embed_dim, nhead=12,
                              dropout=hidden_dropout)

        self.bert_dropout = nn.Dropout(bert_dropout)
        self.tagger_dropout = nn.Dropout(hidden_dropout)
        self.layer_norm = nn.LayerNorm(self.embed_dim)
        self.classifier = nn.Linear(self.hidden_size, num_labels)

        self.padding_target_idx = -100   # used for computing masked loss

    def _forward(self, words, seq_len, target=None):         
        """
        Compute loss or predict
        """
        # mask shape: (bs, max_seq_len)
        mask = seq_len_to_mask(seq_len, max_len=words.size(1))
        # after embedding, shape of words: (bs, seq_len, embed_dim)
        words = self.embedding(words)
        tagger_input = self.bert_dropout(words)

        if self.tagger_type == 'bert_gru':
            clf_input = self.tagger(tagger_input)
            clf_input = self.tagger_dropout(clf_input)
        elif self.tagger_type == 'bert_san':
            clf_input = self.tagger(tagger_input)

        logits = self.classifier(clf_input)

        if target is None:
            # shape: (bs, seq_len)
            prediction = torch.argmax(logits, dim=-1)
            return {'pred': prediction}
        else:
            # those padding token will be '-100' in the target
            masked_target = target.masked_fill(mask.eq(False), self.padding_target_idx)

            # flatten the loss of logits and target to
            # shape: (bs*seq_len, num_labels) and (bs*seq_len)
            loss = F.cross_entropy(logits.view(-1, self.num_labels),
                                   masked_target.view(-1),
                                   ignore_index=self.padding_target_idx)

            return {'loss': loss}

    def forward(self, words, seq_len, target):
        return self._forward(words, seq_len, target)

    def predict(self, words, seq_len):
        return self._forward(words, seq_len)
