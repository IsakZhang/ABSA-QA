# This script contains the QA matching model (for pre-training)

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastNLP.modules import LSTM, MaxPoolWithMask, MLP
from fastNLP.core.utils import seq_len_to_mask
from fastNLP.embeddings import BertEmbedding

from modules.modules import MHALayer


class QAMatchingModel(nn.Module):
    def __init__(self, embed, embed_dropout=0.1, 
                 context_encoder=True, context_hidden_dim=100,
                 inter_attn_heads=12, qa_transform_dim=256, qa_match_hidden=256):
        
        super().__init__()
        self.embedding = embed
        self.embed_dim = embed.embed_size
        self.embed_dropout = nn.Dropout(embed_dropout)

        self.context_encoder = context_encoder
        if self.context_encoder:
            self.context_encoder = LSTM(input_size=self.embed_dim, 
                                        hidden_size=context_hidden_dim, 
                                        bidirectional=True)
            self.context_hidden_dim = context_hidden_dim * 2
        else:
            self.context_hidden_dim = embed.embed_size

        self.inter_sent_attn_layer_q = MHALayer(self.context_hidden_dim, 
                                                nhead=inter_attn_heads, dropout=0.2)
        self.inter_sent_attn_layer_a = MHALayer(self.context_hidden_dim, 
                                                nhead=inter_attn_heads, dropout=0.2)

        self.max_pool = MaxPoolWithMask()
        self.qa_transform = nn.Linear(self.context_hidden_dim, qa_transform_dim)
        self.qa_match_dropout = nn.Dropout(0.1)
        self.qa_match_clf = MLP([qa_transform_dim*4, qa_match_hidden, 2], 'relu')
        # self.qa_match_clf = nn.Linear(self.context_hidden_dim*2, 2)

    def _forward(self, question, answer, q_len, a_len, target=None):
      
        # shape: (bs, q_len/a_len) 
        q_mask = seq_len_to_mask(q_len, question.size(1))
        a_mask = seq_len_to_mask(a_len, answer.size(1))

        # q_vec/a_vec shape: (bs, seq_len, embed_dim)
        q_vec = self.embedding(question)
        a_vec = self.embedding(answer)

        if isinstance(self.embedding, BertEmbedding):
            q_vec = self.embed_dropout(q_vec)
            a_vec = self.embed_dropout(a_vec)

        # encode intra-sentence context information for both Q/A
        # shape: (bs, q_len/a_len, context_hidden_dim)
        if self.context_encoder:
            q_vec, _ = self.context_encoder(q_vec, q_len)
            a_vec, _ = self.context_encoder(a_vec, a_len)

        # --------------------------------------------------------------
        # capture inter-sentence interactions between Q&A
        # shape: (bs, q_len, hidden_dim)
        attended_q = self.inter_sent_attn_layer_q(q_vec, a_vec, a_vec, a_mask.eq(False))
        attended_q = self.qa_transform(attended_q)

        attended_a = self.inter_sent_attn_layer_a(a_vec, q_vec, q_vec, q_mask.eq(False))
        attended_a = self.qa_transform(attended_a)

        pooled_q = self.max_pool(attended_q, q_mask)
        pooled_a = self.max_pool(attended_a, a_mask)

        # qa_match_input = torch.cat([pooled_q, pooled_a], dim=-1)
        qa_match_input = [pooled_q, pooled_a, pooled_q - pooled_a, pooled_q * pooled_a]
        qa_match_input = self.qa_match_dropout(torch.cat(qa_match_input, dim=-1))
        qa_match_logits = self.qa_match_clf(qa_match_input)

        output = {"pred": torch.argmax(qa_match_logits, dim=-1)}
        if target is not None:
            output["loss"] = F.cross_entropy(qa_match_logits, target)               

        return output

    def forward(self, question, answer, q_len, a_len, target):
        return self._forward(question, answer, q_len, a_len, target)
    
    def predict(self, question, answer, q_len, a_len):
        return self._forward(question, answer, q_len, a_len)

