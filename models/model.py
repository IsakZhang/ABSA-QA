# This script contains the proposed model

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastNLP.modules import LSTM
from fastNLP.core.utils import seq_len_to_mask
from fastNLP.embeddings import BertEmbedding

from modules.modules import MHALayer, SelfAttentiveEncoding, LocalContextEncoderCNN


def _compute_loss_or_predict(logits, mask, target=None, num_labels=None):

    # shape: (bs, seq_len)
    prediction = torch.argmax(logits, dim=-1)

    if target is not None:
        # check dim match first, need to have same seq len
        assert target.shape[1] == logits.shape[1]
        # those padding token will be '-100' in the target
        masked_target = target.masked_fill(mask.eq(False), -100)

        # flatten the loss of logits and target to
        # shape: (bs*seq_len, num_labels) and (bs*seq_len)
        loss = F.cross_entropy(logits.view(-1, num_labels),
                               masked_target.view(-1),
                               ignore_index=-100)

        return {'loss': loss, 'pred': prediction}
    else:
        return {'pred': prediction}


class Model(nn.Module):
    def __init__(self, embed, embed_dropout=0.1, 
                 context_encoder=True, context_hidden_dim=100,
                 inter_attn_heads=12, intra_attn_heads=12, attn_dropout=0.1,
                 a_enc_hidden_dim=512, a_enc_trans=True, a_enc_output_dim=64,
                 kernel_size=3, d_self_hidden=128, d_local_hidden=128,
                 ate_task=True, append_a=True, tagger_dropout=0.1,
                 ate_target_vocab=None, unified_target_vocab=None):
        
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
                                                nhead=inter_attn_heads, dropout=attn_dropout)
        self.inter_sent_attn_layer_a = MHALayer(self.context_hidden_dim, 
                                                nhead=inter_attn_heads, dropout=attn_dropout)

        self.Wrq = nn.Linear(self.context_hidden_dim, self.context_hidden_dim)
        self.Waq = nn.Linear(self.context_hidden_dim, self.context_hidden_dim)
        self.fusion_dropout = nn.Dropout(0.1)
        self.fusion_ln = nn.LayerNorm(self.context_hidden_dim)

        self.intra_q_attn_layer = MHALayer(self.context_hidden_dim, 
                                           nhead=intra_attn_heads, dropout=attn_dropout)

        # whether to conduct Aspect Term Extraction task
        self.ate_task = ate_task
        if self.ate_task:
            self.num_ate_labels = len(ate_target_vocab)
            self.ate_clf = nn.Linear(self.context_hidden_dim, self.num_ate_labels)
        
        self.a_attn_encoding = SelfAttentiveEncoding(input_dim=self.context_hidden_dim, 
                                                     hidden_dim=a_enc_hidden_dim,
                                                     trans=a_enc_trans,
                                                     output_dim=a_enc_output_dim)
        self.append_a = append_a
        if self.append_a:
            self.local_context_input = self.context_hidden_dim + a_enc_output_dim
        else:
            self.local_context_input = self.context_hidden_dim

        self.local_context_layer = LocalContextEncoderCNN(
                                    d_input=self.local_context_input, 
                                    kernel_size=kernel_size, 
                                    d_self_hidden=d_self_hidden,
                                    d_local_hidden=d_local_hidden, dropout=0.2)       
        
        self.local_context_dropout = nn.Dropout(tagger_dropout)
        self.tagger_ln = nn.LayerNorm(d_local_hidden)

        self.num_unified_labels = len(unified_target_vocab)
        self.unified_clf = nn.Linear(d_local_hidden, self.num_unified_labels)

    def _forward(self, question, answer, q_len, a_len,
                 ate_target=None, unified_target=None):

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
        if self.context_encoder:
            q_vec, _ = self.context_encoder(q_vec, q_len)
            a_vec, _ = self.context_encoder(a_vec, a_len)

        # --------------------------------------------------------------
        # capture inter-sentence interactions between Q&A
        attended_q = self.inter_sent_attn_layer_q(q_vec, a_vec, a_vec, a_mask.eq(False))

        # --------------------------------------------------------------
        # fusion layer to combine both Q and A
        fusion_gate = torch.sigmoid(self.Wrq(q_vec) + self.Waq(attended_q))
        enriched_q = self.fusion_dropout(fusion_gate * q_vec + (1-fusion_gate) * attended_q)
        enriched_q = self.fusion_ln(enriched_q)

        # --------------------------------------------------------------
        # question self attention to capture aspect information
        self_attended_q = self.intra_q_attn_layer(enriched_q, enriched_q, enriched_q, q_mask.eq(False))

        # --------------------------------------------------------------
        # Since q now have both Q+A info, they can be used to perform ATE task
        if self.ate_task:
            ate_logits = self.ate_clf(self_attended_q)
            ate_results = _compute_loss_or_predict(ate_logits, q_mask, ate_target, self.num_ate_labels)

        attended_a = self.inter_sent_attn_layer_a(a_vec, q_vec, q_vec, q_mask.eq(False))
        encoded_a = self.a_attn_encoding(attended_a, answer)

        # if stack answer vec to each question token to enhance the sentiment info:
        if self.append_a:
            max_q_len = q_mask.shape[1]
            answer_broadcasted = encoded_a.unsqueeze(1).repeat(1, max_q_len, 1)
            self_attended_q = torch.cat([self_attended_q, answer_broadcasted], dim=-1)

        # --------------------------------------------------------------        
        # model local context to control the sentiment consistency
        q_with_local_context = self.local_context_dropout(self.local_context_layer(self_attended_q))
        
        unified_clf_input = self.tagger_ln(q_with_local_context)
        unified_logits = self.unified_clf(unified_clf_input)
        unified_results = _compute_loss_or_predict(unified_logits, q_mask, 
                                                   unified_target, self.num_unified_labels)

        output = {"unified": unified_results}

        if self.ate_task:
            output['ate'] = ate_results

        return output

    def forward(self, question, answer, q_len, a_len, ate_target, unified_target):
        results = self._forward(question, answer, q_len, a_len, ate_target, unified_target)
        loss = results['unified']['loss']
        pred = results['unified']['pred']
        output = {"pred": pred, "loss": loss}
        if self.ate_task:
            output['ate_pred'] = results['ate']['pred']
            output['loss'] += results['ate']['loss'] * 0.5
        return output

    def predict(self, question, answer, q_len, a_len):
        results = self._forward(question, answer, q_len, a_len)
        pred = results['unified']['pred']
        output = {"pred": pred}
        if self.ate_task:
            output['ate_pred'] = results['ate']['pred']
        return output
