# This script contains various network modules

import torch.nn as nn
from fastNLP import Callback
from fastNLP.modules import SelfAttention


class MHALayer(nn.Module):
    """
    A wrapper for Pytorch's MultiheadAttention with LN and Dropout 
    """
    def __init__(self, input_size, nhead=12, dropout=0.1):
        super(MHALayer, self).__init__()
        self.input_size = input_size
        self.nhead = nhead
        self.self_attn = nn.MultiheadAttention(input_size, nhead, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(input_size)

    def forward(self, Q, K, V, input_mask=None):
        """
        `Q`: a seq of vec: (bs, seq_len_q, encoding_dim)
        `K/V`: a seq of vec: (bs, seq_len_s, encoding_dim)
        `input_mask`: the mask of input of shape (bs, seq_len_s)
            Note that the PADDING token item needs to be True in the mask
        `output`: a seq of vec: (bs, seq_len, encoding_dim_q)
        """
        # adapt the input shape for the self attention networks (make bs in 2nd dim)
        Q = Q.transpose(0, 1)
        K = K.transpose(0, 1)
        V = V.transpose(0, 1)
        attn_output, _ = self.self_attn(Q, K, V, key_padding_mask=input_mask)
        src = Q + self.dropout(attn_output)
        output = self.layer_norm(src).transpose(0, 1)

        return output


class SelfAttentiveEncoding(nn.Module):
    """
    Encode a sequence with the attention for each token 
    """
    def __init__(self, input_dim, hidden_dim, attn_hops=1, 
                 combinehead='max', trans=True, output_dim=128):
        super(SelfAttentiveEncoding, self).__init__()
        self.attn_layer = SelfAttention(input_size=input_dim,
                                        attention_unit=hidden_dim,
                                        attention_hops=attn_hops)
        self.attn_hops = attn_hops

        # how to process multi-head results
        self.combinehead = combinehead

        self.trans = trans
        if self.trans:
            self.output_dim = output_dim
            self.trans_input = nn.Linear(input_dim, output_dim)
        else:
            self.output_dim = input_dim

    def forward(self, input, raw_input):
        # input: (bs, seq_len, hidden_dim), raw_input: (bs, seq_len)
        # encoded_input: (bs, num_heads, hidden_dim)
        encoded_input, _ = self.attn_layer(input, raw_input)

        if self.attn_hops == 1:
            encoded_input = encoded_input.squeeze(1)  # (bs, hidden_dim)
        else:
            if self.combinehead == 'max':
                # max-pooling
                encoded_input = encoded_input.max(dim=1)[0]

        if self.trans:
            output = self.trans_input(encoded_input)

        return output


class LocalContextEncoderCNN(nn.Module):
    """ 
    Capture local context information with CNN 
    """
    def __init__(self, d_input, kernel_size=3, d_self_hidden=None,
                 d_local_hidden=None, dropout=0.1):
        super(LocalContextEncoderCNN, self).__init__()
        if d_self_hidden is None:
            d_self_hidden = d_input
        if d_local_hidden is None:
            d_local_hidden = d_self_hidden
        
        # point-wise CNN for feature transformation
        self.feature_trans = nn.Conv1d(in_channels=d_input, 
                                       out_channels=d_self_hidden,
                                       kernel_size=1)
        
        # local CNN for capturing context info
        self.local_feature = nn.Conv1d(in_channels=d_self_hidden,
                                       out_channels=d_local_hidden,
                                       kernel_size=kernel_size,
                                       padding=kernel_size//2)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, input):
        # input shape: (bs, seq_len, d_input)
        output = self.relu(self.feature_trans(input.transpose(1, 2)))
        output = self.relu(self.local_feature(output)).transpose(2, 1)
        output = self.dropout(output)
        return output


class RecordLossCallback(Callback):
    """
    Record average loss in each epoch for the model
    """
    def __init__(self, save_path):
        super().__init__()
        self.total_loss = 0
        self.start_step = 0
        self.save_path = save_path

    def on_backward_begin(self, loss):
        self.total_loss += loss.item()

    def on_epoch_end(self):
        n_steps = self.step - self.start_step
        avg_loss = self.total_loss / n_steps
        with open(self.save_path, "a") as f:
            loss = f'{avg_loss}\n'
            f.write(loss)
        self.start_step = self.step
        self.total_loss = 0
