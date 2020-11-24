"""
This module implements a LSTM with peephole connections in PyTorch.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class peepLSTM(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, num_classes,
                 batch_size, device):

        super(peepLSTM, self).__init__()

        ########################
        # PUT YOUR CODE HERE  #
        #######################

        # the embedding dimension is set based on TA recommendations
        embedding_dim = int(hidden_dim / 2)

        # number of embeddings is hardcoded to 3, to account for the binary values + padding
        self.embedding = nn.Embedding(num_embeddings=3, embedding_dim=embedding_dim)

        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.device = device

        # cell state
        w_cx = torch.empty(embedding_dim, hidden_dim)
        nn.init.kaiming_normal_(w_cx, nonlinearity='linear')
        self.w_cx = nn.Parameter(w_cx)
        b_c = torch.zeros(hidden_dim)
        self.b_c = nn.Parameter(b_c)

        # input gate
        w_ix = torch.empty(embedding_dim, hidden_dim)
        nn.init.kaiming_normal_(w_ix, nonlinearity='linear')
        self.w_ix = nn.Parameter(w_ix)
        w_ih = torch.empty(hidden_dim, hidden_dim)
        nn.init.kaiming_normal_(w_ih, nonlinearity='linear')
        self.w_ih = nn.Parameter(w_ih)
        b_i = torch.zeros(hidden_dim)
        self.b_i = nn.Parameter(b_i)

        # forget gate
        w_fx = torch.empty(embedding_dim, hidden_dim)
        nn.init.kaiming_normal_(w_fx, nonlinearity='linear')
        self.w_fx = nn.Parameter(w_fx)
        w_fh = torch.empty(hidden_dim, hidden_dim)
        nn.init.kaiming_normal_(w_fh, nonlinearity='linear')
        self.w_fh = nn.Parameter(w_fh)
        b_f = torch.zeros(hidden_dim)
        self.b_f = nn.Parameter(b_f)

        # output gate
        w_ox = torch.empty(embedding_dim, hidden_dim)
        nn.init.kaiming_normal_(w_ox, nonlinearity='linear')
        self.w_ox = nn.Parameter(w_ox)
        w_oh = torch.empty(hidden_dim, hidden_dim)
        nn.init.kaiming_normal_(w_oh, nonlinearity='linear')
        self.w_oh = nn.Parameter(w_oh)
        b_o = torch.zeros(hidden_dim)
        self.b_o = nn.Parameter(b_o)

        # output layer
        w_ph = torch.empty(hidden_dim, num_classes)
        nn.init.kaiming_normal_(w_ph, nonlinearity='linear')
        self.w_ph = nn.Parameter(w_ph)
        b_p = torch.zeros(num_classes)
        self.b_p = nn.Parameter(b_p)
        ########################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################

        # initializes initial values, and embed the sequences
        c = torch.zeros(self.batch_size, self.hidden_dim, device=self.device)
        embedded_x = self.embedding(x.type(torch.cuda.LongTensor).squeeze())

        # create a tensor that assigns 1 or 0 to sequence values to identify padding values
        padding_correction = (x.type(torch.cuda.LongTensor).squeeze() + 1) // 2

        # iterate over the sequences
        for j in range(self.seq_length - 1):
            digit_batch = embedded_x[:, j, :]
            i = torch.sigmoid(digit_batch @ self.w_ix + c @ self.w_ih + self.b_i)
            f = torch.sigmoid(digit_batch @ self.w_fx + c @ self.w_fh + self.b_f)
            o = torch.sigmoid(digit_batch @ self.w_ox + c @ self.w_oh + self.b_o)
            c = torch.sigmoid(digit_batch @ self.w_cx + self.b_c) * i + c * f
            h = torch.tanh(c) * o

            # set hidden states to 0 if the input of the iteration was a padding value
            h = h * padding_correction[:, j, None]

        # calculate the log probabilities of the two classes
        p = h @ self.w_ph + self.b_p
        y = nn.functional.log_softmax(p, dim=0)

        return y
        ########################
        # END OF YOUR CODE    #
        #######################

