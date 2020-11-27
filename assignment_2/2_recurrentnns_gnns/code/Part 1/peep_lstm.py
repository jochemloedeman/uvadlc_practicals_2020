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


def init_weights(dim_1, dim_2):
    weights = torch.empty(dim_1, dim_2)
    nn.init.kaiming_normal_(weights, nonlinearity='linear')
    return weights


def init_bias(dim):
    bias = torch.zeros(dim)
    return bias


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
        self.w_cx = nn.Parameter(init_weights(embedding_dim, hidden_dim))
        self.b_c = nn.Parameter(init_bias(hidden_dim))

        # input gate
        self.w_ix = nn.Parameter(init_weights(embedding_dim, hidden_dim))
        self.w_ih = nn.Parameter(init_weights(hidden_dim, hidden_dim))
        self.b_i = nn.Parameter(init_bias(hidden_dim))

        # forget gate
        self.w_fx = nn.Parameter(init_weights(embedding_dim, hidden_dim))
        self.w_fh = nn.Parameter(init_weights(hidden_dim, hidden_dim))
        self.b_f = nn.Parameter(init_bias(hidden_dim))

        # output gate
        self.w_ox = nn.Parameter(init_weights(embedding_dim, hidden_dim))
        self.w_oh = nn.Parameter(init_weights(hidden_dim, hidden_dim))
        self.b_o = nn.Parameter(init_bias(hidden_dim))

        # output layer
        self.w_ph = nn.Parameter(init_weights(hidden_dim, num_classes))
        self.b_p = nn.Parameter(init_bias(num_classes))
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

