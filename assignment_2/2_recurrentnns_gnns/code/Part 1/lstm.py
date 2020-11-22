"""
This module implements a LSTM model in PyTorch.
You should fill in code into indicated sections.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch


class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, num_classes,
                 batch_size, device):
        super(LSTM, self).__init__()
        ########################
        # PUT YOUR CODE HERE  #
        #######################

        embedding_dim = int(hidden_dim / 4)
        self.embedding = nn.Embedding(num_embeddings=3, embedding_dim=embedding_dim)
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.device = device

        # input modulation gate
        w_gx = torch.empty(embedding_dim, hidden_dim)
        nn.init.kaiming_normal_(w_gx, nonlinearity='linear')
        self.w_gx = nn.Parameter(w_gx)
        w_gh = torch.empty(hidden_dim, hidden_dim)
        nn.init.kaiming_normal_(w_gh, nonlinearity='linear')
        self.w_gh = nn.Parameter(w_gh)
        b_g = torch.zeros(hidden_dim)
        self.b_g = nn.Parameter(b_g)

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
        h = torch.zeros(self.hidden_dim, device=self.device)
        c = torch.zeros(self.hidden_dim, device=self.device)
        embedded_x = self.embedding(x.type(torch.cuda.LongTensor).squeeze())

        # iterate over the sequence
        for i in range(self.seq_length):
            digit_batch = embedded_x[:, i, :]
            g = torch.tanh(digit_batch @ self.w_gx + h @ self.w_gh + self.b_g)
            i = torch.sigmoid(digit_batch @ self.w_ix + h @ self.w_ih + self.b_i)
            f = torch.sigmoid(digit_batch @ self.w_fx + h @ self.w_fh + self.b_f)
            o = torch.sigmoid(digit_batch @ self.w_ox + h @ self.w_oh + self.b_o)
            c = g * i + c * f
            h = torch.tanh(c) * o

        # calculate the log probabilities of the two classes
        p = h @ self.w_ph + self.b_p
        y = nn.functional.log_softmax(p, dim=0)
        return y
        ########################
        # END OF YOUR CODE    #
        #######################


if __name__ == '__main__':
    model = LSTM(seq_length=6, input_dim=1, hidden_dim=256, num_classes=2, batch_size=6, device=torch.device("cuda:0"))
    print('hi')
