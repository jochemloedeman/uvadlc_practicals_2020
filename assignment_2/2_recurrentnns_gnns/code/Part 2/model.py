# MIT License
#
# Copyright (c) 2019 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2019
# Date Created: 2019-09-06
###############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch


class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0'):
        super(TextGenerationModel, self).__init__()

        embedding_dim = 64
        self.embedding = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=lstm_num_hidden,
                            num_layers=lstm_num_layers)
        self.linear = nn.Linear(lstm_num_hidden, vocabulary_size)
        self.device = device
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.nr_layers = lstm_num_layers
        self.nr_hidden = lstm_num_hidden
        self.h_n = None
        self.c_n = None

    def forward(self, x, h_0, c_0):
        embedded_x = self.embedding(x.type(torch.cuda.LongTensor))
        out, (self.h_n, self.c_n) = self.lstm(embedded_x, (h_0, c_0))
        out = self.linear(out)
        return out
