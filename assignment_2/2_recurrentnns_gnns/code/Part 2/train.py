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

import os
import time
from datetime import datetime
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset import TextDataset
from model import TextGenerationModel
from torch.utils.tensorboard import SummaryWriter


###############################################################################
def plot_curve(x, values, label, title):
    plt.plot(x, values, label=label)
    plt.xlabel('Steps')
    plt.ylabel(label)
    plt.legend()
    plt.title(title)
    plt.savefig(fname='texgen' + label + '.eps', format='eps', bbox_inches='tight', dpi=300)
    plt.clf()


def to_tensor_rep(batch):
    tensor = torch.empty(len(batch), len(batch[0]))
    for i in range(len(batch)):
        tensor[i, :] = batch[i]
    return tensor.type(torch.LongTensor)


def softmax_temp(inputs, temp):
    max_input = torch.max(temp * inputs)
    return torch.exp(temp * inputs - max_input) / torch.sum(torch.exp(temp * inputs - max_input))


def generate_sequence(model, initial_digit, dataset, length=150, temperature=2):
    with torch.no_grad():
        next_digit = initial_digit
        print("\n" + dataset.convert_to_string([initial_digit]), end='')

        # initialize the hidden states with zeros. batch size is now set to 1.
        h = torch.zeros(model.nr_layers, 1, model.nr_hidden, device=model.device)
        c = torch.clone(h)
        for i in range(length):
            # pack input as tensor
            next_input = torch.reshape(torch.tensor([next_digit], dtype=torch.long), (1, 1))
            output = model(next_input, h, c).squeeze()
            digit_distribution = torch.distributions.Categorical(probs=softmax_temp(output, temperature))
            next_digit = digit_distribution.sample().item()
            next_digit_str = dataset.convert_to_string([next_digit])

            # extract the hidden states to prevent duplicate calculations
            h = model.h_n
            c = model.c_n
            print(next_digit_str, end='')

        print("\n")


def train(config):
    writer = torch.utils.tensorboard.SummaryWriter()

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length)
    data_loader = DataLoader(dataset, config.batch_size, config.seq_length, )

    # Initialize the model that we are going to use
    vocabulary_size = dataset.vocab_size
    model = TextGenerationModel(batch_size=config.batch_size, seq_length=config.seq_length,
                                vocabulary_size=vocabulary_size)
    model.to(device)

    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    accuracies = []
    losses = []

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Only for time measurement of step through network
        t1 = time.time()

        #######################################################

        # Move to GPU
        batch_inputs = to_tensor_rep(batch_inputs).to(device)
        batch_targets = to_tensor_rep(batch_targets).to(device)

        # Reset for next iteration
        model.zero_grad()

        #######################################################
        model_output = model(batch_inputs, c_0=torch.zeros(config.lstm_num_layers, batch_inputs.shape[1],
                                                           config.lstm_num_hidden, device=device),
                             h_0=torch.zeros(config.lstm_num_layers, batch_inputs.shape[1],
                                             config.lstm_num_hidden, device=device))

        # for each timestep, the crossentropy loss is computed and subsequently averaged
        batch_losses = torch.zeros(config.seq_length, device=device)
        for i in range(config.seq_length):
            batch_losses[i] = criterion(model_output[i], batch_targets[i])

        loss = (1 / config.seq_length) * torch.sum(batch_losses)

        # compute the gradients, clip them to prevent exploding gradients and backpropagate
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       max_norm=config.max_norm)
        optimizer.step()

        # calculate accuracy
        predictions = torch.argmax(model_output, dim=2)
        correct = (predictions == batch_targets).sum().item()
        accuracy = correct / (model_output.size(0) * model_output.size(1))

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size / float(t2 - t1)

        if (step + 1) % config.print_every == 0:
            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, \
                    Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                config.train_steps, config.batch_size, examples_per_second,
                accuracy, loss
            ))

            # save loss and accuracy
            accuracies.append(accuracy)
            losses.append(loss)
            writer.add_scalar("loss", loss)
            writer.add_scalar("accuracy", accuracy)

        if (step + 1) % config.sample_every == 0:
            model.eval()
            generate_sequence(model, 62, dataset)
            model.train()

        if step == config.train_steps:
            break

    print('Done training.')

    # make loss and accuracy plots
    x = np.arange(len(accuracies)) * config.print_every
    plot_curve(x, accuracies, "Accuracy", "Training accuracy")
    plot_curve(x, losses, "Loss", "Training Loss")


###############################################################################
###############################################################################

if __name__ == "__main__":
    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, required=True,
                        help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30,
                        help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=256,
                        help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2,
                        help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3,
                        help='Learning rate')

    # It is not necessary to implement the following three params,
    # but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96,
                        help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000,
                        help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0,
                        help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=int(10000),
                        help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/",
                        help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=10,
                        help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=200,
                        help='How often to sample from the model')
    parser.add_argument('--device', type=str, default=("cpu" if not torch.cuda.is_available() else "cuda"),
                        help="Device to run the model on.")

    # If needed/wanted, feel free to add more arguments

    config = parser.parse_args()

    # Train the model
    train(config)
