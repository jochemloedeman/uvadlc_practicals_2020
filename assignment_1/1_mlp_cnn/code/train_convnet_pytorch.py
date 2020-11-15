"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
from statistics import mean

import numpy as np
import os
import matplotlib.pyplot as plt
from convnet_pytorch import ConvNet
import cifar10_utils

import torch
import torch.nn as nn

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def plot_curve(values, label):
    plt.plot(values, label=label)
    plt.xlabel('Batches')
    plt.ylabel(label)
    plt.savefig(fname='torch' + label + '.eps', format='eps', bbox_inches='tight', dpi=200)
    plt.show()


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.
    
    Args:
      predictions: 2D float array of size [batch_size, n_classes]
      labels: 2D int array of size [batch_size, n_classes]
              with one-hot encoding. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    batch_size = predictions.shape[0]
    correct_predictions = 0
    for i in range(batch_size):
        if torch.argmax(predictions[i]) == targets[i]:
            correct_predictions += 1

    accuracy = correct_predictions / batch_size
    ########################
    # END OF YOUR CODE    #
    #######################

    return accuracy


def train():
    """
    Performs training and evaluation of ConvNet model.
    """

    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir, one_hot=False)
    test_set = cifar10["test"]
    test_batch_size = 2000
    n_test_batches = int(test_set.num_examples / test_batch_size)

    conv_net = ConvNet(n_channels=3, n_classes=10)
    loss_module = nn.CrossEntropyLoss()

    accuracies = []
    losses = []

    conv_net.to(device)
    optimizer = torch.optim.Adam(conv_net.parameters(), lr=FLAGS.learning_rate)
    conv_net.train()
    start_time = time.time()
    for i in range(FLAGS.max_steps):

        # load data
        images, labels = cifar10['train'].next_batch(FLAGS.batch_size)
        images, labels = torch.from_numpy(images).to(device), torch.from_numpy(labels).to(device)

        # forward pass
        model_pred = conv_net(images)

        # calculate the loss
        loss = loss_module(model_pred, labels)

        # backward pass
        optimizer.zero_grad()
        loss.backward()

        # update the parameters
        optimizer.step()

        # evaluate the model on the data set every eval_freq steps
        conv_net.eval()
        if i % FLAGS.eval_freq == 0:
            test_accuracy = test_conv(conv_net, test_set, n_test_batches, test_batch_size)
            accuracies.append(test_accuracy)
            losses.append(loss)

        conv_net.train()

    print(time.time() - start_time)
    print(accuracies)
    plot_curve(accuracies, 'Accuracy')
    plot_curve(losses, 'Loss')

    ########################
    # END OF YOUR CODE    #
    #######################


def test_conv(model, test_set, n_batches, batch_size):
    batch_accuracy = 0
    for j in range(n_batches):
        with torch.no_grad():
            test_images, test_labels = test_set.next_batch(batch_size)
            test_images, test_labels = torch.from_numpy(test_images).to(device), torch.from_numpy(
                test_labels).to(device)
            test_pred = model(test_images)
            test_accuracy = accuracy(test_pred, test_labels)
            batch_accuracy += test_accuracy

    return batch_accuracy / n_batches


def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def main():
    """
    Main function
    """
    # Print all Flags to confirm parameter settings
    print_flags()

    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)

    # Run the training operation
    train()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()

    main()
