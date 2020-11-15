"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils
import time

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100, 100'
LEARNING_RATE_DEFAULT = 1e-3
MAX_STEPS_DEFAULT = 3000
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None


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


def plot_curve(values, label):
    plt.plot(values, label=label)
    plt.xlabel('Batches')
    plt.ylabel(label)
    plt.savefig(fname='torch' + label + '.eps', format='eps', bbox_inches='tight', dpi=200)
    plt.show()


def train():
    """
    Performs training and evaluation of MLP model.
  
    TODO:
    Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
    """

    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    ## Prepare all functions
    # Get number of units in each hidden layer specified in the string such as 100,100
    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    else:
        dnn_hidden_units = []

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir, one_hot=False)
    mlp_model = MLP(3072, dnn_hidden_units, 10)
    loss_module = nn.CrossEntropyLoss()
    test_images, test_labels = torch.from_numpy(cifar10['test'].images).to(device), \
                               torch.from_numpy(cifar10['test'].labels).to(device)

    test_vectors = reshape_images(test_images)

    accuracies = []
    losses = []
    mlp_model.to(device)
    optimizer = torch.optim.Adam(mlp_model.parameters(), lr=FLAGS.learning_rate)
    mlp_model.train()

    for i in range(FLAGS.max_steps):

        # load data
        images, labels = cifar10['train'].next_batch(FLAGS.batch_size)
        image_vectors = reshape_images(images)
        image_vectors, labels = torch.from_numpy(image_vectors), torch.from_numpy(labels)
        image_vectors, labels = image_vectors.to(device), labels.to(device)
        labels.to(device)

        # forward pass
        model_pred = mlp_model(image_vectors)

        # calculate the loss
        loss = loss_module(model_pred, labels)

        # backward pass
        optimizer.zero_grad()
        loss.backward()

        # update the parameters
        optimizer.step()

        # evaluate the model on the data set every eval_freq steps
        mlp_model.eval()
        if i % FLAGS.eval_freq == 0:
            with torch.no_grad():
                test_pred = mlp_model(test_vectors)
                test_accuracy = accuracy(test_pred, test_labels)
                accuracies.append(test_accuracy)
                losses.append(loss)

        mlp_model.train()

    print(accuracies)
    plot_curve(accuracies, 'Accuracy')
    plot_curve(losses, 'Loss')
    ########################
    # END OF YOUR CODE    #
    #######################


def reshape_images(images):
    return images.reshape((images.shape[0], -1))


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
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
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
