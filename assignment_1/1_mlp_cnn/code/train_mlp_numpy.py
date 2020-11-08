"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils as cifar10_utils
import matplotlib.pyplot as plt

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 1e-3
MAX_STEPS_DEFAULT = 1400
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

    TODO:
    Implement accuracy computation.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    batch_size = predictions.shape[0]
    correct_predictions = 0

    for i in range(batch_size):
        prediction_index = np.argmax(predictions[i])
        if predictions[i] @ targets[i] == predictions[i][prediction_index]:
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
    plt.savefig(fname='numpy' + label + '.eps', format='eps', bbox_inches='tight', dpi=200)
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

    cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)
    mlp_model = MLP(3072, dnn_hidden_units, 10)
    loss_module = CrossEntropyModule()

    test_images, test_labels = cifar10['test'].images, cifar10['test'].labels
    test_vectors = reshape_images(test_images)

    accuracies = []
    losses = []

    for i in range(FLAGS.max_steps):
        images, labels = cifar10['train'].next_batch(FLAGS.batch_size)
        image_vectors = reshape_images(images)

        # forward pass
        model_pred = mlp_model.forward(image_vectors)

        # backward pass
        loss = loss_module.forward(model_pred, labels)
        loss_grad = loss_module.backward(model_pred, labels)
        mlp_model.backward(loss_grad)

        # update all weights and biases
        mlp_model.update(FLAGS.learning_rate)

        # evaluate the model on the data set every eval_freq steps
        if i % FLAGS.eval_freq == 0:
            test_pred = mlp_model.forward(test_vectors)
            test_accuracy = accuracy(test_pred, test_labels)
            accuracies.append(test_accuracy)
            losses.append(loss)

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
