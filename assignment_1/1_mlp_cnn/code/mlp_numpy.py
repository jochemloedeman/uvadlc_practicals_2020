"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import *


class MLP(object):
    """
    This class implements a Multi-layer Perceptron in NumPy.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward and backward.
    """

    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes MLP object.

        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any hidden layers, and the model
                    will simply perform a multinomial logistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        layers = []
        sizes = [n_inputs] + n_hidden
        for i in range(1, len(sizes)):
            layers.extend([LinearModule(sizes[i - 1], sizes[i]), ELUModule()])
        layers.append(LinearModule(n_hidden[-1], n_classes))
        layers.append(SoftMaxModule())
        self.layers = layers
        self.linear_modules = [mod for mod in self.layers if isinstance(mod, LinearModule)]
        ########################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        for mod in self.layers:
            x = mod.forward(x)
        out = x
        ########################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Performs backward pass given the gradients of the loss.

        Args:
          dout: gradients of the loss
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        for mod in self.layers[::-1]:
            dout = mod.backward(dout)
        ########################
        # END OF YOUR CODE    #
        #######################

        return

    def update(self, learning_rate):
        """
        Performs the SGD update step

        Args:
          learning_rate
        """

        for mod in self.linear_modules:
            mod.params['weight'] -= learning_rate * mod.grads['weight']
            mod.params['bias'] -= learning_rate * mod.grads['bias']