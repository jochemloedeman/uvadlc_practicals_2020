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
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP

        TODO:
        Implement initialization of the network.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        if not n_hidden:
            self.hidden_layers = None
            self.output = LinearModule(n_inputs, n_classes)
            self.softmax = SoftMaxModule()
            self.linear_modules = [self.output]

        else:
            hidden_layers = []
            linear_sizes = [n_inputs] + n_hidden

            for i in range(1, len(linear_sizes)):
                hidden_layers.extend([LinearModule(linear_sizes[i - 1], linear_sizes[i]), ELUModule()])
            self.hidden_layers = hidden_layers
            self.output = LinearModule(n_hidden[-1], n_classes)
            self.softmax = SoftMaxModule()
            self.linear_modules = [mod for mod in self.hidden_layers if isinstance(mod, LinearModule)] + [self.output]
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

        TODO:
        Implement forward pass of the network.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        if self.hidden_layers is not None:
            for mod in self.hidden_layers:
                x = mod.forward(x)
        x = self.output.forward(x)
        x = self.softmax.forward(x)
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

        TODO:
        Implement backward pass of the network.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        dout = self.softmax.backward(dout)
        dout = self.output.backward(dout)
        for mod in self.hidden_layers[::-1]:
            dout = mod.backward(dout)
        ########################
        # END OF YOUR CODE    #
        #######################

        return

    def update(self, learning_rate):
        for mod in self.linear_modules:
            mod.params['weight'] -= learning_rate * mod.grads['weight']
            mod.params['bias'] -= learning_rate * mod.grads['bias']


if __name__ == '__main__':
    mlp = MLP(3, [4], 2)
    inputs = np.array([24, 2, 5])[np.newaxis, :]
    output = mlp.forward(inputs)
    mlp.backward(np.array([1, 0])[np.newaxis, :])
    print('hi')
