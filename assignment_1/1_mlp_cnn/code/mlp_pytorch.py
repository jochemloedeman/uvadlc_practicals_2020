"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch

from custom_layernorm import CustomLayerNormAutograd


class MLP(nn.Module):
    """
    This class implements a Multi-layer Perceptron in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward.
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

        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        super().__init__()
        sizes = [n_inputs] + n_hidden
        layers = []
        for i in range(1, len(sizes)):
            layers.extend([nn.Linear(sizes[i - 1], sizes[i]),
                           nn.ELU(), nn.BatchNorm1d(sizes[i])])
        layers.append(nn.Linear(sizes[-1], n_classes))
        self.layers = nn.Sequential(*layers)
        for mod in self.layers:
            if isinstance(mod, nn.Linear):
                nn.init.normal_(mod.weight, mean=0, std=0.0001)

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
        out = self.layers(x)
        ########################
        # END OF YOUR CODE    #
        #######################

        return out


if __name__ == '__main__':
    mlp = MLP(3, [4], 2)
    inputs = torch.Tensor([24, 2, 5])
    output = mlp(inputs)

    print('hi')
