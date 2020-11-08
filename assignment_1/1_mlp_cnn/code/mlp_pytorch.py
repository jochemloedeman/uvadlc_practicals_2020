"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch

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
        if not n_hidden:
            self.hidden_layers = None
            self.output = nn.Linear(n_inputs, n_classes)
        else:
            linear_sizes = [n_inputs] + n_hidden
            hidden_layers = nn.ModuleList([])
            for i in range(1, len(linear_sizes)):
                hidden_layers.extend([nn.Linear(linear_sizes[i - 1], linear_sizes[i]),
                                      nn.ELU()])
            self.hidden_layers = hidden_layers
            self.output = nn.Linear(n_hidden[-1], n_classes)
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
                x = mod(x)
        x = self.output(x)
        out = x
        ########################
        # END OF YOUR CODE    #
        #######################
        
        return out


if __name__ == '__main__':
    mlp = MLP(3, [4], 2)
    inputs = torch.Tensor([24, 2, 5])
    output = mlp(inputs)

    print('hi')
