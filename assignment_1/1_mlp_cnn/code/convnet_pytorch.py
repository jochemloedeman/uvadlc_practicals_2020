"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn


class ConvNet(nn.Module):
    """
    This class implements a Convolutional Neural Network in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an ConvNet object can perform forward.
    """

    def __init__(self, n_channels, n_classes):
        """
        Initializes ConvNet object.
        
        Args:
          n_channels: number of input channels
          n_classes: number of classes of the classification problem
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        super().__init__()
        up_scalings = [64, 128, 256, 512]
        self.input_processing = nn.Sequential(nn.Conv2d(n_channels, up_scalings[0], kernel_size=3, padding=1))
        upscale_blocks = []
        for i in range(1, len(up_scalings)):
            upscale_blocks.extend([nn.Conv2d(up_scalings[i - 1], up_scalings[i], kernel_size=1),
                                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                   ResNetLike(up_scalings[i]), ResNetLike(up_scalings[i])])

        additional_blocks = [nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                             ResNetLike(up_scalings[-1]), ResNetLike(up_scalings[-1]),
                             nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]

        total_blocks = upscale_blocks + additional_blocks
        self.inner_blocks = nn.Sequential(*total_blocks)
        self.output = nn.Sequential(nn.BatchNorm2d(up_scalings[-1]), nn.ReLU(),
                                    nn.Flatten(), nn.Linear(up_scalings[-1], n_classes))
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
        x = self.input_processing(x)
        x = self.inner_blocks(x)
        out = self.output(x)
        ########################
        # END OF YOUR CODE    #
        #######################

        return out


class ResNetLike(nn.Module):

    def __init__(self, input_channels):
        super().__init__()
        self.layers = nn.Sequential(nn.BatchNorm2d(input_channels), nn.ReLU(),
                                     nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1))

    def forward(self, x):
        out = x + self.layers(x)
        return out
