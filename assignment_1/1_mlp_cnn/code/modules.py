import numpy as np


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """

    def __init__(self, in_features, out_features):
        """
        Initializes the parameters of the module.
    
        Args:
          in_features: size of each input sample
          out_features: size of each output sample
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################

        self.params = {'weight': np.random.normal(0, 0.0001, size=(out_features, in_features)),
                       'bias': np.zeros((1, out_features))}
        self.grads = {'weight': np.zeros(shape=(out_features, in_features)), 'bias': np.zeros(out_features)}
        self.output = None
        self.input = None
        self.batch_size = None

        ########################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Forward pass.
    
        Args:
          x: input to the module
        Returns:
          out: output of the module
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################

        self.batch_size = x.shape[0] if len(x.shape) > 1 else 1
        self.input = x
        bias_matrix = np.tile(self.params['bias'], (self.batch_size, 1))
        out = x @ self.params['weight'].T + bias_matrix
        self.output = out
        self.input = x

        ########################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.
    
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################

        self.grads['weight'] = dout.T @ self.input
        self.grads['bias'] = np.sum(dout, axis=0)
        dx = dout @ self.params['weight']

        ########################
        # END OF YOUR CODE    #
        #######################
        return dx


class SoftMaxModule(object):
    """
    Softmax activation module.
    """

    def __init__(self):
        self.output = None

    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################

        max_x = x.max()
        out = np.exp(x - max_x) / np.einsum('ik->i', np.exp(x - max_x))[:, np.newaxis]
        self.output = out

        ########################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################

        dx = dout * self.output - np.einsum('in,in,ij->ij', dout, self.output, self.output)

        ########################
        # END OF YOUR CODE    #
        #######################

        return dx


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """

    def __init__(self):
        self.batch_size = None

    def forward(self, x, y):
        """
        Forward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################

        self.batch_size = x.shape[0]
        out = - 1 / self.batch_size * np.einsum('ik,ik', y, np.log(x))

        ########################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, x, y):
        """
        Backward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################

        self.batch_size = x.shape[0]
        dx = - 1 / self.batch_size * (y / x)

        ########################
        # END OF YOUR CODE    #
        #######################

        return dx


def elu_scalar(x):
    return np.where(x > 0, x, np.exp(x) - 1)


def d_elu_scalar(x):
    return np.where(x > 0, 1, np.exp(x))


class ELUModule(object):
    """
    ELU activation module.
    """

    def __init__(self):
        self.elu = np.vectorize(elu_scalar)
        self.d_elu = np.vectorize(d_elu_scalar)
        self.input = None

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module
        """
        ########################
        # PUT YOUR CODE HERE  #
        #######################

        self.input = x
        out = self.elu(self.input)

        ########################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        dx = dout * self.d_elu(self.input)
        ########################
        # END OF YOUR CODE    #
        #######################
        return dx

