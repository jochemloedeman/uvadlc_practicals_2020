"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
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
    
        TODO:
        Initialize weights self.params['weight'] using normal distribution with mean = 0 and
        std = 0.0001. Initialize biases self.params['bias'] with 0.
    
        Also, initialize gradients with zeros.
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
    
        TODO:
        Implement forward pass of the module.
    
        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
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
    
        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to
        layer parameters in self.grads['weight'] and self.grads['bias'].
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        self.grads['weight'] = dout.T @ self.input
        self.grads['bias'] = np.ones(shape=(1, self.batch_size)) @ dout
        dx = dout @ self.params['weight']
        ########################
        # END OF YOUR CODE    #
        #######################
        return dx


# class LinearList(object):
#
#     def __init__(self, n_inputs, n_hidden):
#         self.sizes = [n_inputs] + n_hidden
#         self.modules = [LinearModule(self.sizes[i - 1], self.sizes[i]) for i in range(1, len(self.sizes))]
#
#     def forward(self, x):
#         for i in range(len(self.modules)):
#             x = self.modules[i].forward(x)
#
#         return x
#
#     def backward(self, dout):


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
    
        TODO:
        Implement forward pass of the module.
        To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    
        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
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
    
        TODO:
        Implement backward pass of the module.
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
    
        TODO:
        Implement forward pass of the module.
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
    
        TODO:
        Implement backward pass of the module.
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
    return x if x >= 0 else (np.exp(x) - 1)


def d_elu_scalar(x):
    return 1 if x > 0 else np.exp(x)


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

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        self.input = x
        out = self.elu(self.input)

        # for element in np.nditer(out):
        #     element = elu_scalar(element)
        #
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

        TODO:
        Implement backward pass of the module.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        dx = dout * self.d_elu(self.input)
        ########################
        # END OF YOUR CODE    #
        #######################
        return dx


if __name__ == '__main__':
    input_data = np.array([4, 1])
    output_data = np.array([[1, 0], [0, 1]])
    module = LinearModule(2, 3)
    a = module.forward(input_data)
    # b = module.backward(input_data, output_data)
    print('hi')
