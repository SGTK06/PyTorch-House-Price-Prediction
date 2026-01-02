import torch
from torch import nn


class HousePricePredictor(nn.Module):
    """
    Model structure for predicting the house price
    """
    def __init__(self, input_dim):
        """
        Initialize the neural network structure.
        This is a simple feedforward neural network referred to as
        a Multi-Layer Perceptron (MLP)

        input(input_dim) ->
        hidden layer 1 (128 dim) ->
        hidden layer 2 (32 dim) ->
        output_layer(1 dim)

        :definitions:
        feed-forward -> sequential flow from one layer to another
                        (no loops or branches)
            MLP      -> A simple feedforward neural network with multiple layers
                        (input -> hidden -> output)

        :param input_dim: dimensions of the input tensor/vector
                          containing input data used for making
                          a prediction
        """
        super().__init__()
        self.hidden_layer_1 = nn.Linear(
            in_features=input_dim,
            out_features=128
        )
        self.activation1 = nn.ReLU()
        self.hidden_layer = nn.Linear(
            in_features=128,
            out_features=32
        )
        self.activation2 = nn.ReLU()
        self.output_layer = nn.Linear(
            in_features=32,
            out_features=1
        )
        #alternatively for linear flow of input vector X, nn.Sequential can be used
        #self.model = nn.Sequential(
        #    nn.Linear(input_dim, 128),
        #    nn.ReLU(),
        #    nn.Linear(128, 32),
        #    nn.ReLU(),
        #    nn.Linear(32, 1)
        #)

    def forward(self, x):
        """
        method for forward pass that passes the input tensor/vector
        through the different layers of the neural network.

         - Linear Layer: Made up of several neurons
         - Neuron      : Linear function
                         n = w.X + b (output is weight times input plus bias)
         - Activation  : Activation function used in the neural network.
                         ->It is used to introduce non-linearity in nnet
                         ->allows nnet to learn complex patterns
                         ->without it the nnet will behave as one large linear
                           function performing only linear regression on data
                           =>(w1.x + b1)w2 + b2 = w1.w2.x + (b1.w2 + b2) = w.x + b

        :param x: Input for model to predict the output
        """
        n1 = self.hidden_layer_1(x)
        a1 = self.activation1(n1)
        n2 = self.hidden_layer(a1)
        a2 = self.activation2(n2)
        y = self.output_layer(a2)
        #alternatively the input can directly be passed to model when sequential is used
        #y = self.model(x)
        return y
