import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import Block, Linear, ReLU, Sigmoid, Dropout, Sequential


class MLP(Block):
    """
    A simple multilayer perceptron model based on our custom Blocks.
    Architecture is (with ReLU activation):

        FC(in, h1) -> ReLU -> FC(h1,h2) -> ReLU -> ... -> FC(hn, num_classes)

    Where FC is a fully-connected layer and h1,...,hn are the hidden layer
    dimensions.
    If dropout is used, a dropout layer is added after every activation
    function.
    """
    def __init__(self, in_features, num_classes, hidden_features=(),
                 activation='relu', dropout=0, **kw):
        super().__init__()
        """
        Create an MLP model Block.
        :param in_features: Number of features of the input of the first layer.
        :param num_classes: Number of features of the output of the last layer.
        :param hidden_features: A sequence of hidden layer dimensions.
        :param activation: Either 'relu' or 'sigmoid', specifying which 
        activation function to use between linear layers.
        :param: Dropout probability. Zero means no dropout.
        """
        blocks = []

        # TODO: Build the MLP architecture as described.
        # ====== YOUR CODE: ======
        raise NotImplementedError()
        # ========================

        self.sequence = Sequential(*blocks)

    def forward(self, x, **kw):
        return self.sequence(x, **kw)

    def backward(self, dout):
        return self.sequence.backward(dout)

    def params(self):
        return self.sequence.params()

    def train(self, training_mode=True):
        self.sequence.train(training_mode)

    def __repr__(self):
        return f'MLP, {self.sequence}'


class ConvClassifier(nn.Module):
    """
    A convolutional classifier model based on PyTorch nn.Modules.

    The architecture is:
    [(Conv -> ReLU)*P -> MaxPool]*(N/P) -> (Linear -> ReLU)*M -> Linear
    """
    def __init__(self, in_size, out_classes, filters, pool_every, hidden_dims):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param filters: A list of of length N containing the number of
            filters in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        """
        super().__init__()
        self.in_size = in_size
        self.out_classes = out_classes
        self.filters = filters
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims

        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        # [(Conv -> ReLU)*P -> MaxPool]*(N/P)
        # Use only dimension-preserving 3x3 convolutions. Apply 2x2 Max
        # Pooling to reduce dimensions.
        # ====== YOUR CODE: ======

        N = len(self.filters)
        P = self.pool_every

        kernel_size_conv, stride_conv, padding_conv = 3, 1, 1
        kernel_size_pool, stride_pool, padding_pool = 2, 2, 0

        for i in range(N // P):  # (Conv -> ReLU)*P -> MaxPool
            for j in range(P):  # Conv -> ReLU
                if i == 0 and j == 0:
                    layers.append(nn.Conv2d(in_channels=in_channels, out_channels=self.filters[0],
                                            kernel_size=kernel_size_conv, stride=stride_conv, padding=padding_conv))
                else:
                    layers.append(nn.Conv2d(in_channels=self.filters[i*P + j - 1], out_channels=self.filters[i*P + j],
                                            kernel_size=kernel_size_conv, stride=stride_conv, padding=padding_conv))

                layers.append(nn.ReLU())

            layers.append(nn.MaxPool2d(kernel_size=kernel_size_pool, stride=stride_pool, padding=padding_pool))

        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the classifier part of the model:
        # (Linear -> ReLU)*M -> Linear
        # You'll need to calculate the number of features first.
        # The last Linear layer should have an output dimension of out_classes.
        # ====== YOUR CODE: ======
        M = len(self.hidden_dims)
        N = len(self.filters)
        P = self.pool_every

        features_num = self.filters[-1] * (in_h // 2 ** (N // P)) * (in_w // 2 ** (N // P))

        layers.append(nn.Flatten())

        for i in range(M):  # Linear -> ReLU
            if i == 0:
                layers.append(nn.Linear(features_num, self.hidden_dims[0]))
            else:
                layers.append(nn.Linear(self.hidden_dims[i - 1], self.hidden_dims[i]))

            layers.append(nn.ReLU())

        # Linear
        layers.append(nn.Linear(self.hidden_dims[-1], self.out_classes))
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        # TODO: Implement the forward pass.
        # Extract features from the input, run the classifier on them and
        # return class scores.
        # ====== YOUR CODE: ======
        f = self.feature_extractor(x)
        out = self.classifier(f)
        # ========================
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class YourCodeNet(ConvClassifier):
    def __init__(self, in_size, out_classes, filters, pool_every, hidden_dims):
        super().__init__(in_size, out_classes, filters, pool_every, hidden_dims)

    # TODO: Change whatever you want about the ConvClassifier to try to
    # improve it's results on CIFAR-10.
    # For example, add batchnorm, dropout, skip connections, change conv
    # filter sizes etc.
    # ====== YOUR CODE: ======
    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        # [(Conv -> ReLU)*P -> MaxPool]*(N/P)
        # Use only dimension-preserving 3x3 convolutions. Apply 2x2 Max
        # Pooling to reduce dimensions.
        # ====== YOUR CODE: ======

        N = len(self.filters)
        P = self.pool_every

        kernel_size_conv, stride_conv, padding_conv = 3, 1, 1
        kernel_size_pool, stride_pool, padding_pool = 2, 2, 0

        for i in range(N // P):  # (Conv -> ReLU)*P -> MaxPool
            for j in range(P):  # Conv -> ReLU
                k = i * P + j

                if j == P - 1 and i != (N // P) - 1:
                    # Skip connection
                    layers.append(
                        nn.Conv2d(in_channels=self.filters[k - 1], out_channels=self.filters[k],
                                  kernel_size=1, stride=stride_conv, padding=0))
                    layers.append(nn.BatchNorm2d(self.filters[k]))  # Batch normalization
                    continue

                if k == 0:
                    layers.append(nn.Conv2d(in_channels=in_channels, out_channels=self.filters[0],
                                            kernel_size=kernel_size_conv, stride=stride_conv, padding=padding_conv))
                else:
                    layers.append(
                        nn.Conv2d(in_channels=self.filters[k - 1], out_channels=self.filters[k],
                                  kernel_size=kernel_size_conv, stride=stride_conv, padding=padding_conv))

                layers.append(nn.BatchNorm2d(self.filters[i * P + j]))  # Batch normalization
                layers.append(nn.ReLU())

            layers.append(nn.MaxPool2d(kernel_size=kernel_size_pool, stride=stride_pool))

        # ========================
        seq = nn.Sequential(*layers)
        return seq
    # ========================
