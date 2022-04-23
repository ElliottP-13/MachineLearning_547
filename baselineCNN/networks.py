import os.path
import torch.nn.functional as F
import torch
import torch.nn as nn
import math
import torch.optim as optim
from tqdm import tqdm
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import argparse
from torch.nn import init


class Net(nn.Module):
    def __init__(self, input_shape=(224, 224), num_classes=12):
        #
        super().__init__()

        conv_layers = []

        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        self.conv1 = nn.Conv2d(3, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        # Second Convolution Block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        # Second Convolution Block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4]

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=64, out_features=num_classes)

        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)

    # ----------------------------
    # Forward pass computations
    # ----------------------------
    def forward(self, x):
        # Run the convolutional blocks
        x = self.conv(x)

        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)
        x = x.view(x.shape[0], -1)

        # Linear layer
        x = self.lin(x)

        # Final output
        return x


class Net2(nn.Module):
    def __init__(self, input_shape=(224, 224), num_classes=12):
        #
        super().__init__()

        conv_layers = []

        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        self.conv1a = nn.Conv2d(3, 8, kernel_size=(5, 5), stride=(2, 2), padding=(0, 0))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)
        conv_layers += [self.conv1a, self.relu1, self.bn1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        conv_layers += [self.conv2, self.relu2, self.bn2]

        # Third Convolution Block
        self.conv3a = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.conv3b = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        self.bn3 = nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv3b.weight, a=0.1)
        self.conv3b.bias.data.zero_()
        conv_layers += [self.conv3a, self.relu3, self.pool3, self.conv3b, self.bn3]

        # fourth Convolution Block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        conv_layers += [self.conv4, self.relu4, self.bn4]

        # fifth Convolution Block
        self.conv5a = nn.Conv2d(64, 84, kernel_size=(2, 2), stride=(1, 1), padding=(0, 0))
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(kernel_size=2)
        self.conv5b = nn.Conv2d(84, 128, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
        self.bn5 = nn.BatchNorm2d(128)
        init.kaiming_normal_(self.conv5b.weight, a=0.1)
        self.conv5b.bias.data.zero_()
        conv_layers += [self.conv5a, self.relu5, self.pool5, self.conv5b, self.bn5]

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)

        self.lin = nn.Linear(in_features=128, out_features=num_classes)

        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)

    # ----------------------------
    # Forward pass computations
    # ----------------------------
    def forward(self, x):
        # Run the convolutional blocks
        x = self.conv(x)

        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)
        x = x.view(x.shape[0], -1)

        # Linear layer
        x = self.lin(x)

        # Final output
        return x
