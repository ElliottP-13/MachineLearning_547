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
from torchvision import models
from torch.nn import init


def compute_size(input_size, conv_layers):
    """
    Computes the final output shape of a set of convolutional layers
    @param input_size:
    @param conv_layers:
    @return:
    """
    w, h = input_size

    for layer in conv_layers:
        kernel, stride, padding = layer
        w = math.floor((w - kernel + 2 * padding) / stride + 1)
        h = math.floor((h - kernel + 2 * padding) / stride + 1)

    return w * h


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


class Net3(nn.Module):
    def __init__(self, input_shape=(224, 224), num_classes=12):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=6, out_channels=12, kernel_size=2, stride=1, padding=0),
                                   nn.BatchNorm2d(12), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=12, out_channels=16, kernel_size=3, stride=1, padding=0),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=20, kernel_size=5, stride=1, padding=0),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=20, out_channels=32, kernel_size=2, stride=1, padding=0),
                                   nn.BatchNorm2d(32), nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        conv_layers = [(3, 1, 0), (2, 1, 0), (3, 1, 0), (2, 2, 0), (5, 1, 0), (2, 1, 0), (2, 2, 0)]

        self.fc1 = nn.Sequential(nn.Linear(compute_size(input_shape, conv_layers) * 32, 120),
                                 nn.Tanh())
        self.fc2 = nn.Sequential(nn.Linear(120, num_classes),
                                 nn.Softmax(dim=1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class Net4(nn.Module):
    def __init__(self, model_name, num_classes=12, use_pretrained=True):
        """

        @param model_name: resnet, alexnet, vgg, squeezenet, densenet
        @param num_classes:
        @param use_pretrained:
        """
        super().__init__()
        # Initialize these variables which will be set in this if statement. Each of these
        #   variables is model specific.
        model_ft = None
        input_size = 0

        if model_name == "resnet":
            """ Resnet50
            """
            model_ft = models.resnet50(pretrained=use_pretrained)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "alexnet":
            """ Alexnet
            """
            model_ft = models.alexnet(pretrained=use_pretrained)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
            input_size = 224

        elif model_name == "vgg":
            """ VGG11_bn
            """
            model_ft = models.vgg16(pretrained=use_pretrained)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
            input_size = 224

        elif model_name == "squeezenet":
            """ Squeezenet
            """
            model_ft = models.squeezenet1_0(pretrained=use_pretrained)
            model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
            model_ft.num_classes = num_classes
            input_size = 224

        elif model_name == "densenet":
            """ Densenet
            """
            model_ft = models.densenet121(pretrained=use_pretrained)
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        self.model = model_ft


    def forward(self, x):
        return self.model(x)