import torch
from torch import nn


class SimpleClassifierWBNormBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(5, 5), stride=stride, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor):
        return self.block(x)


class SimpleClassifierWBNorm(nn.Module):
    def __init__(self, block, num_classes: int = 23):
        super(SimpleClassifierWBNorm, self).__init__()
        self.conv1 = block(3, 32, 3)
        self.conv2 = block(32, 32, 3)
        self.conv3 = block(32, 32, 3)
        self.conv4 = block(32, 32, 3)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=5*5*32, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=num_classes)
        )

    def forward(self, x: torch.Tensor):
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.classifier(output)
        return output

def simple_classifier(num_classes: int = 23):
    return SimpleClassifierWBNorm(SimpleClassifierWBNormBlock, num_classes)