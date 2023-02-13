from pydoc import classname
import torch 
from torch import nn

import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader

print(f"PyTorch version: {torch.__version__}\ntorchvision version: {torchvision.__version__}")


trainData = datasets.CIFAR10(
    root="data", # where to download data to?
    train=True, # get training data
    download=True, # download data if it doesn't exist on disk
    transform=ToTensor(), # images come as PIL format, we want to turn into Torch tensors
    target_transform=None # you can transform labels as well
)

# Setup testing data
testData = datasets.CIFAR10(
    root="data",
    train=False, # get test data
    download=True,
    transform=ToTensor()
)

image, label = trainData[0]
print(image, label)

print(len(trainData.data), len(trainData.targets), len(testData.data), len(testData.targets))

classNames = trainData.classes
print(classNames)

BATCHSIZE=32

trainDataloader = DataLoader(trainData,
                             batch_size = BATCHSIZE,
                             shuffle = True)
testDataloader = DataLoader(testData,
                           batch_size = BATCHSIZE,
                           shuffle = False)

print(f"""Dataloaders: {trainDataloader , testDataloader}\n
        length train {len(trainDataloader)} length of test {len(testDataloader)}\n
        """)

trainFtrsBatch, trainLabelsBatch = next(iter(trainDataloader))

flattenModel = nn.Flatten()

x = trainFtrsBatch[0]

forwardtren = flattenModel(x)

print(forwardtren.shape) #[3, 1024]

class CIFAR10Model(nn.Module):
    def __init__(self, inputShape: int, hiddenUnits: int, outputShape:int):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=inputShape, out_features = hiddenUnits),
            nn.Linear(in_features=inputShape, out_features = outputShape)
            )
    def forward(self, x):
        return self.layer(x)

torch.manual_seed(3)
model = CIFAR10Model(inputShape = 1024, hiddenUnits = 10, outputShape = len(classNames))
print(model)
