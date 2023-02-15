from pydoc import classname
import torch 
from torch import nn
from timeit import default_timer as timer 


import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader

from tqdm.auto import tqdm


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
model = CIFAR10Model(inputShape = 3072, hiddenUnits = 10, outputShape = len(classNames))
print(model)

def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.
    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.
    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return 

lossfunc = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)

torch.manual_seed(3)
epochs = 3

for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n------")
    
    trainLoss = 0

    for batch, (X, y ) in enumerate(trainDataloader):
        model.train()

        yPred = model(X)
        loss = lossfunc(yPred, y)
        trainLoss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 400 ==0:
            print(f"Looked at {batch * len(X)}/{len(trainDataloader.dataset)} samples")
        testLoss, testAcc  = 0, 0
        model.eval()
        with torch.inference_mode():
            for X,y in testDataloader:
                testPred = model(X)
                testLoss += lossfunc(testPred, y)
                testAcc += accuracy_fn(y_true = y, y_pred=testPred.argmax(dim=1))
            testLoss /= len(testDataloader)
            testAcc /= len(testDataloader)
        print(f"\nTrain loss: {trainLoss:.5f} | Test loss: {testLoss:.5f}, Test acc: {testAcc:.2f}%\n")