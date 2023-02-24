

import torch 
from torch import nn
from timeit import default_timer as timer 

import random
from pathlib import Path

import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader

from tqdm.auto import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
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

BATCHSIZE=128

trainDataloader = DataLoader(trainData,
                             batch_size = BATCHSIZE,
                             shuffle = True)
testDataloader = DataLoader(testData,
                           batch_size = BATCHSIZE,
                           shuffle = False)

print(f"""Dataloaders: {trainDataloader , testDataloader}\n
        length train {len(trainDataloader)} length of test {len(testDataloader)}\n
        """)

class CIFAR10Model(nn.Module):
     def __init__(self, input_shape: tuple, num_classes: int, hidden_units: int = 64):
        super().__init__()
        self.hidden_units = hidden_units
        self.block_1 = nn.Sequential(
             nn.Conv2d(in_channels=input_shape[0], 
                               out_channels=hidden_units, 
                               kernel_size=3, 
                               stride=1, 
                               padding=1),
             nn.BatchNorm2d(hidden_units),
             nn.ReLU(),
             nn.Conv2d(in_channels=hidden_units, 
                               out_channels=hidden_units,
                               kernel_size=3,
                               stride=1,
                               padding=1),
             nn.BatchNorm2d(hidden_units),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=2,
                                  stride=2) 
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, 
                               out_channels=2*hidden_units, 
                               kernel_size=3, 
                               stride=1, 
                               padding=1),
            nn.BatchNorm2d(2*hidden_units),
            nn.ReLU(),
            nn.Conv2d(in_channels=2*hidden_units, 
                               out_channels=2*hidden_units,
                               kernel_size=3,
                               stride=1,
                               padding=1),
            nn.BatchNorm2d(2*hidden_units),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                                  stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(2*hidden_units*8*8, hidden_units*4),
            nn.BatchNorm1d(hidden_units*4),
            nn.ReLU(),
            nn.Dropout(0.5),
        
            nn.Linear(hidden_units*4, num_classes)
        )
    
     def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        x = self.block_2(x)
        x = x.view(-1, 2 * self.hidden_units*8*8)
        x = self.classifier(x)
        return x

torch.manual_seed(3)
model = CIFAR10Model(input_shape=(3, 32, 32), num_classes = len(classNames)).to(device)
print(model)
"""
def accuracy_fn(yTrue, yPred):

    correct = torch.eq(yTrue, yPred).sum().item()
    acc = (correct / len(yPred)) * 100
    return acc

lossfunc = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.005)


def training(model: torch.nn.Module,
             data_loader: torch.utils.data.DataLoader,
             loss_fn: torch.nn.Module,
             optimizer: torch.optim.Optimizer,
             accuracy_fn,
             device: torch.device = device):
    trainLoss, trainAcc = 0, 0
    for batch, (X, y ) in enumerate(data_loader):
        X, y  = X.to(device), y.to(device)
        yPred = model(X)
        loss = loss_fn(yPred, y)
        trainLoss += loss
        trainAcc += accuracy_fn(yTrue=y, yPred=yPred.argmax(dim=1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    trainLoss/=len(data_loader)
    trainAcc/=len(data_loader)
    print(f"Train loss: {trainLoss:.5f} | Train accuracy: {trainAcc:.2f}%")

def testing(data_loader: torch.utils.data.DataLoader,
            model: torch.nn.Module,
            loss_fn: torch.nn.Module,
            accuracy_fn,
            device: torch.device = device):
    testLoss, testAcc = 0,0
    model.eval()
    with torch.inference_mode():
        for X,y in data_loader:
            X, y = X.to(device), y.to(device)
            testPred = model(X)
            testLoss += loss_fn(testPred, y)
            testAcc+=accuracy_fn(yTrue = y, yPred=testPred.argmax(dim=1))
        testLoss/=len(data_loader)
        testAcc/=len(data_loader)
        print(f"Test loss: {testLoss:.5f} | Test accuracy: {testAcc:.2f}%\n")

epochs = 8
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n---------")
    training(data_loader=trainDataloader, 
        model=model, 
        loss_fn=lossfunc,
        optimizer=optimizer,
        accuracy_fn=accuracy_fn
    )
    testing(data_loader=testDataloader,
        model=model,
        loss_fn=lossfunc,
        accuracy_fn=accuracy_fn
    )
"""



def make_predictions(model: torch.nn.Module, data: list, device: torch.device = device):
    pred_probs = []
    model.eval()
    with torch.inference_mode():
        for sample in data:

            sample = torch.unsqueeze(sample, dim=0).to(device) 

            pred_logit = model(sample)

            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)

            pred_probs.append(pred_prob.cpu())
           
    return torch.stack(pred_probs)



test_samples = []
test_labels = []
for sample, label in random.sample(list(testData), k=9):
    test_samples.append(sample)
    test_labels.append(label)


pred_probs= make_predictions(model=model, 
                             data=test_samples)

pred_classes = pred_probs.argmax(dim=1)




plt.figure(figsize=(9, 9))
nrows = 3
ncols = 3
for i, sample in enumerate(test_samples):

  plt.subplot(nrows, ncols, i+1)

  image = np.transpose(sample,(1,2,0))
  image = torch.mean(image, dim=2)
  plt.imshow(image)


  pred_label = classNames[pred_classes[i]]


  truth_label = classNames[test_labels[i]] 


  title_text = f"Pred: {pred_label} | Truth: {truth_label}"
  
  if pred_label == truth_label:
      plt.title(title_text, fontsize=10, c="g") #right
  else:
      plt.title(title_text, fontsize=10, c="r") #wrong
  plt.axis(False);

plt.show()


MODEL_PATH = Path("model")
#MODEL_PATH.mkdir(parents=True,
#                 exist_ok=True)
MODEL_NAME = "Pixel Perception Model"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
#print(f"Saving model to: {MODEL_SAVE_PATH}")
#torch.save(obj=model.state_dict(),
 #          f=MODEL_SAVE_PATH)

loaded_model = CIFAR10Model(input_shape=(3, 32, 32), num_classes = len(classNames)).to(device)

loaded_model.state_dict(torch.load(f = MODEL_SAVE_PATH)) 

loaded_model = loaded_model.to(device)

print(loaded_model)