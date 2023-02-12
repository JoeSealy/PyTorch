#Import PyTorch
import torch
from torch import nn

#Import vision
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

#Utils
from torch.utils.data import DataLoader

#For visualisation
import matplotlib.pyplot as plt
import numpy as np

print(f"PyTorch version: {torch.__version__}\n torchvision version:{torchvision.__version__}")

#in this projects we'll be using CIFAR-10 as our primary dataset

trainDataset = torchvision.datasets.CIFAR10(root =  "data", 
                                       train = True,
                                       target_transform=None,
                                       transform = ToTensor(),
                                       download=True)
testDataset = torchvision.datasets.CIFAR10(root="data",
                                           train = False,
                                           download =True,
                                           transform=ToTensor())

#now inspect first picrtue in the list
image, label = trainDataset[1]
print(f"Image {image} Label {label}" )

#image shape
print(image.shape)

#how many sameples are there
print(f"TrainData: {len(trainDataset.data)}\nTrainTargets: {len(trainDataset.targets)}\nTestData: {len(testDataset.data)}\nTestTargets: {len(testDataset.targets)}" )

#class names
classNames = trainDataset.classes
print(classNames)

#Now for visualisation to see what were working with
#firstly the image is a 3d one aswell as using 3 colour channels
#youll need to make it use one colour channel and transpose it making it a 2d image
#since matplotlib can only display 1d and 2d images
#----------------------------------------------------------------
#image = np.transpose(image, (1,2,0))
#image = torch.mean(image,dim=2, keepdim=True)
#plt.imshow(image, cmap="gray")
#plt.title(classNames[label])
#plt.show()
#--------------------------------------------------------------

torch.manual_seed(3)
#fig = plt.figure(figsize = (9,9))
#rows, cols = 4, 4
#for i in range(1, rows * cols+1):
#    randomIdx = torch.randint(0, len(trainDataset), size=[1]).item()
 #   img, label = trainDataset[randomIdx]
  #  img = np.transpose(img, (1,2,0))
 #   img = torch.mean(img,dim=2, keepdim=True)
  #  fig.add_subplot(rows, cols, i)
 #   plt.imshow(img, cmap="gray")
 #   plt.title(classNames[label])
 #   plt.axis(False)
#plt.show()

#now that we can see our data
#lets put it into a dataloader
BATCH_SIZE = 128

trainDataloader = DataLoader(trainDataset,
                             batch_size = BATCH_SIZE,
                             shuffle=True)
testDataloader = DataLoader(testDataset,
                            batch_size = BATCH_SIZE,
                            shuffle = True)

#lets see it printed
print(f"""Dataloaders:{trainDataloader, testDataloader}\n
          Length of train dataloader: {len(trainDataloader)} batches of {BATCH_SIZE}\n
          Length of test dataloader: {len(testDataloader)} batch of {BATCH_SIZE}""")

#inside the training data loader
trainFeaturesBatch, trainLabelsBatch = next(iter(trainDataloader)) 
print(f"Features: {trainFeaturesBatch.shape}\n Labels: {trainLabelsBatch.shape}")

#now lets make sure we have what when want in these dataloaders
randomIdx = torch.randint(0, len(trainFeaturesBatch), size=[1]).item()
img, label = trainFeaturesBatch[randomIdx], trainLabelsBatch[randomIdx]
img = np.transpose(img, (1,2,0))
img = torch.mean(img,dim=2, keepdim=True)
plt.imshow(img, cmap="gray")
plt.title(classNames[label])
plt.axis(False)
plt.show()

#now that it works
#we can start building our base model
#you use a base model to begin with and make it more complex as time goes on
