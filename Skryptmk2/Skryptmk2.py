import torch
from torch import nn

#Dataloader
from torch.utils.data import DataLoader, SubsetRandomSampler

# Import torchvision 
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

#path to file
from pathlib import Path

#visualisation
import matplotlib.pyplot as plt

print(f"PyTorch version: {torch.__version__}\ntorchvision version: {torchvision.__version__}")


device = 'cuda' if torch.cuda.is_available() else 'cpu'


import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class IAMHandwritingDatasetCompose(Dataset):
    def __init__(self, ROOT_DIR):
        self.HWDATA_DIR = ROOT_DIR + "/IAMHandwriting"
        self.HWOLDATA_DIR = ROOT_DIR + "/IAMOLHandwriting"

    def Handwriting(self, HWDATA_DIR):

        #this is the struct, reading from an xml file
        #-------------------example-----------------------------------------
#        self.line_1 = [LINE_IMAGE_PATH, LINE_ID_IMAGE, LINE_TRUE_LABEL,
#                       [WORD_IMAGE_PATH, WORD_ID_IMAGE,WORD_TRUE_LABEL,TAG,
#                        [H,W,X,Y],[H,W,X,Y],[H,W,X,Y],[H,W,X,Y],[H,W,X,Y]]]









ROOT_DIR = "Database"

class EMNISTModel(nn.Module):
    def __init__(self, input_shape: int, output_shape: int):
        super(EMNISTModel, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=input_shape, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=128)
        
        self.fc1 = nn.Linear(in_features=128*3*3, out_features=256)
        self.bn4 = nn.BatchNorm1d(num_features=256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.bn5 = nn.BatchNorm1d(num_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=output_shape)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, kernel_size=2)
        x = x.view(-1, 128*3*3)
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x