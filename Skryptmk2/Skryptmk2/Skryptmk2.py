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

class IAMHandwritingDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []

        # loop over all subfolders and get the image paths and labels
        for subfolder in os.listdir(root_dir):
            if os.path.isdir(os.path.join(root_dir, subfolder)):
                for file_name in os.listdir(os.path.join(root_dir, subfolder)):
                    if file_name.endswith('.png'):
                        image_path = os.path.join(root_dir, subfolder, file_name)
                        label_file = os.path.join(root_dir, subfolder, file_name.replace('.png', '.txt'))

                        with open(label_file, 'r') as f:
                            label = f.read().strip()

                        self.image_paths.append(image_path)
                        self.labels.append(label)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(image_path).convert('L')
        # preprocess the image here

        # convert label to numerical label
        numerical_label = [ord(char) - ord('a') for char in label]

        return image, numerical_label

    def __len__(self):
        return len(self.image_paths)

# create the dataset and dataloader
dataset = IAMHandwritingDataset('path/to/iam-handwriting-database')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)



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