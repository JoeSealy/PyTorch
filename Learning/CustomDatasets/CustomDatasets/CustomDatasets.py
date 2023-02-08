import torch 
import os
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
torch.__version__

device = "cuda" if torch.cuda.is_available else "cpu"

import requests
import zipfile
from pathlib import Path

# Setup path to data folder
data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

# If the image folder doesn't exist, download it and prepare it... 
if image_path.is_dir():
    print(f"{image_path} directory exists.")
else:
    print(f"Did not find {image_path} directory, creating one...")
    image_path.mkdir(parents=True, exist_ok=True)
    
    # Download pizza, steak, sushi data
    with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
        request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
        print("Downloading pizza, steak, sushi data...")
        f.write(request.content)

    # Unzip pizza, steak, sushi data
    with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
        print("Unzipping pizza, steak, sushi data...") 
        zip_ref.extractall(image_path)

def walk_through_dir(dir_path):
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


walk_through_dir(image_path)


trainDir = image_path/"train"
testDir = image_path/"test"

print(trainDir, testDir)

import random

from PIL import Image

random.seed(3)

imagePathList = list(image_path.glob("*/*/*.jpg"))
randomImagePath=random.choice(imagePathList)
imageClass = randomImagePath.parent.stem
img = Image.open(randomImagePath)
print(f"Random image path: {randomImagePath}")
print(f"Image class: {imageClass}")
print(f"Image height: {img.height}") 
print(f"Image width: {img.width}")

# Turn the image into an array
img_as_array = np.asarray(img)

# Plot the image with matplotlib
plt.figure(figsize=(10, 7))
plt.imshow(img_as_array)
plt.title(f"Image class: {imageClass} | Image shape: {img_as_array.shape} -> [height, width, color_channels]")
plt.axis(False);
#plt.show()


dataTransform = transforms.Compose([
    transforms.Resize(size=(64,64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()])

trainData = datasets.ImageFolder(root=trainDir,
                                transform=dataTransform,
                                target_transform=None)
testData = datasets.ImageFolder(root=testDir,
                                transform=dataTransform)

print(f"Train data:\n{trainData}\nTest data:\n{testData}")

print(trainData.classes)
print(trainData.class_to_idx)

print(len(trainData), len(testData))

