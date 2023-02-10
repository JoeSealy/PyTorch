import torch 
import os
import pathlib
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from PIL import Image
from typing import Tuple, Dict, List
from torchvision import transforms
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

# Get class names as a list
class_names = trainData.classes
class_names


print(f"Train data:\n{trainData}\nTest data:\n{testData}")

print(trainData.classes)
print(trainData.class_to_idx)

print(len(trainData), len(testData))

trainDataloader = DataLoader(dataset= trainData,
                             batch_size=1,
                             num_workers=1,
                             shuffle=False)

testDataloader = DataLoader(dataset=testData,
                            batch_size=1,
                            num_workers=1,
                            shuffle=False)

print(trainDataloader, testDataloader)

#trainDataImg, trainDataLabel = next(iter(trainDataloader))
#print(f"image shape{trainDataImg.shape}", f"image shape{trainDataLabel.shape}")



#2--------------------------------------------------------
# Setup path for target directory
target_directory = trainDir
print(f"Target directory: {target_directory}")

# Get the class names from the target directory
class_names_found = sorted([entry.name for entry in list(os.scandir(image_path / "train"))])
print(f"Class names found: {class_names_found}")

# Make function to find classes in target directory
def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folder names in a target directory.
    
    Assumes target directory is in standard image classification format.

    Args:
        directory (str): target directory to load classnames from.

    Returns:
        Tuple[List[str], Dict[str, int]]: (list_of_class_names, dict(class_name: idx...))
    
    Example:
        find_classes("food_images/train")
        >>> (["class_1", "class_2"], {"class_1": 0, ...})
    """
    # 1. Get the class names by scanning the target directory
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    
    # 2. Raise an error if class names not found
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")
        
    # 3. Crearte a dictionary of index labels (computers prefer numerical rather than string labels)
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


class ImageFolderCustom(Dataset):
    def __init__(self, targ_dir:str, transform=None) -> None:
        self.paths = list(pathlib.Path(targ_dir).glob("*/*.jpg"))
        self.transform = transform
        self.classes, self.class_to_idx = find_classes(targ_dir)

    def load_image(self, index:int) -> Image.Image:
        image_path = self.paths[index]
        return Image.open(image_path)
    def __len__(self) -> int:
        return len(self.paths)
    def __getitem__(self, index:int) -> Tuple[torch.Tensor, int]:
        img = self.load_image(index)
        class_name = self.paths[index].parent.name
        class_idx = self.class_to_idx[class_name]
        
        if self.transform:
            return self.transform(img), class_idx
        else:
            return img,class_idx

trainTransforms = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
    ])

testTransforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
    ])

trainDataCustom = ImageFolderCustom(targ_dir= trainDir,
                                    transform = trainTransforms)
testDataCustom = ImageFolderCustom(targ_dir= testDir,
                                    transform = testTransforms)

print(trainDataCustom, testDataCustom)

print(len(trainDataCustom), len(testDataCustom))
print(trainDataCustom.classes, trainDataCustom.class_to_idx)

# Check for equality amongst our custom Dataset and ImageFolder Dataset
print((len(trainDataCustom) == len(trainData)) & (len(testDataCustom) == len(testData)))
print(trainDataCustom.classes == trainData.classes)
print(trainDataCustom.class_to_idx == trainData.class_to_idx)

def display_random_images(dataset: torch.utils.data.dataset.Dataset,
                          classes: List[str] = None,
                          n:int = 10,
                          display_shape: bool = True,
                          seed: int = None):
    if n > 10:
        n = 10
        display_shape = False
        print("set to 10 removing shape")

    if seed:
        random.seed(seed)

    random_samples_idx = random.sample(range(len(dataset)), k=n)

    plt.figure(figsize = (16, 8))

    for i, targ_sample in enumerate(random_samples_idx):
        targ_image, targ_label = dataset[targ_sample][0], dataset[targ_sample][1]

        targ_image_adjust = targ_image.permute(1,2,0)

        plt.subplot(1, n, i+1)
        plt.imshow(targ_image_adjust)
        plt.axis("off")
        if classes:
            title = f"class: {classes[targ_label]}"
            if display_shape:
                title = title + f"\n shape: {targ_image_adjust.shape}"
        plt.title(title)

display_random_images(trainData, 
                      n=5, 
                      classes=class_names,
                      seed=None)


display_random_images(trainDataCustom, 
                      n=12, 
                      classes=class_names,
                      seed=None) # Try setting the seed for reproducible images

trainDataloaderCustom = DataLoader(dataset=trainDataCustom,
                                    batch_size=1,
                                    num_workers=0,
                                    shuffle=True)
testDataloaderCustom = DataLoader(dataset=testDataCustom,
                                    batch_size=1,
                                    num_workers=0,
                                    shuffle=False)
print(trainDataloaderCustom, testDataloaderCustom)


img_custom, label_custom = next(iter(trainDataloaderCustom))

print(f"Image shape: {img_custom.shape} -> [batch_size, color_channels, height, width]")
print(f"Label shape: {label_custom.shape}")

trainTransforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31), # how intense 
    transforms.ToTensor() # use ToTensor() last to get everything between 0 & 1
    ])

testTransforms = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor()
])


simpleTransform = transforms.Compose([ 
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

trainDataSimple = datasets.ImageFolder(root=trainDir, transform=simpleTransform)
testDataSimple = datasets.ImageFolder(root=testDir, transform=simpleTransform)

BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()
print(f"Dataloader working with Batch{BATCH_SIZE} and {NUM_WORKERS}")

trainDataloaderSimple = DataLoader(trainDataSimple, 
                                     batch_size=BATCH_SIZE, 
                                     shuffle=True, 
                                     num_workers=NUM_WORKERS)

testDataloaderSimple = DataLoader(testDataSimple, 
                                    batch_size=BATCH_SIZE, 
                                    shuffle=False, 
                                    num_workers=NUM_WORKERS)

print(trainDataloaderSimple, testDataloaderSimple)