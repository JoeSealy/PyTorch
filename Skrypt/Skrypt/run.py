

#import PyTorch
from random import sample
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
# Import torchvision 
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

# Import matplotlib for visualization
import matplotlib.pyplot as plt

#dataloader
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm.auto import tqdm
from pathlib import Path

from PIL import Image
from PIL import ImageOps

import numpy as np

import os
from skimage.feature import canny
import cv2

from skimage import morphology
# Check versions
# Note: your PyTorch version shouldn't be lower than 1.10.0 and torchvision version shouldn't be lower than 0.11
print(f"PyTorch version: {torch.__version__}\ntorchvision version: {torchvision.__version__}")


device = 'cuda' if torch.cuda.is_available() else 'cpu'

trainData = datasets.EMNIST(
    root = "data",
    split= "byclass",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=None
)

testData = datasets.EMNIST(
    root = "data",
    split = "byclass",
    train=False,
    download=True,
    transform=ToTensor())

image, label = trainData[0]
print(image, label)
print(image.shape)
print(len(trainData.data), len(trainData.targets), len(testData.data), len(testData.targets))
class_names = trainData.classes

print(class_names)

trainSamplesNum = 300000
trainSampler = SubsetRandomSampler(range(trainSamplesNum))
testSamplesNum = 50000
testSampler = SubsetRandomSampler(range(testSamplesNum))

BATCHSIZE = 128

trainDataloader = DataLoader(trainData,
                             batch_size = BATCHSIZE,
                             sampler=trainSampler)
testDataloader = DataLoader(testData,
                            batch_size = BATCHSIZE,
                            sampler=testSampler)
print(f"Dataloaders: {trainDataloader, testDataloader}")
print(f"Length train data: {len(trainDataloader)} batches of {BATCHSIZE}")
print(f"Length tEST data: {len(testDataloader)} batches of {BATCHSIZE}")

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


"""
model = EMNISTModel(input_shape = 1, output_shape=len(class_names)).to(device)
print(model)
print(next(model.parameters()).device)
"""
lossFunc = nn.CrossEntropyLoss()
"""
optimizer = torch.optim.Adam(params=model.parameters(), 
                            lr=0.004)
"""
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

"""
torch.manual_seed(3)


def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):
    train_loss, train_acc = 0, 0
    i=0
    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        
        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_pred.argmax(dim=1))
     
        optimizer.zero_grad()
   
        loss.backward()
     
        optimizer.step()
        i += 1 
        if i % 100 == 0:
            print(f"Iteration {i}")
    print("7")
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")


def test_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device = device):


    test_loss, test_acc = 0, 0
    model.eval() 
    i=0
    with torch.inference_mode(): 
        for X, y in data_loader:

            X, y = X.to(device), y.to(device)

            test_pred = model(X)

            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y,
                y_pred=test_pred.argmax(dim=1) 
            )
            i += 1 
            if i % 50 == 0:
                print(f"Iteration {i}")

        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")


epochs = 10
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n---------")
    train_step(data_loader=trainDataloader, 
        model=model, 
        loss_fn=lossFunc,
        optimizer=optimizer,
        accuracy_fn=accuracy_fn,
        device = device
    )
    test_step(data_loader=testDataloader,
        model=model,
        loss_fn=lossFunc,
        accuracy_fn=accuracy_fn,
        device = device
    )


def make_predictions(model: torch.nn.Module, data: list, device: torch.device = device):
    pred_probs = []
    model.eval()
    with torch.inference_mode():
        for sample in data:
            # Prepare sample
            sample = torch.unsqueeze(sample, dim=0).to(device) # Add an extra dimension and send sample to device

            # Forward pass (model outputs raw logit)
            pred_logit = model(sample)

            # Get prediction probability (logit -> prediction probability)
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)

            # Get pred_prob off GPU for further calculations
            pred_probs.append(pred_prob.cpu())
            
    # Stack the pred_probs to turn list into a tensor
    return torch.stack(pred_probs)

import random
random.seed(42)
test_samples = []
test_labels = []
for sample, label in random.sample(list(testData), k=9):
    test_samples.append(sample)
    test_labels.append(label)

# View the first test sample shape and label
print(f"Test sample image shape: {test_samples[0].shape}\nTest sample label: {test_labels[0]} ({class_names[test_labels[0]]})")


# Make predictions on test samples with model 2
pred_probs= make_predictions(model=model, 
                             data=test_samples)

# Turn the prediction probabilities into prediction labels by taking the argmax()
pred_classes = pred_probs.argmax(dim=1)





# Plot predictions
plt.figure(figsize=(9, 9))
nrows = 3
ncols = 3
for i, sample in enumerate(test_samples):
  # Create a subplot
  plt.subplot(nrows, ncols, i+1)

  # Plot the target image
  plt.imshow(sample.squeeze(), cmap="gray")

  # Find the prediction label (in text form, e.g. "Sandal")
  pred_label = class_names[pred_classes[i]]

  # Get the truth label (in text form, e.g. "T-shirt")
  truth_label = class_names[test_labels[i]] 

  # Create the title text of the plot
  title_text = f"Pred: {pred_label} | Truth: {truth_label}"
  
  # Check for equality and change title colour accordingly
  if pred_label == truth_label:
      plt.title(title_text, fontsize=10, c="g") # green text if correct
  else:
      plt.title(title_text, fontsize=10, c="r") # red text if wrong
  plt.axis(False);

plt.show()


from pathlib import Path

# Create models directory (if it doesn't already exist), see: https://docs.python.org/3/library/pathlib.html#pathlib.Path.mkdir
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, # create parent directories if needed
                 exist_ok=True # if models directory already exists, don't error
)

# Create model save path
MODEL_NAME = "ENMIST_test_model"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model.state_dict(), # only saving the state_dict() only saves the learned parameters
           f=MODEL_SAVE_PATH)
"""

def eval_model(model: torch.nn.Module, 
               data_loader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               accuracy_fn,
               device: torch.device):
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:

            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            

            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y, 
                                y_pred=y_pred.argmax(dim=1)) 
        
        loss /= len(data_loader)
        acc /= len(data_loader)
        
    return {"model_name": model.__class__.__name__, 
            "model_loss": loss.item(),
            "model_acc": acc}


MODEL_PATH = Path("models")
MODEL_NAME = "ENMIST_test_model"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
loaded_model = EMNISTModel(input_shape = 1, output_shape=len(class_names)).to(device)

loaded_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

loaded_model = loaded_model.to(device)


torch.manual_seed(42)

loaded_model_results = eval_model(
    model=loaded_model,
    data_loader=testDataloader,
    loss_fn=lossFunc, 
    accuracy_fn=accuracy_fn,
    device = device
)

print(loaded_model_results)

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

def images_to_tensor(folder_path: str, img_size: int = 28) -> torch.Tensor:

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        #transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # duplicate the single channel for compatibility with pretrained models
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    images = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        with Image.open(img_path) as img:
            # Apply the transform to the image
            img_tensor = transform(img)
            # Crop the image to remove unwanted borders or noise
            img_tensor = TF.resized_crop(img_tensor, top=3, left=3, height=20, width=20, size=(28,28))
            img_tensor[img_tensor < 0.5] = 0
            # Enhance the contrast and edges of the image using the Canny filter from skimage
            img_np = img_tensor.numpy()[0] # Convert the tensor to a numpy array
            img_edges = canny(img_np, sigma=1)
            img_edges = torch.from_numpy(img_edges).unsqueeze(0) # Convert the numpy array back to a tensor
            img_tensor = TF.adjust_contrast(img_tensor, contrast_factor=2)
            img_tensor = img_edges.float() * img_tensor
            # Normalize the pixel values and append to the list of images
            images.append(TF.normalize(img_tensor, (0.1307,), (0.3081,)))
    # Stack the list of images into a single tensor and return
    return images

folder_path = "letters/alphabetcaps"
images = images_to_tensor(folder_path)

random.seed(42)
test_samples = []
test_labels = []
for sample, label in random.sample(list(testData), k=9):
    test_samples.append(sample)
    test_labels.append(label)

images = images + test_samples

pred_probs= make_predictions(model=loaded_model, 
                             data=images)

pred_classes = pred_probs.argmax(dim=1)

plt.figure(figsize=(18, 18))
nrows = 6
ncols = 6
for i, image in enumerate(images):

  plt.subplot(nrows, ncols, i+1)
  plt.imshow(image.squeeze(), cmap="gray")
  pred_label = class_names[pred_classes[i]]
  title_text = f"Pred: {pred_label}"
  plt.title(title_text, fontsize=10, c="g")


plt.show()

