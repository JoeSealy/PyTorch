

#import PyTorch
import torch
from torch import nn
import torch.nn.functional as F

# Import torchvision 
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

# Import matplotlib for visualization
import matplotlib.pyplot as plt

#dataloader
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

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

image, label = trainData[9]
print(f"Image shape: {image.shape}")
plt.imshow(image.squeeze()) # image shape is [1, 28, 28] (colour channels, height, width)
plt.title(label);


torch.manual_seed(3)
fig = plt.figure(figsize=(9, 9))
rows, cols = 4, 4
for i in range(1, rows * cols + 1):
    random_idx = torch.randint(0, len(trainData), size=[1]).item()
    img, label = trainData[random_idx]
    fig.add_subplot(rows, cols, i)
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(class_names[label])
    plt.axis(False);



BATCHSIZE = 32

trainDataloader = DataLoader(trainData,
                             batch_size = BATCHSIZE,
                             shuffle=True)
testDataloader = DataLoader(testData,
                            batch_size = BATCHSIZE,
                            shuffle=False)
print(f"Dataloaders: {trainDataloader, testDataloader}")
print(f"Length train data: {len(trainDataloader)} batches of {BATCHSIZE}")
print(f"Length tEST data: {len(testDataloader)} batches of {BATCHSIZE}")

class EMNISTModel(nn.Module):
    def __init__(self, input_shape: int, output_shape: int):
        super(EMNISTModel, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=input_shape, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(in_features=128*3*3, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=output_shape)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = x.view(-1, 128*3*3)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = EMNISTModel(input_shape = 1, output_shape=len(class_names)).to(device)
print(model)
print(next(model.parameters()).device)

lossFunc = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


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
        # Send data to GPU
        X, y = X.to(device), y.to(device)
        
        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_pred.argmax(dim=1)) # Go from logits -> pred labels
     
        # 3. Optimizer zero grad
        optimizer.zero_grad()
   
        # 4. Loss backward
        loss.backward()
     
        # 5. Optimizer step
        optimizer.step()
        i += 1 
        if i % 1000 == 0:
            print(f"Iteration {i}")

    # Calculate loss and accuracy per epoch and print out what's happening
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
            # Send data to GPU
            X, y = X.to(device), y.to(device)
            # 1. Forward pass
            test_pred = model(X)

            # 2. Calculate loss and accuracy
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y,
                y_pred=test_pred.argmax(dim=1) # Go from logits -> pred labels
            )
            i += 1 
            if i % 1000 == 0:
                print(f"Iteration {i}")
        # Adjust metrics and print out
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")


epochs = 3
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