#02_PyTorchExerciseNeuaralNetworkClassification


# Import torch
import torch

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Setup random seed
RANDOM_SEED = 42

# Create a dataset with Scikit-Learn's make_moons()
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

X_moon, y_moon = make_moons(n_samples = 1000,random_state=RANDOM_SEED)

# Turn data into a DataFrame
import pandas as pd

moons = pd.DataFrame({"col1":X_moon[:,0],
                      "col2":X_moon[:,1],
                      "label": y_moon})

print(moons.head(10))

# Visualize the data on a scatter plot
import matplotlib.pyplot as plt

plt.scatter(x=X_moon[:, 0],
            y=X_moon[:, 1],
            c=y_moon,
            cmap=plt.cm.RdYlBu);
plt.show()

# Turn data into tensors of dtype float
X_moon = torch.from_numpy(X_moon).type(torch.float)
y_moon = torch.from_numpy(y_moon).type(torch.float)

# Split the data into train and test sets (80% train, 20% test)
from sklearn.model_selection import train_test_split

X_train_moon, y_train_moon, X_test_moon, y_test_moon = train_test_split(X_moon, y_moon, test_size=0.2, random_state = RANDOM_SEED)

import torch
from torch import nn

# Inherit from nn.Module to make a model capable of fitting the mooon data
class MoonModelV0(nn.Module):
    def __init__(self, input_features, output_features, hidden_units):
        super().__init__()
        self.Linear_Layer_stack = nn.Sequential(
            nn.Linear(in_features= input_features, out_features = hidden_units),
            nn.ReLU(),
            nn.Linear(in_features= hidden_units, out_features= hidden_units),
            nn.ReLU(),
            nn.Linear(in_features= hidden_units, out_features = output_features)
        )
    def forward(self, x):
        return self.Linear_Layer_Stack(x)

# Instantiate the model
model = MoonModelV0(input_features = 2, output_features = 1, hidden_units=5).to(device)

# Setup loss function
loss_fn = nn.BCEWithLogitsLoss()

# Setup optimizer to optimize model's parameters
optimizer = torch.optim.SGD(params=model.parameters(),lr=0.1)

# What's coming out of our model?
print(model)
# logits (raw outputs of model)
print("Logits:")
## Your code here ##

# Prediction probabilities
print("Pred probs:")
## Your code here ##

# Prediction probabilities
print("Pred labels:")
## Your code here ##

# Let's calculuate the accuracy using accuracy from TorchMetrics
#!pip -q install torchmetrics # Colab doesn't come with torchmetrics
from torchmetrics import Accuracy

## TODO: Uncomment this code to use the Accuracy function
acc_fn = Accuracy(task="multiclass", num_classes=2).to(device) # send accuracy function to device

## TODO: Uncomment this to set the seed
torch.manual_seed(RANDOM_SEED)
torch.cuda.manaul_seed(RANDOM_SEED)

# Setup epochs
epochs=100

# Send data to the device
X_train, y_train = X_train_moon.to(device), y_train_moon.to(device)
X_test, y_test = X_test_moon.to(device), y_test_moon.to(device)


# Loop through the data
for epoch in range(epochs):
  ### Training

  # 1. Forward pass (logits output)
  logits = model(X_train_moon).squeeze()
  
  # Turn logits into prediction probabilities
  y_pred = torch.round(torch.sigmoid(logits))

  # Turn prediction probabilities into prediction labels
  

  # 2. Calculaute the loss
  # loss = loss_fn(y_logits, y_train) # loss = compare model raw outputs to desired model outputs

  # Calculate the accuracy
  # acc = acc_fn(y_pred, y_train.int()) # the accuracy function needs to compare pred labels (not logits) with actual labels

  # 3. Zero the gradients
  

  # 4. Loss backward (perform backpropagation) - https://brilliant.org/wiki/backpropagation/#:~:text=Backpropagation%2C%20short%20for%20%22backward%20propagation,to%20the%20neural%20network's%20weights.
  
  # 5. Step the optimizer (gradient descent) - https://towardsdatascience.com/gradient-descent-algorithm-a-deep-dive-cf04e8115f21#:~:text=Gradient%20descent%20(GD)%20is%20an,e.g.%20in%20a%20linear%20regression) 
  

  ### Testing
  # model_0.eval() 
  # with torch.inference_mode():
    # 1. Forward pass (to get the logits)
    
    # Turn the test logits into prediction labels
    

    # 2. Caculate the test loss/acc
    

  # Print out what's happening every 100 epochs
  # if epoch % 100 == 0:
    