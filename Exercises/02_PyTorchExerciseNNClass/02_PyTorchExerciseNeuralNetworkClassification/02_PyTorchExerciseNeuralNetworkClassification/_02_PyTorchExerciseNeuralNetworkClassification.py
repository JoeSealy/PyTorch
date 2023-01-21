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

X_moon, y_moon = make_moons(n_samples = 1000,random_state=RANDOM_SEED,noise=0.07)
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
X_moon, y_moon = X_moon.to(device), y_moon.to(device)

# Split the data into train and test sets (80% train, 20% test)
from sklearn.model_selection import train_test_split

X_train_moon, X_test_moon, y_train_moon, y_test_moon = train_test_split(X_moon, y_moon, test_size=0.2, random_state = RANDOM_SEED)

import torch
from torch import nn

# Inherit from nn.Module to make a model capable of fitting the mooon data
class MoonModelV0(nn.Module):
    def __init__(self, input_features, output_features, hidden_units):
        super().__init__()
        self.Linear_Layer_Stack = nn.Sequential(
            nn.Linear(in_features= input_features, out_features = hidden_units),
            nn.ReLU(),
            nn.Linear(in_features= hidden_units, out_features= hidden_units),
            nn.ReLU(),
            nn.Linear(in_features= hidden_units, out_features = output_features)
        )
    def forward(self, x):
        return self.Linear_Layer_Stack(x)

# Instantiate the model
model = MoonModelV0(input_features = 2, output_features = 1, hidden_units=10).to(device)

# Setup loss function
loss_fn = nn.BCEWithLogitsLoss()

# Setup optimizer to optimize model's parameters
optimizer = torch.optim.SGD(params=model.parameters(),lr=0.1)

# What's coming out of our model?
print(model)
print("Logits:")
print(model(X_moon.to(device)[:10]).squeeze())

# Prediction probabilities
print("Pred probs:")
print(torch.sigmoid(model(X_moon.to(device)[:10]).squeeze()))

# Prediction probabilities
print("Pred labels:")
print(torch.round(torch.sigmoid(model(X_moon.to(device)[:10]).squeeze())))

# Let's calculuate the accuracy using accuracy from TorchMetrics
#!pip -q install torchmetrics # Colab doesn't come with torchmetrics
from torchmetrics import Accuracy

## TODO: Uncomment this code to use the Accuracy function
#acc_fn = Accuracy(task = "binary", num_classes=2).to(device) # send accuracy function to device

## TODO: Uncomment this to set the seed
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)

# Setup epochs
epochs=1000

# Send data to the device
X_train_moon, y_train_moon = X_train_moon.to(device), y_train_moon.to(device)
X_test_moon, y_test_moon = X_test_moon.to(device), y_test_moon.to(device)
print(X_train_moon.shape, y_train_moon.shape)

y_train_moon = torch.squeeze(y_train_moon)

# Loop through the data
for epoch in range(epochs):
  ### Training
  model.train()
  # 1. Forward pass (logits output)
  y_logits = model(X_train_moon).squeeze()
  
  # Turn logits into prediction probabilities
  y_pred = torch.round(torch.sigmoid(y_logits))

  # Turn prediction probabilities into prediction labels

  # 2. Calculaute the loss
  loss = loss_fn(y_logits, y_train_moon) # loss = compare model raw outputs to desired model outputs

  # Calculate the accuracy
  #acc = acc_fn(y_pred, y_train_moon.int())  # the accuracy function needs to compare pred labels (not logits) with actual labels

  # 3. Zero the gradients
  optimizer.zero_grad()

  # 4. Loss backward (perform backpropagation) - https://brilliant.org/wiki/backpropagation/#:~:text=Backpropagation%2C%20short%20for%20%22backward%20propagation,to%20the%20neural%20network's%20weights.
  loss.backward()
  # 5. Step the optimizer (gradient descent) - https://towardsdatascience.com/gradient-descent-algorithm-a-deep-dive-cf04e8115f21#:~:text=Gradient%20descent%20(GD)%20is%20an,e.g.%20in%20a%20linear%20regression) 
  optimizer.step()

  ### Testing
  model.eval() 
  with torch.inference_mode():
    # 1. Forward pass (to get the logits)
    test_logits = model(X_test_moon).squeeze()
    # Turn the test logits into prediction labels
    test_pred = torch.round(torch.sigmoid(test_logits))

    # 2. Caculate the test loss/acc
    test_loss = loss_fn(test_logits, y_test_moon)

    #test_acc = acc_fn(y_true = y_test_moon, y_pred = test_pred) 
    

  # Print out what's happening every 100 epochs
  if epoch % 100 == 0:
      print(f"Epoch: {epoch} | Loss: {loss} | Test Loss: {test_loss}")

# Import torch

# Plot the model predictions

import numpy as np

# TK - this could go in the helper_functions.py and be explained there
def plot_decision_boundary(model, X, y):
  
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Source - https://madewithml.com/courses/foundations/neural-networks/ 
    # (with modifications)
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), 
                         np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1) # mutli-class
    else: 
        y_pred = torch.round(torch.sigmoid(y_logits)) # binary
    
    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


    # Plot decision boundaries for training and test sets
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model, X_train_moon, y_train_moon)
plt.show()
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model, X_test_moon, y_test_moon)
plt.show()
     
     