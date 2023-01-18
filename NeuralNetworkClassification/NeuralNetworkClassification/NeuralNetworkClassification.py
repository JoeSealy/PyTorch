#2 Neural Network Classification

import sklearn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn

n_samples = 1000

X, y = make_circles(n_samples,
                    noise=0.03,
                    random_state=42)

print(len(X), len(y))
print(f"First 5 samples of X:\n{X[:5]}")
print(f"First 5 samples of y:\n{y[:5]}")
circles = pd.DataFrame({"X1": X[:, 0],
                        "X2": X[:, 1],
                        "label": y})
print(circles.head(10))

#visualise
plt.scatter(x=X[:, 0],
            y=X[:, 1],
            c=y,
            cmap=plt.cm.RdYlBu);
#plt.show()

#check input and output shapes
print(X.shape, y.shape) 
print(X)

X_sample = X[0]
y_sample = y[0]

print(f"Valus for one sample of X: {X_sample} and the same for y: {y_sample}")
print(f"Valus for one sample of X: {X_sample.shape} and the same for y: {y_sample.shape}")

#tensors
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

print(X[:5], y[:5])
print(type(X), X.dtype, y.dtype)

#split
torch.manual_seed(42)
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)
print(len(X_train),len(X_test),len(y_train),len(y_test))
print(n_samples)

#building a model

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
print(X_test)
print(X_train.shape)
class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features = 2, out_features = 5) # 2-5
        self.layer_2 = nn.Linear(in_features = 5, out_features = 1) # 5-1
    def forward(self, x):
        return self.layer_2(self.layer_1(x)) #x -> L1 -> L2 -> output

#instance

model_0 = CircleModelV0().to(device)
print(model_0)
next(model_0.parameters()).device

model_0 = nn.Sequential(
    nn.Linear(in_features= 2, out_features = 5),
    nn.Linear(in_features= 5, out_features = 1)).to(device)

print(model_0)
print(model_0.state_dict())
with torch.inference_mode():
    untrained_preds = model_0(X_test.to(device))

print("Length of predictions: {len(untrained_preds)}, Shape: {Untrained_preds.shape} ")
print("Length of predictions: {len(X_test)}, Shape: {X_test.shape} ")
print(f"\nFirst 10 predictions:\n {untrained_preds[:10]}")
print(f"\nFirst 10 labels:\n {y_test[:10]}")

#loss function
loss_fn= nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)

#accuracy

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred)) * 100
    return acc

#train model
model_0.eval()
with torch.inference_mode():
    y_logits = model_0(X_test.to(device))[:5]
print(y_logits)

y_pred_probs = torch.sigmoid(y_logits)
print(y_pred_probs)

y_preds = torch.round(y_pred_probs)

y_preds_label = torch.round(torch.sigmoid(model_0(X_test.to(device))[:5]))

print(torch.eq(y_preds.squeeze(), y_preds_label.squeeze()))

print(y_preds.squeeze())


#training and testing loop
torch.cuda.manual_seed(42)
torch.manual_seed(42)
epochs = 1

X_train, y_train = X_train.to(device), y_train.to(device)


for epoch in range(epochs):

    







