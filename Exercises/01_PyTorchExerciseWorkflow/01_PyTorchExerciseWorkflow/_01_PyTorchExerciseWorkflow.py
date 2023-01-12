#01_pytorch exercise workflow

from pathlib import Path
import torch
from torch import nn
import matplotlib.pyplot as plt

weight = 0.3
bias = 0.9

start = 0
end = 1
step = 0.02
X= torch.arange(start,end,step).unsqueeze(dim=1)
y = weight * X + bias

train_split = int(0.8 *len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

def plot_predictions(train_data=X_train, 
                     train_labels=y_train, 
                     test_data=X_test,
                     test_labels =y_test,
                     predictions=None):
    plt.figure(figsize= (10,7))
    plt.scatter(train_data, train_labels, c="g", s=4, label = "Training data")
    plt.scatter(test_data, test_labels, c = "b", s=4, label = "Test data")
    if predictions is not None:
        plt.scatter(test_data,predictions, c="r", s=4, label="Predictions")

    plt.legend(prop={"size":14})
    plt.show()


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, 
                                               requires_grad=True,
                                               dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1,
                                             requires_grad=True,
                                             dtype=torch.float))
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.weight * x + self.bias
 
    
model = LinearRegressionModel()

with torch.inference_mode():
    y_preds = model(X_test)

loss = nn.L1Loss()

optimizer = torch.optim.SGD(params = model.parameters(), lr = 0.01)

epoch_count = []
loss_values = []
test_loss_values = []

torch.manual_seed(42)
epochs = 300

for epoch in range(epochs):
    print("amount of attemps: ", epoch)
    
    model.train()
    y_pred = model(X_train)
    lossNum = loss(y_pred, y_train)
    optimizer.zero_grad()
    lossNum.backward()
    optimizer.step()
    model.eval()
    with torch.inference_mode():
        test_pred = model(X_test)
        test_loss = loss(test_pred, y_test)

    if(epoch % 20 == 0):
        epoch_count.append(torch.tensor(epoch).numpy())
        loss_values.append(torch.tensor(lossNum).numpy())
        test_loss_values.append(torch.tensor(test_loss).numpy())
        print(f"Epoch: {epoch} | Loss: {loss} | Test loss: {test_loss}")

with torch.inference_mode():
    y_preds_new = model(X_test)

print(plot_predictions(predictions=y_preds_new))

def results():
    plt.plot(epoch_count, loss_values, label = "Train loss")
    plt.plot(epoch_count, test_loss_values, label = "Test loss")
    plt.title("Training and test loss curves")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.show()
    
results()
# 1. Create models directory 
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents = True,exist_ok = True)
# 2. Create model save path 
MODEL_NAME = "pytorch_model"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME 
# 3. Save the model state dict
print(f"Saving model to {MODEL_SAVE_PATH}")
torch.save(obj = model.state_dict(),f = MODEL_SAVE_PATH)


device = "cuda"
# Create new instance of model and load saved state dict (make sure to put it on the target device)
loaded_model = LinearRegressionModel()
loaded_model.load_state_dict(torch.load(f = MODEL_SAVE_PATH))
loaded_model.to(device)

# Make predictions with loaded model and compare them to the previous
y_preds_new = loaded_model(X_test)
y_preds == y_preds_new

loaded_model.state_dict()