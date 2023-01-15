#PyTorch workflow

import torch
from torch import nn
import matplotlib.pyplot as plt
from pathlib import Path 

print(torch.__version__)

##DATA###############################
weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02
X= torch.arange(start,end,step).unsqueeze(dim=1)
y = weight * X + bias
print(X[:10], y[:10] )

#create a train/test split

train_split = int(0.8 *len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

#how might we visualise our data?
#Visualise, visualise, visualise

def plot_predictions(train_data=X_train, 
                     train_labels=y_train, 
                     test_data=X_test,
                     test_labels =y_test,
                     predictions=None):
    
    #plots training data, test data and compares predictions.
    
    plt.figure(figsize = (10,7))

    #Plot training data in blue

    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    plt.scatter(test_data, test_labels, c="g", s=4, label="Test data")

    #are there predictions
    if predictions is not None:
        plt.scatter(test_data,predictions, c="r", s=4, label="Predictions")

    plt.legend(prop={"size":14})

    plt.show()

#print(plot_predictions())


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, 
                                                requires_grad=True,
                                                dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1, 
                                                requires_grad=True,
                                                dtype=torch.float))
        # foward method
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias


torch.randn(1)

model_0 = LinearRegressionModel()

print(list(model_0.parameters()))
print(list(model_0.state_dict()))

with torch.inference_mode():
    y_preds = model_0(X_test)

print(y_preds)

print(y_test)

print(plot_predictions(predictions=y_preds))
#loss function
loss_fn = nn.L1Loss()
#optimiser
optimizer = torch.optim.SGD(params=model_0.parameters(),lr =0.01)

#Tracking
epoch_count = []
loss_values = []
test_loss_values = []



torch.manual_seed(42)
#an epoch is 1 loop through the data
epochs = 1000

#0.Loop through the data
for epoch in range(epochs):

    print("amount of attemps: ", epoch)
    #set the model to raining mode
    model_0.train() #train mode in pytorch sets all parameters that require gradients to require gradients

    #Forward pass
    y_pred = model_0(X_train)

    #Claculate loss
    loss = loss_fn(y_pred, y_train)
    print("Loss: ",loss)

    #Optermiser zero grad
    optimizer.zero_grad()

    #Perform backpropagation on the loss with repect to the parameters of the model
    loss.backward()

    #step the optimiser
    optimizer.step()

    #testing
    model_0.eval() #turns off gradient tracking
    with torch.inference_mode():
        test_pred = model_0(X_test)

        test_loss = loss_fn(test_pred, y_test)
    
    if epoch % 10 == 0:
        epoch_count.append(torch.tensor(epoch).numpy())
        loss_values.append(torch.tensor(loss).numpy())
        test_loss_values.append(torch.tensor(test_loss).numpy())
        print(f"Epoch: {epoch} | Loss: {loss} | Test loss: {test_loss}")
        
with torch.inference_mode():
    y_preds_new = model_0(X_test)


print(plot_predictions(predictions=y_preds_new))

plt.plot(epoch_count, loss_values, label = "Train loss")
plt.plot(epoch_count, test_loss_values, label = "Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.show()


#SAVE
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents = True, exist_ok = True)
MODEL_NAME = "01_pytorchWorkflow.pt"
MODEL_SAVE_PATH = MODEL_PATH/MODEL_NAME
print(f"Saving model to:{MODEL_SAVE_PATH}")
torch.save(obj=model_0.state_dict(), f=MODEL_SAVE_PATH)

#LOADING
print(model_0.state_dict())
loaded_model_0 = LinearRegressionModel()
loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
print(loaded_model_0.state_dict())





#better alternative for the foward pass

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1,
                                        out_features=1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)

