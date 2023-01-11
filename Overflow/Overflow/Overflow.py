#PyTorch workflow

from importlib.metadata import requires
from pickle import TRUE
import torch
from torch import nn
import matplotlib.pyplot as plt

print(torch.__version__)
"""
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
    
    plots training data, test data and compares predictions.
    
    plt.figure(figsize = (10,7))

    #Plot training data in blue

    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    plt.scatter(test_data, test_labels, c="g", s=4, label="Test data")

    #are there predictions
    if predictions is not None:
        plt.scatter(test_data,predictions, c="r", s=4, label="Predictions")

    plt.legend(prop={"size":14})

    plt.show()

print(plot_predictions())
"""

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
