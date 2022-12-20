import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print(torch.__version__)
'''
TENSOR = torch.tensor([[[1,2,3],
                        [3,4,5],
                        [5,6,7]]])

#print(TENSOR.shape)

#RANDOM_TENSOR = torch.rand(3,4)

print(RANDOM_TENSOR.ndim)

RANDOM_IMAGE_SIZE_TENSOR = torch.rand(size = (224,224,3))
print(RANDOM_IMAGE_SIZE_TENSOR.shape, RANDOM_IMAGE_SIZE_TENSOR.ndim)

ZEROS = torch.zeros(size=(0, 3)) 

print(ZEROS * RANDOM_TENSOR)


TENSOR_32 = torch.tensor([3.345678,6.456567,9.45678], dtype=torch.float16, device= "cpu", requires_grad= False)
TENSOR_RAND = torch.rand(3, 4)

print(TENSOR_32)

print(f"Data type: {TENSOR_RAND.dtype} Size of data: {TENSOR_RAND.shape} device type: {TENSOR_RAND.device}  ")



#####################################################
#Element wise multiplication

TENSOR = torch.tensor([1,2,3])

print(TENSOR*TENSOR)

# Matrix mulitplication
print(torch.matmul(TENSOR, TENSOR))

#INTENTIONAL SHAPE ERROR (3,2)(3,2)
TENSOR_A = torch.tensor([[1,2],
                          [3,4],
                          [5,6]])

TENSOR_B = torch.tensor([[7,8],
                          [9,10],
                          [11,12]])
#TRANSPOSED
print(torch.mm(TENSOR_A,TENSOR_B.T))


#####################################################
#TENSOR AGGRIGATION

X = torch.arange(0,100,10)

print(X)
print(torch.min(X), X.min())
print(torch.max(X), X.max())
print(torch.mean(X.type(torch.float32)), X.type(torch.float32).mean())
print(torch.sum(X), X.sum())

#POSITIONAL MIN AND MAX
print(torch.argmin(X), torch.argmax(X))

###############################################################

#RESHAPIONG, STACKING SQUEEZING AND UNSQUEEZING TENSORS

a = torch.rand(2,4,4)
a_reshape = a.reshape(1,8,4)
print(a_reshape)

#change the view
print(a.view(8,1,4))

#stacking on top = dim=0 | side by side = dim=1
a_stacked = torch.stack([a,a,a,a], dim=2)
print(a_stacked)

#SQUEEZE
a_squeezed = a_reshape.squeeze(dim=0)
print(f"squeezed: ", a_squeezed.shape)

#UNSQUEEZE
a_unsqueeze = a_squeezed.unsqueeze(dim=1)
print(f"unsquezed: ", a_unsqueeze.shape)

#PERMUTE
print(f"reshaped: ",a_reshape.shape)
a_permute = torch.permute(a_reshape, (2,0,1))
print(f"permuted: ",a_permute.shape)

#COLONS
x = torch.arange(1,10).reshape(1,3,3)
print(x, x.shape)
print(x[0][0])
print(x[0][1][1])
print(x[:, 0])
print(x[:, :, 1])
print(x[:, 1, 1])
print(x[:, 2, 2])
print(x[:, :, 2])

#############################################
#Numpy array's and torch tensors

array = np.arange(1.0,8.0)
tensor = torch.from_numpy(array)
print(array, tensor)

#change the value of array ,obviously tensor doesnt change
array = array + 1
print(array, tensor)

tensor_one = torch.ones(7)
numpy_tensor = tensor.numpy()
print(tensor_one, numpy_tensor)
###############################################
'''
#reproducability
RTensor_A = torch.rand(3,4)
RTensor_B = torch.rand(3,4)
print(RTensor_A == RTensor_B)

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
RTensor_C = torch.rand(3,4)
torch.manual_seed(RANDOM_SEED)
RTensor_D = torch.rand(3,4)

print(RTensor_C, RTensor_D)
print(RTensor_C == RTensor_D)