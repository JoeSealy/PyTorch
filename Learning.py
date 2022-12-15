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

'''
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



