import torch
##########################################################1
#document reading

###########################################################2
# Create random tensor
random_tensor = torch.rand(7,7)
print(random_tensor)

###########################################################3
# Create another random tensor
random_tensor2 = torch.rand(1,7)
print(random_tensor2)

# Perform matrix multiplication 
mult_tensor = torch.matmul(random_tensor,random_tensor2.T)
print(mult_tensor)

###########################################################4
# Set manual seed
seed = 0

# Create two random tensors
torch.manual_seed(seed)
random_tensor3 = torch.rand(2,3)
print(random_tensor3)
torch.manual_seed(seed)
random_tensor4 = torch.rand(2,3)
print(random_tensor4)

# Matrix multiply tensors
mult_tensor2 = torch.matmul(random_tensor3,random_tensor4.T)
print(mult_tensor2)

#############################################################5
# Set random seed on the GPU
print(torch.seed())

#############################################################6
# Set random seed
GPUseed = torch.seed()

# Check for access to GPU
print(torch.cuda.is_available())

# Create two random tensors on GPU
device = "cuda"
random_tensor5 = torch.rand(2,3).to(device)
print(random_tensor5)
torch.manual_seed(GPUseed)
random_tensor6 = torch.rand(2,3).to(device)
print(random_tensor6)

###############################################################7
# Perform matmul on tensor_5 and tensor_6
mult_tensor3 = torch.matmul(random_tensor5,random_tensor6.T)
print(mult_tensor3, mult_tensor3.shape)

################################################################8
# Find max
print(torch.max(mult_tensor3))
# Find min
print(torch.min(mult_tensor3))

#################################################################9
# Find arg max
print(torch.argmax(mult_tensor3))
# Find arg min
print(torch.argmin(mult_tensor3))

#################################################################10
# Set seed
last_seed = 7

# Create random tensor
torch.manual_seed(last_seed)
random_tensor7 = torch.rand(1,1,1,10)

# Remove single dimensions
squeezed_tensor = torch.squeeze(random_tensor7)

# Print out tensors and their shapes
print(random_tensor7, random_tensor7.shape)
print(squeezed_tensor, squeezed_tensor.shape)
##################################################################end