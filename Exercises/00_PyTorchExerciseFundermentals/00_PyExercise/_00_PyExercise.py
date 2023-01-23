import torch
##########################################################1
#document reading

###########################################################2 correct
# Create random tensor
random_tensor = torch.rand(7,7)
print(random_tensor)

###########################################################3 correct
# Create another random tensor
random_tensor2 = torch.rand(1,7)
print(random_tensor2)

# Perform matrix multiplication 
mult_tensor = torch.matmul(random_tensor,random_tensor2.T)
print(mult_tensor)

###########################################################4 correct
# Set manual seed
seed = 0

# Create two random tensors
torch.manual_seed(seed)
random_tensor3 = torch.rand(7,7)
print(random_tensor3)
torch.manual_seed(seed)
random_tensor4 = torch.rand(1,7)
print(random_tensor4)

# Matrix multiply tensors
mult_tensor2 = torch.matmul(random_tensor3,random_tensor4.T)
print(mult_tensor2)

#############################################################5 correct lol
# Set random seed on the GPU
print(torch.cuda.manual_seed(1234))

#############################################################6 correct
# Set random seed
torch.manual_seed(1234)

# Check for access to GPU
print(torch.cuda.is_available())

# Create two random tensors on GPU
device = "cuda"
random_tensor5 = torch.rand(2,3).to(device)
print(random_tensor5)
random_tensor6 = torch.rand(2,3).to(device)
print(random_tensor6)

###############################################################7 correct
# Perform matmul on tensor_5 and tensor_6
mult_tensor3 = torch.matmul(random_tensor5,random_tensor6.T)
print(mult_tensor3, mult_tensor3.shape)

################################################################8 correct
# Find max
print(torch.max(mult_tensor3))
# Find min
print(torch.min(mult_tensor3))

#################################################################9 correct
# Find arg max
print(torch.argmax(mult_tensor3))
# Find arg min
print(torch.argmin(mult_tensor3))

#################################################################10 correct
# Set seed
last_seed = 7

#Create random tensor
torch.manual_seed(last_seed)
random_tensor7 = torch.rand(1,1,1,10)

# Remove single dimensions
squeezed_tensor = torch.squeeze(random_tensor7)

# Print out tensors and their shapes
print(random_tensor7, random_tensor7.shape)
print(squeezed_tensor, squeezed_tensor.shape)
##################################################################end