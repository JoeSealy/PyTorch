
import torch 

print(torch.cuda.is_available())

device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)

tensor = torch.tensor([1,2,3])

print(tensor, tensor.device)

tensor_on_gpu = tensor.to(device)
print(tensor_on_gpu)

print(tensor_on_gpu.cpu().numpy())


