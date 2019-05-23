import multiscale_dense as msd
import torch
import time
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N = 1

# Set up problem
img = torch.nn.Parameter(torch.ones((1, 1, 512, 512))).to(device)
model = msd.MSDBlock2d(1, [i + 1 for i in range(100)], blocksize=1).to(device)

# warmup
result1 = model(img)
loss = torch.mean(result1)
loss.backward()

# Use the PyTorch profiler to get more information
#with torch.autograd.profiler.profile() as prof:
 #   result1 = model(img)
#print('=== DENSE ===')
#print(prof)

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    result1 = model(img)
    loss = torch.mean(result1)
    loss.backward()
print('=== DENSE GRAD ===')
print(prof)

print(str(prof).count('thnn_conv2d_forward'),
      str(prof).count('thnn_conv_transpose2d_forward'))