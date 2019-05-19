import multiscale_dense as msd
import torch
import time
import numpy as np

N = 1

# Set up problem
img = torch.nn.Parameter(torch.ones((1, 1, 512, 512)))
model = msd.MSDBlock2d(1, [1]*10, blocksize=4)

# Use the PyTorch profiler to get more information
#with torch.autograd.profiler.profile() as prof:
 #   result1 = model(img)
#print('=== DENSE ===')
#print(prof)

with torch.autograd.profiler.profile() as prof:
    result1 = model(img)
    loss = torch.mean(result1)
    loss.backward()
print('=== DENSE GRAD ===')
print(prof)

print(str(prof).count('thnn_conv2d_forward'),
      str(prof).count('thnn_conv_transpose2d_forward'))