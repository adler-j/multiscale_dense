import multiscale_dense as msd
import torch
import time
import numpy as np

N = 1

# Set up problem
img = torch.nn.Parameter(torch.ones((1, 1, 512, 512)))
model = msd.MSDBlock2d(1, [i + 1 for i in range(10)], blocksize=4)

# Apply dense block
t0 = time.time()
for i in range(N):
    result1 = model(img)
t1 = time.time()
print('Dense Block timing: ', (t1 - t0) / N)

# Compare to convolution
conv = torch.nn.Sequential(
        torch.nn.Conv2d(1, 32, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(32, 32, 3, padding=1))

# Apply convolution
t0 = time.time()
for i in range(N):
    result2 = conv(img)
t1 = time.time()
print('Convolution timing: ', (t1 - t0) / N)

# Timing for backward call of dense block
t0 = time.time()
for i in range(N):
    result1 = model(img)
    loss = torch.mean(result1)
    loss.backward()
t1 = time.time()
print('Dense gradient:', (t1 - t0) / N)

# Timing for backward call of convolution
t0 = time.time()
for i in range(N):
    result2 = conv(img)
    loss = torch.mean(result2)
    loss.backward()
t1 = time.time()
print('Convolution gradient:', (t1 - t0) / N)

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

print(get_n_params(model))
print(get_n_params(conv))