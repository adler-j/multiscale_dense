import multiscale_dense as msd
import torch
import time
import numpy as np

N = 10

# Set up problem
img = torch.nn.Parameter(torch.ones((1, 1, 512, 512)))
model = msd.MSDBlock2d(1, [1], blocksize=64)

# Apply n times
t0 = time.time()
for i in range(N):
    result1 = model(img)
t1 = time.time()
print(t1 - t0)

# Compare to convolution
conv = torch.nn.Conv2d(1, 64, 3, padding=1)

# Apply n times
t0 = time.time()
for i in range(N):
    result2 = conv(img)
t1 = time.time()
print(t1 - t0)

t0 = time.time()
for i in range(N):
    result1 = model(img)
    loss = torch.mean(result1)
    loss.backward()
t1 = time.time()
print(t1 - t0)

# Apply n times
t0 = time.time()
for i in range(N):
    result2 = conv(img)
    loss = torch.mean(result2)
    loss.backward()
t1 = time.time()
print(t1 - t0)

with torch.autograd.profiler.profile() as prof:
    result1 = model(img)
print(prof)

with torch.autograd.profiler.profile() as prof:
    result2 = conv(img)
print(prof)

with torch.autograd.profiler.profile() as prof:
    result1 = model(img)
    loss = torch.mean(result1)
    loss.backward()
print(prof)

with torch.autograd.profiler.profile() as prof:
    result2 = conv(img)
    loss = torch.mean(result2)
    loss.backward()
print(prof)
