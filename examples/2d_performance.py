import multiscale_dense as msd
import torch
import time
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

N = 1

# Set up problem
img = torch.nn.Parameter(torch.ones((1, 1, 512, 512))).to(device)
model = msd.MSDBlock2d(1, [i + 1 for i in range(100)], blocksize=1).to(device)

# Warmup
result1 = model(img)
loss = torch.mean(result1)
loss.backward()

# Compare to convolution
conv = torch.nn.Sequential(
        torch.nn.Conv2d(1, 32, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(32, 32, 3, padding=1)).to(device)

# Warmup
result2 = model(img)
loss = torch.mean(result2)
loss.backward()

# Apply dense block
start.record()
for i in range(N):
    result1 = model(img)
end.record()
torch.cuda.synchronize()
print(start.elapsed_time(end))


# Apply convolution
start.record()
for i in range(N):
    result2 = conv(img)
end.record()
torch.cuda.synchronize()
print('Convolution timing: ', start.elapsed_time(end) / N)

# Timing for backward call of dense block
start.record()
for i in range(N):
    result1 = model(img)
    loss = torch.mean(result1)
    loss.backward()
end.record()
torch.cuda.synchronize()
print('Dense gradient: ', start.elapsed_time(end) / N)

# Timing for backward call of convolution
start.record()
for i in range(N):
    result2 = conv(img)
    loss = torch.mean(result2)
    loss.backward()
end.record()
torch.cuda.synchronize()
print('Convolution gradient: ', start.elapsed_time(end) / N)

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