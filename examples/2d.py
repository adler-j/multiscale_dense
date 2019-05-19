import multiscale_dense as msd
import torch

# Test forward evaluation
img = torch.nn.Parameter(torch.ones((1, 1, 5, 5)))

model = msd.MSDBlock2d(1, [1, 1, 1], blocksize=1)
result = model(img)

print(result)

# Backward evaluation
loss = torch.mean(result)
loss.backward()

for weight in model.weights:
    print(weight.grad)
