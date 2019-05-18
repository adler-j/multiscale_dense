import multiscale_dense as msd
import torch

# Test forward evaluation
img = torch.nn.Parameter(torch.randn((1, 1, 3, 3)))

model = msd.MSDBlock(1, [1, 2, 3, 4], blocksize=2)
result = model(img)

# Test gradients
torch.autograd.gradcheck(msd.msdblock,
                         (img, model.weight, model.bias, model.dilations, model.blocksize),
                         eps=1e-3, atol=3e-3, rtol=3e-3)