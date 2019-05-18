import multiscale_dense as msd
import torch
from torch.autograd.gradcheck import get_numerical_jacobian, get_analytical_jacobian
import matplotlib.pyplot as plt
import numpy as np

# Test forward evaluation
img = torch.nn.Parameter(torch.ones((1, 1, 5, 5)))

model = msd.MSDBlock2d(1, [1, 2], blocksize=2)

inp = (img, model.weight, model.bias, model.dilations, model.blocksize)

jac_num = get_numerical_jacobian(
        lambda x: msd.msdblock(*x), inp, eps=1e-4)
jac_an = get_analytical_jacobian(
        inp, msd.msdblock(*inp))

print('input error:',
      torch.max(torch.abs(jac_num[0] - jac_an[0][0])) / torch.max(torch.abs(jac_num[0])))

print('weight error:',
      torch.max(torch.abs(jac_num[1] - jac_an[0][1])) / torch.max(torch.abs(jac_num[1])))