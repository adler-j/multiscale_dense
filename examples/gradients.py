import multiscale_dense as msd
import torch
from torch.autograd.gradcheck import get_numerical_jacobian, get_analytical_jacobian
import matplotlib.pyplot as plt
import numpy as np

# Test forward evaluation
img = torch.nn.Parameter(torch.ones((1, 1, 5, 5)))

dilations = [1, 1, 1, 1, 1]
model = msd.MSDBlock2d(1, dilations=dilations, blocksize=2)

inp = (img, model.bias, model.dilations, model.blocksize, *model.weights)

jac_num = get_numerical_jacobian(
        lambda x: msd.msdblock(*x), inp, eps=1e-4)
jac_an = get_analytical_jacobian(
        inp, msd.msdblock(*inp))

print('input error:',
      torch.max(torch.abs(jac_num[0] - jac_an[0][0])) / torch.max(torch.abs(jac_num[0])))

for i in range(len(dilations)):
    print('weight{} error:'.format(i),
          torch.max(torch.abs(jac_num[i + 1] - jac_an[0][i + 1])) / torch.max(torch.abs(jac_num[1])))
