import multiscale_dense as msd
import torch

def test_2d():
    # Test forward evaluation
    img = torch.nn.Parameter(torch.ones((1, 1, 3, 3)))

    model = msd.MSDBlock2d(1, [1, 2], blocksize=2)
    result = model(img)

    assert result.shape == torch.Size([1, 5, 3, 3])

def test_grad_2d():
    # Test forward evaluation
    img = torch.nn.Parameter(torch.ones((1, 1, 3, 3)))

    model = msd.MSDBlock2d(1, [1, 2], blocksize=2)

    # Test gradients
    assert torch.autograd.gradcheck(
            msd.msdblock,
            (img, model.weight, model.bias, model.dilations, model.blocksize),
            eps=1e-4, atol=1e-2, rtol=1e-2, raise_exception=False)

def test_3d():
    # Test forward evaluation
    img = torch.nn.Parameter(torch.ones((1, 1, 3, 3, 3)))

    model = msd.MSDBlock3d(1, [1, 2], blocksize=2)
    result = model(img)

    assert result.shape == torch.Size([1, 5, 3, 3, 3])

def test_grad_3d():
    # Test forward evaluation
    img = torch.nn.Parameter(torch.ones((1, 1, 3, 3, 3)))

    model = msd.MSDBlock3d(1, [1, 2], blocksize=2)

    # Test gradients
    assert torch.autograd.gradcheck(
            msd.msdblock,
            (img, model.weight, model.bias, model.dilations, model.blocksize),
            eps=1e-4, atol=1e-2, rtol=1e-2, raise_exception=False)

if __name__ == '__main__':
    import pytest

    args = [str(__file__.replace('\\', '/')), '-v', '--capture=sys']
    pytest.main(args)