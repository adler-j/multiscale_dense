# Multiscale Dense
Efficient implementation of multiscale dense networks.

# Example
Using the block is straightforward. Simply specify what dilations to use and the blocksize to use

    >>> import multiscale_dense as msd
    >>> img = torch.nn.Parameter(torch.randn((1, 1, 3, 3)))
    >>> model = msd.MSDBlock2d(1, dilations=[1, 2], blocksize=2)
    >>> result = model(img)
    >>> result.shape
    torch.Size([1, 5, 3, 3])
