import multiscale_dense as msd

img = torch.nn.Parameter(torch.randn((1, 1, 3, 3)))

model = MSDBlock(1, [1, 2], blocksize=1)
result = model(img)