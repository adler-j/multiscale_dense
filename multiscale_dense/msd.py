import torch
import numpy as np
from multiscale_dense.pytorch_workaround import conv2d_weight, conv2d_input

__all__ = ('MSDBlock', 'msdblock')

class MSDBlockImpl(torch.autograd.Function):
    @staticmethod
    def forward(cxt, input, weight, bias, dilations, blocksize):
        bias = None
        cxt.stride = 1
        cxt.paddings = dilations
        cxt.groups = 1
        cxt.dilations = dilations
        cxt.n_conv = len(dilations)
        cxt.blocksize = blocksize

        result = torch.Tensor(input.shape[0], input.shape[1] + blocksize * cxt.n_conv, *input.shape[2:])
        result[:, :input.shape[1]] = input

        w_start = 0
        w_len = blocksize
        w_end = w_start + blocksize
        result_start = input.shape[1]
        bias_start = 0

        for i in range(cxt.n_conv):
            # Extract variables
            sub_weight = weight[w_start:w_start + w_len, :result_start]
            padding = cxt.paddings[i]
            dilation = cxt.dilations[i]

            # Compute convolution
            sub_result = torch.nn.functional.conv2d(
                result[:, :result_start],
                sub_weight,
                bias,
                cxt.stride, padding, dilation, cxt.groups)

            # Apply ReLU
            sub_result[sub_result <= 0] = 0

            # Update result array
            result[:, result_start:result_start+blocksize] = sub_result

            # Update steps etc
            w_start += w_len
            result_start += blocksize
            bias_start += blocksize

        cxt.save_for_backward(input, weight, bias, result)

        return result

    @staticmethod
    def backward(cxt, grad_output):
        input, weight, bias, result = cxt.saved_tensors

        grad_input = grad_weight = grad_bias = None

        n_conv = cxt.n_conv
        blocksize = cxt.blocksize

        w_end = weight.shape[0]
        w_start = w_end - blocksize
        result_end = input.shape[1] + n_conv * blocksize
        result_start = result_end - blocksize

        grad_input = grad_output.clone()
        grad_weight = torch.zeros_like(weight)

        for i in range(n_conv):
            input_shape = [input.shape[0], result_start, *input.shape[2:]]
            padding = cxt.paddings[n_conv - 1 - i]
            dilation = cxt.dilations[n_conv - 1 - i]

            # Get subsets
            sub_grad_output = grad_input[:, -blocksize:]
            sub_weight = weight[w_start:w_end, :result_start]

            # Gradient of ReLU
            sub_grad_output[result[:, result_start:result_end] <= 0] = 0

            # Gradient w.r.t weights
            if cxt.needs_input_grad[1]:
                sub_input = result[:, :result_start]
                sub_weight_shape = sub_weight.shape
                sub_grad_weight = conv2d_weight(sub_input, sub_weight_shape, sub_grad_output,
                                                                         cxt.stride, padding, dilation, cxt.groups)
                grad_weight[w_start:w_end, :result_start] = sub_grad_weight

            # Gradient of Conv
            sub_grad_input = conv2d_input(
                input_shape, sub_weight, sub_grad_output,
                cxt.stride, padding, dilation, cxt.groups)

            # Gradient of concatenation
            grad_input = grad_input[:, :-blocksize] + sub_grad_input

            # Update positions etc
            result_end -= blocksize
            result_start = result_end - blocksize
            w_end -= blocksize
            w_start = w_end - blocksize

        if bias is not None and cxt.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        if bias is not None:
            return grad_input, grad_weight, grad_bias, None, None
        else:
            return grad_input, grad_weight, None, None, None

msdblock = MSDBlockImpl.apply

class MSDBlock(torch.nn.Module):
    def __init__(self, in_channels, dilations, kernel_size=3, blocksize=1, bias=False):
        """Multi-scale dense block

        Parameters
        ----------
        in_channels : int
            Number of input channels
        dilations : tuple of int
            Dilation for each convolution-block
        kernel_size : int or tuple of ints
            Kernel size (only 3 supported ATM)
        blocksize : int
            Number of channels per convolution.

        Notes
        -----
        The number of output channels is in_channels + n_conv * blocksize
        """
        super().__init__()
        self.kernel_size = torch.nn.functional._pair(kernel_size)
        self.blocksize = blocksize
        self.dilations = [torch.nn.functional._pair(dilation)
                          for dilation in dilations]

        n_conv = len(self.dilations)
        max_in = in_channels + blocksize * (n_conv - 1)
        total_out = blocksize * n_conv
        self.weight = torch.nn.Parameter(torch.Tensor(total_out, max_in, *self.kernel_size))

        if bias:
            assert False
            self.bias = torch.nn.Parameter(torch.Tensor(n_conv * blocksize))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return msdblock(input, self.weight, self.bias, self.dilations, self.blocksize)