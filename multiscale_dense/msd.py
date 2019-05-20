import torch
import numpy as np
from multiscale_dense.pytorch_workaround import (
        conv1d_weight, conv1d_input,
        conv2d_weight, conv2d_input,
        conv3d_weight, conv3d_input)

__all__ = ('msdblock', 'MSDBlock1d', 'MSDBlock2d', 'MSDBlock3d')

IDX_WEIGHT_START = 4

class MSDBlockImpl(torch.autograd.Function):

    @staticmethod
    def forward(cxt, input, bias, dilations, blocksize, *weights):
        cxt.stride = 1
        cxt.paddings = dilations
        cxt.groups = 1
        cxt.dilations = dilations
        cxt.n_conv = len(dilations)
        cxt.blocksize = blocksize
        cxt.ndim = input.dim() - 2

        if cxt.ndim == 1:
            conv = torch.nn.functional.conv1d
        elif cxt.ndim == 2:
            conv = torch.nn.functional.conv2d
        elif cxt.ndim == 3:
            conv = torch.nn.functional.conv3d
        else:
            raise ValueError('Only supports 1 2 or 3 dimensions')

        result = torch.empty(input.shape[0],
                             input.shape[1] + blocksize * cxt.n_conv,
                             *input.shape[2:],
                             device=input.device)
        result[:, :input.shape[1]] = input

        result_start = input.shape[1]
        bias_start = 0

        for i in range(cxt.n_conv):
            # Extract variables
            sub_weight = weights[i]
            sub_bias = bias[i * blocksize:(i+1) * blocksize] if bias is not None else None
            padding = cxt.paddings[i]
            dilation = cxt.dilations[i]
            
            # Compute convolution
            sub_result = conv(
                result[:, :result_start],
                sub_weight,
                sub_bias,
                cxt.stride, padding, dilation, cxt.groups)
            
            # Apply ReLU
            torch.relu_(sub_result)

            # Update result array
            result[:, result_start:result_start+blocksize] = sub_result

            # Update steps etc
            result_start += blocksize
            bias_start += blocksize

        cxt.save_for_backward(bias, result, *weights)

        return result

    @staticmethod
    def backward(cxt, grad_output):
        bias, result, *weights = cxt.saved_tensors

        if cxt.ndim == 1:
            conv_weight = conv1d_weight
            conv_input = conv1d_input
        elif cxt.ndim == 2:
            conv_weight = conv2d_weight
            conv_input = conv2d_input
        elif cxt.ndim == 3:
            conv_weight = conv3d_weight
            conv_input = conv3d_input
        else:
            raise ValueError('Only supports 1 2 or 3 dimensions')

        n_conv = cxt.n_conv
        blocksize = cxt.blocksize
        
        # Input is part of the result, so we can extract it
        input = result[:, :weights[0].shape[1]]

        grad_bias = torch.empty_like(bias) if bias is not None else None
        grad_input = grad_output.clone()
        grad_weights = []
        
        result_end = input.shape[1] + n_conv * blocksize
        result_start = result_end - blocksize

        for i in range(n_conv):
            idx = n_conv - 1 - i
            input_shape = [input.shape[0], result_start, *input.shape[2:]]
            padding = cxt.paddings[idx]
            dilation = cxt.dilations[idx]

            # Get subsets
            sub_grad_output = grad_input[:, -blocksize:]
            sub_weight = weights[idx]

            # Gradient of ReLU
            sub_grad_output *= (result[:, result_start:result_end] > 0).type(sub_grad_output.dtype)

            # Gradient w.r.t weights
            if True and cxt.needs_input_grad[i + IDX_WEIGHT_START]:
                sub_input = result[:, :result_start]
                sub_weight_shape = sub_weight.shape

                sub_grad_weight = conv_weight(
                    sub_input, sub_weight_shape, sub_grad_output,
                    cxt.stride, padding, dilation, cxt.groups)
                grad_weights.insert(0, sub_grad_weight)
            else:
                grad_weights.insert(0, None)

            # Gradient of Conv
            sub_grad_input = conv_input(
                input_shape, sub_weight, sub_grad_output,
                cxt.stride, padding, dilation, cxt.groups)

            # Gradient of Bias
            if bias is not None and cxt.needs_input_grad[1]:
                sum_idx = [0] + list(range(2,2+cxt.ndim))
                grad_bias[idx * blocksize:(idx + 1) * blocksize] = sub_grad_output.sum(sum_idx)
            
            # Gradient of concatenation
            grad_input = grad_input[:, :-blocksize] + sub_grad_input

            # Update positions etc
            result_end -= blocksize
            result_start = result_end - blocksize

        return (grad_input, grad_bias, None, None, *grad_weights)

msdblock = MSDBlockImpl.apply


class MSDBlock1d(torch.nn.Module):
    def __init__(self, in_channels, dilations, kernel_size=3, blocksize=1, bias=True):
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
        self.kernel_size = torch.nn.functional._single(kernel_size)
        self.blocksize = blocksize
        self.dilations = [torch.nn.functional._single(dilation)
                          for dilation in dilations]

        n_conv = len(self.dilations)

        self.weights = []
        for i in range(n_conv):
            n_in = in_channels + blocksize * i

            weight = torch.nn.Parameter(torch.Tensor(
                blocksize, n_in, *self.kernel_size))

            self.register_parameter('weight{}'.format(i), weight)
            self.weights.append(weight)

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(n_conv * blocksize))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.weights:
            torch.nn.init.kaiming_uniform_(weight, a=np.sqrt(5))

        if self.bias is not None:
            # TODO: improve
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weights[0])
            bound = 1 / np.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return msdblock(input, self.bias, self.dilations, self.blocksize,
                        *self.weights)
    
    
class MSDBlock2d(torch.nn.Module):
    def __init__(self, in_channels, dilations, kernel_size=3, blocksize=1, bias=True):
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

        self.weights = []
        for i in range(n_conv):
            n_in = in_channels + blocksize * i

            weight = torch.nn.Parameter(torch.Tensor(
                blocksize, n_in, *self.kernel_size))

            self.register_parameter('weight{}'.format(i), weight)
            self.weights.append(weight)

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(n_conv * blocksize))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.weights:
            torch.nn.init.kaiming_uniform_(weight, a=np.sqrt(5))

        if self.bias is not None:
            # TODO: improve
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weights[0])
            bound = 1 / np.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return msdblock(input, self.bias, self.dilations, self.blocksize,
                        *self.weights)


class MSDBlock3d(torch.nn.Module):
    def __init__(self, in_channels, dilations, kernel_size=3, blocksize=1, bias=True):
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
        self.kernel_size = torch.nn.functional._triple(kernel_size)
        self.blocksize = blocksize
        self.dilations = [torch.nn.functional._triple(dilation)
                          for dilation in dilations]

        n_conv = len(self.dilations)

        self.weights = []
        for i in range(n_conv):
            n_in = in_channels + blocksize * i

            weight = torch.nn.Parameter(torch.Tensor(
                blocksize, n_in, *self.kernel_size))

            self.register_parameter('weight{}'.format(i), weight)
            self.weights.append(weight)

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(n_conv * blocksize))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.weights:
            torch.nn.init.kaiming_uniform_(weight, a=np.sqrt(5))

        if self.bias is not None:
            # TODO: improve
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weights[0])
            bound = 1 / np.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return msdblock(input, self.bias, self.dilations, self.blocksize,
                        *self.weights)