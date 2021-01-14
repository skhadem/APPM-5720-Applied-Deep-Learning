"""
Note: These classes are copy-pasted from ./resnet.ipynb
In that notebook, there is a much more exploratory way of building these out,
with comments. This is meant to be imported so that other tools can be explored 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from functools import partial
from collections import OrderedDict

class Conv2dPad(nn.Conv2d):
    '''Custom Conv2D class to handle padding'''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Automatically sets the padding based on the kernel size
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)

Conv3x3 = partial(Conv2dPad, kernel_size=3, bias=False)

# Helper function that returns a Conv. followed by a batch norm, forwards all args to conv layer
def create_conv_bn(in_dim, out_dim, conv, *args, **kwargs):
    return nn.Sequential(OrderedDict({
            'Convolution' : conv(in_dim, out_dim, *args, **kwargs),
            'Batch Norm' : nn.BatchNorm2d(out_dim)
        }))

class ResidualBlock(nn.Module):
    '''
    Base class for each type of residual block. Keeps track of the dimensions and specifies
    the shortcut to use, if necessary. The main conv blocks should be put into `self.blocks`
    '''
    def __init__(self, in_dim, out_dim, expansion=1, downsampling=1, conv=Conv3x3, *args, **kwargs):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.expansion = expansion
        self.downsampling = downsampling
        self.conv = conv
        self.blocks = nn.Identity()

        self.expanded_dim = self.out_dim * self.expansion
        self.apply_shortcut = self.in_dim != self.expanded_dim

        if self.apply_shortcut:
            self.shortcut = create_conv_bn(in_dim, self.expanded_dim, nn.Conv2d, kernel_size=1, stride=self.downsampling, bias=False)
        else:
            # If the dimensions line up, the input will be passed directly through the residual path
            # Make this None so that it doesn't appear when printing the  
            self.shortcut = None

    # Main forward call
    def forward(self, x):
        residual = x
        if self.apply_shortcut:
            residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        return x

class BasicBlock(ResidualBlock):
    """ Basic ResNet block, has two convs with a Relu in between, each with a batch norm
    """
    # Static property
    expansion = 1
    def __init__(self, in_dim, out_dim, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_dim, out_dim, *args, **kwargs)
        self.blocks = nn.Sequential(
            create_conv_bn(self.in_dim, self.out_dim, self.conv, bias=False, stride=self.downsampling),
            activation(),
            create_conv_bn(self.out_dim, self.expanded_dim, self.conv, bias=False),
        )

class BottleNeckBlock(ResidualBlock):
    """Applies a 1x1 conv, 3x3 conv (with downsampling using strides), and then 1x1 conv
    Each has a Relu and batch norm in between
    """
    # Static property
    expansion = 4
    def __init__(self, in_dim, out_dim, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_dim, out_dim, expansion=4, *args, **kwargs)
        self.blocks = nn.Sequential(
            create_conv_bn(self.in_dim, self.out_dim, self.conv, kernel_size=1),
            activation(),
            create_conv_bn(self.out_dim, self.out_dim, self.conv, kernel_size=3, stride=self.downsampling),
            activation(),
            create_conv_bn(self.out_dim, self.expanded_dim, self.conv, kernel_size=1)
        )

class ResNetLayer(nn.Module):
    """A module containing many residual blocks

    Args:
        in_dim: Input number of channels
        out_dim: Output number of channels
        block_type: A ResidualBlock to use
        n: The number of blocks to stack together
    """
    def __init__(self, in_dim, out_dim, block_type, n=1, *args, **kwargs):
        super().__init__()
        # If the dimensions change, apply downsampling (using stride)
        downsampling = 1 if in_dim == out_dim else 2

        self.residual_blocks = nn.ModuleList()
        self.residual_blocks.append(block_type(in_dim, out_dim, downsampling=downsampling, *args, **kwargs))
        for _ in range(n - 1):
            self.residual_blocks.append(block_type(out_dim * block_type.expansion, out_dim, *args, **kwargs)) 

    def forward(self, x):
        for residual_block in self.residual_blocks:
            x = residual_block(x)
        return x

class ResNet(nn.Module):
    """Full ResNet Model
    Contains an initial input gate, and then multiple ResNet layers, followed by a 
    """
    def __init__(self, in_dim, num_classes, activation=nn.ReLU, block_sizes=[64, 128, 256, 512], depths=[2,2,2,2], *args,**kwargs):
        super().__init__()
        if 'block_type' not in kwargs:
            raise RuntimeError('Must specify block_type!')
        
        if len(block_sizes) != len(depths):
            raise RuntimeError('Expected same length for block sizes and depths, but got %s and %s'%(len(block_sizes), len(depths)))

        block_type = kwargs['block_type']
        
        # Initial input gate, reduces the dimensionality of the image, and makes the channels match
        # that of the first block
        self.gate = nn.Sequential(
            nn.Conv2d(in_dim, block_sizes[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(block_sizes[0]),
            activation(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layers = nn.ModuleList()
        self.layers.append(ResNetLayer(block_sizes[0], block_sizes[0], n=depths[0], activation=activation, *args, **kwargs))
        for i in range(1, len(block_sizes)):
            layer_in_dim = block_sizes[i - 1]
            layer_out_dim = block_sizes[i] 
            self.layers.append(ResNetLayer(layer_in_dim * block_type.expansion, layer_out_dim, n=depths[i], activation=activation, *args, **kwargs))

        last_block = self.layers[-1].residual_blocks[-1]
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(last_block.out_dim * last_block.expansion, num_classes)

    def forward(self, x):
        # Initial gating
        x = self.gate(x)

        # Conv residual blocks
        for layer in self.layers:
            x = layer(x)
        
        # Final FC layer
        x = self.avg(x)
        # Flatten output
        x = x.view(x.size(0), -1) 
        x = self.fc(x)
        return x