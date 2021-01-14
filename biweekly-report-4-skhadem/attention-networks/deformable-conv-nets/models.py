import torch
import torch.nn as nn
from torchvision.ops.deform_conv import DeformConv2d

class Model(nn.Module):
    def __init__(self, inp_channels=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Conv 1 block
        self.features = nn.Sequential(
            nn.Conv2d(inp_channels, 32, (3, 3)),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, (3, 3), stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, (3, 3), stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(4) # Gotten by running once
        )
        self.classify = nn.Linear(128, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 128) # flatten
        x = self.classify(x)
        return x

class DeformConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        kernel_size = (3, 3)

        self.offset_net = nn.Conv2d(
            in_channels=in_channels,
            out_channels=2*kernel_size[0]*kernel_size[1],
            kernel_size=kernel_size,
            padding=1,
            stride=stride,
            bias=True
        )

        self.deform_conv = DeformConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=1,
            stride=stride,
            bias = False
        )

    def forward(self, x):
        offsets = self.offset_net(x)
        x = self.deform_conv(x, offsets)
        return x

class DeformedConvModel(nn.Module):
    def __init__(self, inp_channels=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Conv 1 block
        self.features = nn.Sequential(
            nn.Conv2d(inp_channels, 32, (3, 3)),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            DeformConv(32, 64, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            DeformConv(64, 128),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            DeformConv(128, 128, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(4) # Gotten by running once
        )
        self.classify = nn.Linear(128, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 128) # flatten
        x = self.classify(x)
        return x