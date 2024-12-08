import torch
from torch import nn

def residual(in_chan, out_channel):
    # ! missing the 1x1 conv addition

    residual = nn.Sequential(
            nn.BatchNorm2d(num_features=in_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_chan, out_channel, kernel_size=3, stride=1, padding=1),
        )
    return residual

def residual_maxpool(in_chan, out_channel):
    res_max = nn.Sequential(
        residual(in_chan, out_channel),
        nn.MaxPool2d(kernel_size=2, stride=1, return_indices=True)
    )

    return res_max

def residual_decode(in_chan, out_channel):
    residual = nn.Sequential(
            # nn.MaxUnpool2d(kernel_size=2, stride=1),
            nn.BatchNorm2d(num_features=in_chan),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_chan, out_channel, kernel_size=3, stride=1, padding=1),
        )
    return residual


class SIDE(nn.Module):

    def __init__(self):
        super().__init__()  # super(self, SIDE).__init__() for backward compatiility

        self.residualAdapt = residual_maxpool(12, 64)

        # self.residual1 = nn.Sequential(
        #     nn.BatchNorm2d(num_features=64),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        #     nn.MaxPool2d(kernel_size=2, stride=1),
        # )

        self.residual1 = residual_maxpool(64, 128)

        # self.residual2 = nn.Sequential(
        #     nn.BatchNorm2d(num_features=128),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
        #     nn.MaxPool2d(kernel_size=2, stride=1),  
        # )

        # ! skip these ones, decode from 8 by 8
        # self.residual2 = residual_maxpool(128, 256)

        # self.residual3 = residual_maxpool(256, 512)

        # * Upsampling

         # unpool needs additionnal arg (indices), so seperate from residual block
        self.unpool1 = nn.Sequential(nn.MaxUnpool2d(kernel_size=2, stride=1))
        self.up1 = residual_decode(128, 64)
        # maxunpool, 
        self.unpool2 = nn.Sequential(nn.MaxUnpool2d(kernel_size=2, stride=1))
        # then element wise sum

        self.final = residual_decode(64, 1)


        # self.final = nn.Sequential(
        #     nn.Conv2d(484, 256, kernel_size=1, stride=1),           # 485 = 256 + 128 + 64 + 32 + 4 (input bands)
        #     nn.BatchNorm2d(num_features=256),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 6, kernel_size=1, stride=1)
        # )

    

    def forward(self, x):

        x, ind1 = self.residualAdapt(x)
        x1, ind2 = self.residual1(x)
        x1 = self.unpool1(x1, ind1)    
        x1 = self.up1(x1)
        x1 = self.unpool2(x1, ind2)
        # torch add
        x = torch.add(x, x1)
        x = self.final(x)

        return x