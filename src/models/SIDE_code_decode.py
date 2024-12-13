import torch
from torch import nn

def residual(in_chan, out_channel):
    residual = nn.Sequential(
            nn.BatchNorm2d(num_features=in_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_chan, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
        )
    return residual

# def residual_maxpool(in_chan, out_channel):
#     res_max = nn.Sequential(
#         residual(in_chan, out_channel),
#         nn.MaxPool2d(kernel_size=2, stride=1, return_indices=True)
#     )

#     return res_max

def residual_decode(in_chan, out_channel):
    residual = nn.Sequential(
            # nn.MaxUnpool2d(kernel_size=2, stride=1),
            nn.BatchNorm2d(num_features=in_chan),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_chan, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
        )
    return residual


class BasicBlock(nn.Module):

    def __init__(self, in_channel, out_channel, resblock):
        super(BasicBlock, self).__init__()

        # first block changes channel size, then keeps the same for the 2 other
        self.sub1 = resblock(in_channel, out_channel)
        self.sub23 = resblock(out_channel, out_channel)
        
        if in_channel < out_channel:
            self.skip = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=1)
        elif in_channel > out_channel:
            self.skip = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=1, stride=1, padding=1)
        else:
            raise ValueError("Basis block: in_channel and out_channel should not be equal")



    def forward(self, x):
        out = self.sub1(x)  # first block changes channel size
        out = self.sub23(out)
        out = self.sub23(out)

        out += self.skip(x)

        return out

class SIDE(nn.Module):

    def __init__(self):
        super().__init__()  # super(self, SIDE).__init__() for backward compatiility

        #self.residualAdapt = residual_maxpool(12, 64)
        self.residualAdapt = BasicBlock(12, 64, residual)
        self.residual1 = BasicBlock(64, 128, residual)

        #self.residual1 = residual_maxpool(64, 128)

        # ! skip other layers, decode from 8 by 8

        # * Upsampling

         # unpool needs additionnal arg (indices), so seperate from residual block
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=1)
        #self.up1 = residual_decode(128, 64)
        self.up1 = BasicBlock(128, 64, residual_decode)
        # maxunpool, 
        #self.unpool2 = nn.Sequential(nn.MaxUnpool2d(kernel_size=2, stride=1))

        self.final = BasicBlock(64, 1, residual_decode)

    

    def forward(self, x):

        x, ind1 = self.residualAdapt(x)
        x1, ind2 = self.residual1(x)
        x1 = self.unpool(x1, ind1)    
        x1 = self.up1(x1)
        x1 = self.unpool(x1, ind2)
        # skip connection
        x = torch.add(x, x1)
        x = self.final(x)

        return x