import torch
from torch import nn

def residual(in_chan, out_channel):
    residual = nn.Sequential(
            nn.BatchNorm2d(num_features=in_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_chan, out_channel, kernel_size=3, stride=1, padding=1),
        )
    return residual

# def residual_maxpool(in_chan, out_channel):
#     res_max = nn.Sequential(
#         residual(in_chan, out_channel),
#         nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
#     )

#     return res_max

def residual_decode(in_chan, out_channel):
    residual = nn.Sequential(
            # nn.MaxUnpool2d(kernel_size=2, stride=1),
            nn.BatchNorm2d(num_features=in_chan),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_chan, out_channel, kernel_size=3, stride=1, padding=1),
        )
    return residual

class BasicBlock(nn.Module):

    def __init__(self, in_channel, out_channel, resblock):
        super(BasicBlock, self).__init__()

        # first block changes channel size, then keeps the same for the 2 other
        self.sub1 = resblock(in_channel, out_channel)
        self.sub23 = resblock(out_channel, out_channel)
        
        if in_channel < out_channel:
            self.skip = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1)
        elif in_channel > out_channel:
            self.skip = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=1, stride=1)
        else:
            raise ValueError("Basis block: in_channel and out_channel should not be equal")

    def forward(self, x):
        out = self.sub1(x)  # first block changes channel size
        out = self.sub23(out)
        out = self.sub23(out)

        out = torch.add(out, self.skip(x))

        return out


class SIDE(nn.Module):

    def __init__(self):
        super().__init__()  # super(self, SIDE).__init__() for backward compatiility

        self.residualAdapt = BasicBlock(12, 64, residual)
        # seperate as we need the result for the forward pass
        
        self.residual1 = BasicBlock(64, 128, residual)
        
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # ! skip other ones, decode from 8 by 8


        # * Upsampling

        # unpool needs additionnal arg (indices), so seperate from residual block
        # nn.Sequential doesn't allow for additional params
        # https://stackoverflow.com/questions/59912850/autoencoder-maxunpool2d-missing-indices-argument
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.up1 = BasicBlock(128, 64, residual_decode)
        # maxunpool, 
        #self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=1)
        # need element wise sum in forward before last residual block
        self.final = BasicBlock(64, 1, residual_decode)
    

    def forward(self, x):
        x = self.residualAdapt(x)     # get from 12 to 64 channels
        #print("shape after resadapt : " + str(x.shape))
        out, ind1 = self.maxpool(x)    # first maxpool to 16x16x128
        #print("shape after first_maxpool : " + str(out.shape))
        out = self.residual1(out)     
        #print("shape after res1 : " + str(out.shape))   
        out, ind2 = self.maxpool(out)    # second maxpool to 8x8x256
        #print("shape after maxpool 2 : " + str(out.shape)) 
        out = self.unpool(out, ind2)    # ind 2 as they are the last ones
        #print("shape after unpool 1 : " + str(out.shape))
        out = self.up1(out)
        #print("shape after up1 : " + str(out.shape))
        out = self.unpool(out, ind1)
        #print("shape after unpool2 : " + str(out.shape))

        out = torch.add(x, out)
        #print("shape out after add : " + str(out.shape))
        out = self.final(out)
        #print("shape out before squeeze : " + str(out.shape))
        # need to get 32x32 tensor to compare to label
        out = out.squeeze()
        # print("shape out after squeeze : " + str(out.shape))
        return out