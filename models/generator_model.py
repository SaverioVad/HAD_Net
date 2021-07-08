import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, num_mods, k, p, num_class):
        super(Generator, self).__init__()
        
        # Encoder
        self.conv_d1  = Down(num_mods, [1*k, 1*k], k, p)
        self.conv_d2  = Down(1*k, [2*k, 2*k], k, p)
        self.conv_d3  = Down(2*k, [4*k, 4*k], k, p)
        self.conv_d4  = Down(4*k, [8*k, 8*k], k, p)
        self.conv_c5  = Center(8*k, [16*k, 8*k], k, p)
        
        # Decoder
        self.conv_u4  = Up(8*k, [8*k, 4*k], k, p)
        self.conv_u3  = Up(4*k, [4*k, 2*k], k, p)
        self.conv_u2  = Up(2*k, [2*k, 1*k], k, p)
        self.conv_u1  = Up(1*k, [1*k, 1*k], k, p)
        self.conv_out = nn.Conv3d(1*k, num_class, 1, padding=0)
        
    def forward(self, x):
        
        # Pass the input through the encoder.
        maxpool_d1, conv_d1 = self.conv_d1(x)
        maxpool_d2, conv_d2 = self.conv_d2(maxpool_d1)
        maxpool_d3, conv_d3 = self.conv_d3(maxpool_d2)
        maxpool_d4, conv_d4 = self.conv_d4(maxpool_d3)
        conv_c5 = self.conv_c5(maxpool_d4)

        # Pass the results to the to the decoder.
        conv_u4 = self.conv_u4(conv_d4, conv_c5)
        conv_u3 = self.conv_u3(conv_d3, conv_u4)
        conv_u2 = self.conv_u2(conv_d2, conv_u3)
        conv_u1 = self.conv_u1(conv_d1, conv_u2)
        conv_out = self.conv_out(conv_u1)
        
        return conv_out, [[conv_d2,conv_u3],[conv_d3,conv_u4],[conv_d4,conv_c5]]
    
class Down(nn.Module):
    def __init__(self, in_channels, out_channels, initial_k, drop_prob, pool_kernel=2):
        super(Down, self).__init__()
        p = (in_channels>=initial_k) * drop_prob
        self.drop = nn.Dropout3d(p=p)
        self.conv1 = Conv3d_InstanceNorm3d_LeakyReLU(in_channels,     out_channels[0], 3, 1)
        self.conv2 = Conv3d_InstanceNorm3d_LeakyReLU(out_channels[0], out_channels[1], 3, 1)
        self.maxpool = nn.MaxPool3d(pool_kernel)
    def forward(self, x):
        x = self.drop(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.maxpool(x), x
    
class Center(nn.Module):
    def __init__(self, in_channels, out_channels, initial_k, drop_prob):
        super(Center, self).__init__()
        p = (in_channels>=initial_k) * drop_prob
        self.drop = nn.Dropout3d(p=p)
        self.conv1 = Conv3d_InstanceNorm3d_LeakyReLU(in_channels,     out_channels[0], 3, 1)
        self.conv2 = Conv3d_InstanceNorm3d_LeakyReLU(out_channels[0], out_channels[1], 3, 1)
    def forward(self, x):
        x = self.drop(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, initial_k, drop_prob):
        super(Up, self).__init__()
        p = (in_channels>=initial_k) * drop_prob
        self.drop = nn.Dropout3d(p=p)
        # equivalent to a single convolution on a concatenated tensor but saves on memory
        self.conv_up   = nn.Conv3d(in_channels, out_channels[0], 3, padding=1, bias=True)
        self.conv_left = nn.Conv3d(in_channels, out_channels[0], 3, padding=1, bias=False)
        self.inst      = nn.InstanceNorm3d(out_channels[0], affine=True)
        self.relu      = nn.LeakyReLU(inplace=True, negative_slope=1e-2)
        self.conv_out = Conv3d_InstanceNorm3d_LeakyReLU(out_channels[0], out_channels[1], 3, 1)
    def forward(self, c, x):
        x = F.interpolate(x, size=c.size()[-3:])
        c, x = self.drop(c), self.drop(x)
        x = self.relu(self.inst(self.conv_left(c) + self.conv_up(x)))
        x = self.conv_out(x)
        return x
    
class Conv3d_InstanceNorm3d_LeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernal_size, padding):
        super(Conv3d_InstanceNorm3d_LeakyReLU, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernal_size, padding=padding)
        self.inst = nn.InstanceNorm3d(out_channels, affine=True)
        self.relu = nn.LeakyReLU(inplace=True, negative_slope=1e-2)
    def forward(self, x):
        return self.relu(self.inst(self.conv(x)))    
    
    

# Model weight initialization
def init_weights(m):
    if (type(m) == nn.Conv3d or
        type(m) == nn.ConvTranspose3d):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)