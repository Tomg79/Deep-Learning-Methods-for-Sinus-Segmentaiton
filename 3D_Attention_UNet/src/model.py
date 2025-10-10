# model.py
import torch
from torch import nn
from torch.nn import functional as F

class UNetConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            UNetConvBlock(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = UNetConvBlock(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffZ = x2.size()[2] - x1.size()[2]; diffY = x2.size()[3] - x1.size()[3]; diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(nn.Conv3d(F_g, F_int, 1), nn.BatchNorm3d(F_int))
        self.W_x = nn.Sequential(nn.Conv3d(F_l, F_int, 1), nn.BatchNorm3d(F_int))
        self.psi = nn.Sequential(nn.Conv3d(F_int, 1, 1), nn.BatchNorm3d(1), nn.Sigmoid())
    def forward(self, g, x):
        g1 = self.W_g(g); x1 = self.W_x(x)
        psi = F.relu(g1 + x1, inplace=True); psi = self.psi(psi)
        return x * psi

class AttentionUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(AttentionUNet, self).__init__()
        self.inc = UNetConvBlock(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.att1 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.up2 = Up(512, 256)
        self.att2 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.up3 = Up(256, 128)
        self.att3 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.up4 = Up(128, 64)
        self.att4 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.outc = OutConv(64, out_channels)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        d5 = self.up1(x5, x4)
        x4 = self.att1(g=d5, x=x4)
        d4 = self.up2(d5, x3)
        x3 = self.att2(g=d4, x=x3)
        d3 = self.up3(d4, x2)
        x2 = self.att3(g=d3, x=x2)
        d2 = self.up4(d3, x1)
        x1 = self.att4(g=d2, x=x1)
        logits = self.outc(d2)
        return logits