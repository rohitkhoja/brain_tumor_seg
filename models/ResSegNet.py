import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs=2):
        super(ConvBlock, self).__init__()
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class ResSegNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=4):
        super(ResSegNet, self).__init__()

        self.encoder1 = ConvBlock(in_channels, 64)
        self.encoder2 = ConvBlock(64, 128)
        self.encoder3 = ConvBlock(128, 256)
        self.encoder4 = ConvBlock(256, 512)
        self.encoder5 = ConvBlock(512, 512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        
        self.decoder5 = ConvBlock(1024, 512, num_convs=3)  
        self.decoder4 = ConvBlock(1024, 512, num_convs=3)  
        self.decoder3 = ConvBlock(768, 256, num_convs=3)  
        self.decoder2 = ConvBlock(384, 128, num_convs=3)  
        self.decoder1 = ConvBlock(192, 64)

        self.upsample = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        x = self.pool(enc1)

        enc2 = self.encoder2(x)
        x = self.pool(enc2)

        enc3 = self.encoder3(x)
        x = self.pool(enc3)

        enc4 = self.encoder4(x)
        x = self.pool(enc4)

        enc5 = self.encoder5(x)
        x = self.pool(enc5)

        x = self.upsample(x)
        diffY = enc5.size()[2] - x.size()[2]
        diffX = enc5.size()[3] - x.size()[3]
        padY_top = diffY // 2
        padY_bottom = diffY - padY_top
        padX_left = diffX // 2
        padX_right = diffX - padX_left

   
        x = F.pad(x, (padX_left, padX_right, padY_top, padY_bottom))

      
        x = torch.cat([enc5, x], dim=1)

        x = self.decoder5(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = torch.cat([enc4, x], dim=1)
        x = self.decoder4(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = torch.cat([enc3, x], dim=1)
        x = self.decoder3(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = torch.cat([enc2, x], dim=1)
        x = self.decoder2(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = torch.cat([enc1, x], dim=1)
        x = self.decoder1(x)

        x = self.final_conv(x)

        return x
