import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderBlock_Depth2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super(EncoderBlock_Depth2, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=kernel // 2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel, padding=kernel // 2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x_skip, size = x.clone(), x.size()
        x, indices = self.pool(x)
        return x, indices, size, x_skip

class EncoderBlock_Depth3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super(EncoderBlock_Depth3, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=kernel // 2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel, padding=kernel // 2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel, padding=kernel // 2)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x_skip, size = x.clone(), x.size()
        x, indices = self.pool(x)
        return x, indices, size, x_skip

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=kernel // 2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel, padding=kernel // 2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel, padding=kernel // 2)
        self.bn3 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x

class DecoderBlock_Depth3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super(DecoderBlock_Depth3, self).__init__()

        self.MaxUnpooling = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=kernel // 2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel, padding=kernel // 2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel, padding=kernel // 2)
        self.bn3 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x, indices, output_size, cat_x):
        x = self.MaxUnpooling(x, indices, output_size=output_size)
        x = torch.cat([x, cat_x], dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x

class DecoderBlock_Depth2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super(DecoderBlock_Depth2, self).__init__()

        self.MaxUnpooling = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=kernel // 2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel, padding=kernel // 2)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, indices, output_size, cat_x):
        x = self.MaxUnpooling(x, indices, output_size=output_size)
        x = torch.cat([x, cat_x], dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class DecoderBlock_Depth1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super(DecoderBlock_Depth1, self).__init__()

        self.MaxUnpooling = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=kernel // 2)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x, indices, output_size, cat_x):
        x = self.MaxUnpooling(x, indices, output_size=output_size)
        x = torch.cat([x, cat_x], dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        return x

class SegUnet(nn.Module):
    def __init__(self, in_channels, n_labels, return_bottleneck=False):
        super(SegUnet, self).__init__()
        self.return_bottleneck = return_bottleneck

        # Encoder
        self.down_block_1 = EncoderBlock_Depth2(in_channels=in_channels, out_channels=64, kernel=3)
        self.down_block_2 = EncoderBlock_Depth2(in_channels=64, out_channels=128, kernel=3)
        self.down_block_3 = EncoderBlock_Depth3(in_channels=128, out_channels=256, kernel=3)
        self.down_block_4 = EncoderBlock_Depth3(in_channels=256, out_channels=512, kernel=3)
        self.down_block_5 = EncoderBlock_Depth3(in_channels=512, out_channels=512, kernel=3)

        # Bottleneck
        self.bottleneck = BottleneckBlock(in_channels=512, out_channels=512, kernel=3)

        # Decoder
        self.up_block_1 = DecoderBlock_Depth3(in_channels=512 + 512, out_channels=512, kernel=3)
        self.up_block_2 = DecoderBlock_Depth3(in_channels=512 + 512, out_channels=256, kernel=3)
        self.up_block_3 = DecoderBlock_Depth3(in_channels=256 + 256, out_channels=128, kernel=3)
        self.up_block_4 = DecoderBlock_Depth2(in_channels=128 + 128, out_channels=64, kernel=3)
        self.up_block_5 = DecoderBlock_Depth1(in_channels=64 + 64, out_channels=64, kernel=3)

        self.final_conv = nn.Conv2d(64, n_labels, kernel_size=1)

        self.final_bn = nn.BatchNorm2d(n_labels)
    
    def forward(self, x):
        # Encoder
        db_1, mask_1, size1, skip1 = self.down_block_1(x)
        db_2, mask_2, size2, skip2 = self.down_block_2(db_1)
        db_3, mask_3, size3, skip3 = self.down_block_3(db_2)
        db_4, mask_4, size4, skip4 = self.down_block_4(db_3)
        db_5, mask_5, size5, skip5 = self.down_block_5(db_4)

        # Bottleneck
        bottleneck = self.bottleneck(db_5)

        # Decoder
        ub_1 = self.up_block_1(bottleneck, mask_5, size5, skip5)
        ub_2 = self.up_block_2(ub_1, mask_4, size4, skip4)
        ub_3 = self.up_block_3(ub_2, mask_3, size3, skip3)
        ub_4 = self.up_block_4(ub_3, mask_2, size2, skip2)
        ub_5 = self.up_block_5(ub_4, mask_1, size1, skip1)

        f_c = self.final_conv(ub_5)
        f_bn = self.final_bn(f_c)

        # Output Activation on channel dimension (channels = label for each pixel)
        output = F.softmax(f_bn, dim=1)
        # [batch size, num_labels, height, width]
        # [0, :, 1, 1] -> 4 values (one for each class that the pixel can be)
        if self.return_bottleneck:
            return output, bottleneck
        else:
            return output



if __name__ == "__main__":
    model = SegUnet(in_channels=3, n_labels=3)


    input_tensor = torch.randn((1, 3, 512, 512))
    output = model(input_tensor)

    print(f"Output shape: {output.shape}")