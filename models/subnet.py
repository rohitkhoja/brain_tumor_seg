import torch
import torch.nn as nn
import torch.nn.functional as F

class Subnet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Subnet, self).__init__()

        self.conv1 = nn.Conv2d(input_dim, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, output_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x):
        # flatten input
        # x = F.flatten(x, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.softmax(x, dim=1)
        return x