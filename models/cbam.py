import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, pool_types=["avg", "max"]):
        super(ChannelAttention, self).__init__()
        self.pool_types = pool_types
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == "avg":
                avg_out = self.fc2(self.relu1(self.fc1(F.adaptive_avg_pool2d(x, 1))))
                channel_att_sum = avg_out if channel_att_sum is None else channel_att_sum + avg_out
            elif pool_type == "max":
                max_out = self.fc2(self.relu1(self.fc1(F.adaptive_max_pool2d(x, 1))))
                channel_att_sum = max_out if channel_att_sum is None else channel_att_sum + max_out
        return x * self.sigmoid(channel_att_sum)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return x * self.sigmoid(x)



class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], kernel_size=7):
        super(CBAM, self).__init__()
        self.ChannelAttention = ChannelAttention(gate_channels, reduction_ratio, pool_types)
        self.SpatialAttention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ChannelAttention(x)
        x = self.SpatialAttention(x)
        return x

