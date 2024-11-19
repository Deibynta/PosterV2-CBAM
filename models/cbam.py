import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelAttention, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = torch.mean(x, dim=(2, 3), keepdim=True)
                channel_att_raw = self.mlp(avg_pool.view(x.size(0), -1))
            elif pool_type == 'max':
                max_pool, _ = torch.max(x, dim=(2, 3), keepdim=True)
                channel_att_raw = self.mlp(max_pool.view(x.size(0), -1))
            channel_att_sum = channel_att_sum + channel_att_raw if channel_att_sum is not None else channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3)
        return x * scale



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

