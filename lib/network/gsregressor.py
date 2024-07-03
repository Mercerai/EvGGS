import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.Sequential()

        if stride == 1 and in_planes == planes:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)


    def forward(self, x):
        y = x
        y = self.conv1(y)
        y = self.norm1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.relu(y)

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)

class GSRegressor(nn.Module):
    def __init__(self, input_dim=1+1+8, hidden_dim = 256, norm_fn='group'):
        super().__init__()
        self.embedding = nn.Conv2d(input_dim, hidden_dim, kernel_size=1, stride=1)

        self.res1 = ResidualBlock(hidden_dim, hidden_dim // 4, norm_fn=norm_fn)

        self.rot_head = nn.Sequential(
            nn.Conv2d(hidden_dim // 4, hidden_dim // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 4, 4, kernel_size=1),
        )
        self.scale_head = nn.Sequential(
            nn.Conv2d(hidden_dim // 4, hidden_dim // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 4, 3, kernel_size=1),
            nn.Softplus(beta=100)
        )
        self.opacity_head = nn.Sequential(
            nn.Conv2d(hidden_dim // 4, hidden_dim // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x intensity [B,1,H,W]
          depth [B,1,H,W]
          eframe [B,3,H,W]
          img_feat [B,320,H,W]
        """
        x = self.embedding(x)
        out = self.res1(x)

        rot_out = self.rot_head(out)
        rot_out = torch.nn.functional.normalize(rot_out, dim=1)

        # scale head
        scale_out = torch.clamp_max(self.scale_head(out), 0.001)

        # opacity head
        opacity_out = self.opacity_head(out)

        return rot_out, scale_out, opacity_out
    

    