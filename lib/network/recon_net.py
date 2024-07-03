import torch
import torch.nn as nn
from .unet import UNet
from lib.config import cfg, args
from .submodules import ConvLayer

class E2IM(nn.Module):
    def __init__(self, num_input_channels=6,
                         num_output_channels=1,
                         skip_type="sum",
                         activation='sigmoid',
                         num_encoders=4,
                         base_num_channels=32,
                         num_residual_blocks=2,
                         norm="BN",
                         use_upsample_conv=True):
        super(E2IM, self).__init__()

        self.unet = UNet(num_input_channels=num_input_channels,
                         num_output_channels=num_output_channels,
                         skip_type=skip_type,
                         activation=activation,
                         num_encoders=num_encoders,
                         base_num_channels=base_num_channels,
                         num_residual_blocks=num_residual_blocks,
                         norm=norm,
                         use_upsample_conv=use_upsample_conv)

    def forward(self, event_tensor):
        """
        :param event_tensor: N x num_bins x H x W
        :return: a predicted image of size N x 1 x H x W, taking values in [0,1].
        """
        return self.unet.forward(event_tensor)
    
    def get_features(self, event_tensor):
        img, feat = self.unet.get_features(event_tensor)
        return img, feat


class E2DPT(nn.Module):
    def __init__(self, num_input_channels=6,
                         num_output_channels=1,
                         skip_type="sum",
                         activation='sigmoid',
                         num_encoders=4,
                         base_num_channels=32,
                         num_residual_blocks=2,
                         norm="BN",
                         use_upsample_conv=True):
        super(E2DPT, self).__init__()

        self.unet = UNet(num_input_channels=num_input_channels,
                         num_output_channels=num_output_channels,
                         skip_type=skip_type,
                         activation=activation,
                         num_encoders=num_encoders,
                         base_num_channels=base_num_channels,
                         num_residual_blocks=num_residual_blocks,
                         norm=norm,
                         use_upsample_conv=use_upsample_conv)
        
        self.mask_head = nn.Sequential(ConvLayer(base_num_channels, base_num_channels // 2, kernel_size=3, padding=1, norm=norm), 
                                       ConvLayer(base_num_channels // 2, 1,kernel_size=1, activation=activation, norm=norm))
        

    def forward(self, event_tensor):
        """
        :param event_tensor: N x num_bins x H x W
        :return: a predicted image of size N x 1 x H x W, taking values in [0,1].
        """
    
        depth, feat = self.unet.get_features(event_tensor)
        mask = self.mask_head(feat)
        return depth * cfg.model.max_depth_value, mask

    def get_features(self, event_tensor):
        depth, feat = self.unet.get_features(event_tensor)
        mask = self.mask_head(feat)
        depth =  depth * cfg.model.max_depth_value

        return depth, mask, feat


class E2Msk(nn.Module):
    def __init__(self, num_input_channels=6,
                         num_output_channels=1,
                         skip_type="sum",
                         activation='sigmoid',
                         num_encoders=4,
                         base_num_channels=32,
                         num_residual_blocks=2,
                         norm="BN",
                         use_upsample_conv=True):
        super(E2Msk, self).__init__()

        self.unet = UNet(num_input_channels=num_input_channels,
                         num_output_channels=num_output_channels,
                         skip_type=skip_type,
                         activation=activation,
                         num_encoders=num_encoders,
                         base_num_channels=base_num_channels,
                         num_residual_blocks=num_residual_blocks,
                         norm=norm,
                         use_upsample_conv=use_upsample_conv)
        

    def forward(self, event_tensor):
        """
        :param event_tensor: N x num_bins x H x W
        :return: a predicted image of size N x 1 x H x W, taking values in [0,1].
        """
        depth = self.unet.forward(event_tensor)
        return depth * cfg.model.max_depth_value

    def get_features(self, event_tensor):
        depth, feat = self.unet.get_features(event_tensor)
        depth =  depth * cfg.model.max_depth_value

        return depth, feat

