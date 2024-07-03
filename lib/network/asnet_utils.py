import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.config import cfg


###############################################################################
""" Fundamental Building Blocks """
###############################################################################


def convbn(in_channels, out_channels, kernel_size, stride, pad, dilation):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_channels)
    )


def convbn_dws(inp, oup, kernel_size, stride, pad, dilation, second_relu=True):
    if second_relu:
        return nn.Sequential(
            # dw
            nn.Conv2d(inp, inp, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad,
                      dilation=dilation, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU6(inplace=True),
            # pw
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=False)
            )
    else:
        return nn.Sequential(
            # dw
            nn.Conv2d(inp, inp, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad,
                      dilation=dilation, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU6(inplace=True),
            # pw
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup)
            )

class MobileV1_Residual(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(MobileV1_Residual, self).__init__()

        self.stride = stride
        self.downsample = downsample
        self.conv1 = convbn_dws(inplanes, planes, 3, stride, pad, dilation)
        self.conv2 = convbn_dws(planes, planes, 3, 1, pad, dilation, second_relu=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out



class MobileV2_Residual(nn.Module):
    def __init__(self, inp, oup, stride, expanse_ratio, dilation=1):
        super(MobileV2_Residual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expanse_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        pad = dilation

        if expanse_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, pad, dilation=dilation, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, pad, dilation=dilation, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)



class InsideBlockConv(nn.Module):
    def __init__(self, in_features, out_features):
        super(InsideBlockConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size=3, padding=(1,1)), # For same padding: pad=1 for filter=3
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True), # inplace=True doesn't create additonal memory. Not always correct operation. But here there is no issue
            nn.Conv2d(out_features, out_features, kernel_size=3, padding=(1,1)),
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x1):
        return self.double_conv(x1)

###############################################################################
""" Feature Extraction """
###############################################################################





class feature_extraction(nn.Module):
    def __init__(self):
        super(feature_extraction, self).__init__()

        self.expanse_ratio = 3
        self.inplanes = 32

        self.firstconv0 = nn.Sequential(MobileV2_Residual(1, 4, 2, self.expanse_ratio),
                                           nn.ReLU(inplace=True),
                                           MobileV2_Residual(4, 16, 1, self.expanse_ratio),
                                           nn.ReLU(inplace=True),
                                           MobileV2_Residual(16, 32, 1, self.expanse_ratio),
                                           nn.ReLU(inplace=True)
                                           )
        self.firstconv1 = nn.Sequential(MobileV2_Residual(1, 4, 2, self.expanse_ratio),
                                           nn.ReLU(inplace=True),
                                           MobileV2_Residual(4, 16, 1, self.expanse_ratio),
                                           nn.ReLU(inplace=True),
                                           MobileV2_Residual(16, 32, 1, self.expanse_ratio),
                                           nn.ReLU(inplace=True)
                                           )
        self.firstconv2 = nn.Sequential(MobileV2_Residual(1, 4, 2, self.expanse_ratio),
                                           nn.ReLU(inplace=True),
                                           MobileV2_Residual(4, 16, 1, self.expanse_ratio),
                                           nn.ReLU(inplace=True),
                                           MobileV2_Residual(16, 32, 1, self.expanse_ratio),
                                           nn.ReLU(inplace=True)
                                           )
       
        
        self.conv3d = nn.Sequential(nn.Conv3d(1, 1, kernel_size=(3, 5, 5), stride=[3, 1, 1], padding=[0, 2, 2]),
                                    nn.BatchNorm3d(1),
                                    nn.ReLU())

        self.layer1 = self._make_layer(MobileV1_Residual, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(MobileV1_Residual, 64, 16, 2, 1, 1)#
        self.layer3 = self._make_layer(MobileV1_Residual, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(MobileV1_Residual, 128, 3, 1, 1, 2)

        self.preconv11 = nn.Sequential(
                                       convbn(320, 256, 1, 1, 0, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(256, 128, 1, 1, 0, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(128, 64, 1, 1, 0, 1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(64, 32, 1, 1, 0, 1)
                                       )

        

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None

        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = [block(self.inplanes, planes, stride, downsample, pad, dilation)]
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)
    
    def forward(self, x):

        x0 = torch.unsqueeze(x[:,0,:,:], 1)
        x1 = torch.unsqueeze(x[:,1,:,:], 1)
        x2 = torch.unsqueeze(x[:,2,:,:], 1)

        x0 = self.firstconv0(x0)
        x1 = self.firstconv1(x1)
        x2 = self.firstconv2(x2)

        B, C, H, W = x0.shape
        interwoven_features = x0.new_zeros([B, 3 * C, H, W])
        xall = interweave_tensors3(interwoven_features, x0, x1, x2)

        xall = torch.unsqueeze(xall, 1)
        xall = self.conv3d(xall)
        xall = torch.squeeze(xall, 1)





        xall = self.layer1(xall)
        xall2 = self.layer2(xall)
        xall3 = self.layer3(xall2)
        xall4 = self.layer4(xall3)

        feature_volume = torch.cat((xall2, xall3, xall4), dim=1)

        xALL = self.preconv11(feature_volume)


  
        return xALL




class volume_build(nn.Module):
    def __init__(self, volume_size):
        super(volume_build, self).__init__()
        self.num_groups = 1
        self.volume_size = volume_size

       

     
       
        self.volume11 = nn.Sequential(
                                      convbn(16, 1, 1, 1, 0, 1),
                                      nn.ReLU(inplace=True))
        self.conv3d = nn.Sequential(nn.Conv3d(1, 16, kernel_size=(8, 3, 3), stride=[8, 1, 1], padding=[0, 1, 1]),
                                    nn.BatchNorm3d(16),
                                    nn.ReLU(),
                                    nn.Conv3d(16, 32, kernel_size=(4, 3, 3), stride=[4, 1, 1], padding=[0, 1, 1]),
                                    nn.BatchNorm3d(32),
                                    nn.ReLU(),
                                    nn.Conv3d(32, 16, kernel_size=(2, 3, 3), stride=[2, 1, 1], padding=[0, 1, 1]),
                                    nn.BatchNorm3d(16),
                                    nn.ReLU())
        
    def forward(self, featL,featR):




        B, C, H, W = featL.shape
        volume = featL.new_zeros([B, self.num_groups, self.volume_size, H, W])


  
        interwoven_features = featL.new_zeros([B, 2 * C, H, W])
        for i in range(self.volume_size):
            
            if i > 0:
                x = interweave_tensors(interwoven_features, featL[:, :, :, :-i], featR[:, :, :, i:])
                x = torch.unsqueeze(x, 1)
                x = self.conv3d(x)
                x = torch.squeeze(x, 2)
                x = self.volume11(x)
                volume[:, :, i, :, i:] = x
            else:
                x = interweave_tensors(interwoven_features, featL, featR)
                x = torch.unsqueeze(x, 1)
                x = self.conv3d(x)
                x = torch.squeeze(x, 2)
                x = self.volume11(x)
                volume[:, :, i, :, :] = x

        volume = volume.contiguous()
        volume = torch.squeeze(volume, 1)

        return volume

    



##############################################################################
""" Disparity Regression Function """
###############################################################################


def disparity_regression(x, maxdisp):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=False) / maxdisp  * cfg.model.max_depth_value



def interweave_tensors(interwoven_features, refimg_fea, targetimg_fea):
    B, C, H, W = refimg_fea.shape
    interwoven_features = interwoven_features[:, :, :, 0:W]
    interwoven_features = interwoven_features*0
    interwoven_features[:,::2,:,:] = refimg_fea
    interwoven_features[:,1::2,:,:] = targetimg_fea
    interwoven_features = interwoven_features.contiguous()
    return interwoven_features
def interweave_tensors3(interwoven_features, refimg_fea, targetimg_fea, targetimg2_fea):
    B, C, H, W = refimg_fea.shape
    interwoven_features = interwoven_features[:, :, :, 0:W]
    interwoven_features = interwoven_features*0
    interwoven_features[:,::3,:,:] = refimg_fea
    interwoven_features[:,1::3,:,:] = targetimg_fea
    interwoven_features[:,2::3,:,:] = targetimg2_fea
    interwoven_features = interwoven_features.contiguous()
    return interwoven_features


###############################################################################
""" Loss Function """
###############################################################################


def model_loss(disp_ests, disp_gt, mask):
    weights = [0.5, 0.5, 0.7, 1.0]
    all_losses = []
    for disp_est, weight in zip(disp_ests, weights):
        # all_losses.append(weight * F.smooth_l1_loss(disp_est[mask], disp_gt[mask], reduction='mean'))
        all_losses.append(weight * F.l1_loss(disp_est[mask], disp_gt[mask], reduction='mean'))
    return sum(all_losses)

def model_loss_light(disp_ests, disp_gt, mask):
    weights = [0.5, 1.0]
    all_losses = []
    for disp_est, weight in zip(disp_ests, weights):
        all_losses.append(weight * F.l1_loss(disp_est[mask], disp_gt[mask], reduction='mean'))
    return sum(all_losses)