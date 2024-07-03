import torch
from torch import nn
from lib.config import cfg, args
# from lib.network.asnet import ASNet
from lib.network.recon_net import E2IM, E2DPT
from lib.network.gsregressor import GSRegressor

class EventGaussian(nn.Module):
    def __init__(self):
        super(EventGaussian, self).__init__()
        self.depth_estimator = E2DPT(num_input_channels=8)
        self.intensity_estimator = E2IM(num_input_channels=32+8)
        self.regressor = GSRegressor(input_dim=2 + 32 + 8)
        self.gt_depth = False
        self.us_mask = "net"
        # self.proj1 = nn.Sequential(nn.Conv2D())

    def forward(self, batch):
        leT = torch.cat([batch["leframe"], batch["left_voxel"]], dim=1)
        riT = torch.cat([batch["reframe"], batch["right_voxel"]], dim=1)
        b = leT.shape[0]
        inp = torch.cat([leT, riT], dim=0)

        #only available for debugging
        if not self.gt_depth:
            depths, masks, dfeats = self.depth_estimator.get_features(inp)
            depthL, depthR = depths[:b], depths[b:]
            masksL, masksR = masks[:b], masks[b:]
            dfeatsL, dfeatsR = dfeats[:b], dfeats[b:] #[b, 32, H, W]
        else:  #debug only
            depthL, depthR = batch["ldepth"].unsqueeze(1), batch["rdepth"].unsqueeze(1)

        # only available for debugging
        if self.us_mask == "gt":
            maskL, maskR = batch["lmask"].unsqueeze(1), batch["rmask"].unsqueeze(1)
        elif self.us_mask == "net":
            maskL, maskR = masksL, masksR
        elif self.us_mask == "none":
            maskL, maskR = torch.ones_like(depthL).to(depthL.device), torch.ones_like(depthR).to(depthL.device)

        depthL = depthL * maskL
        depthR = depthR * maskR
#
        L_img_inp = torch.cat([dfeatsL ,leT], dim=1) #depthFeat, frame, voxel
        R_img_inp = torch.cat([dfeatsR ,riT], dim=1)
        img_inp = torch.cat([L_img_inp, R_img_inp], dim=0)
        img, ifeats = self.intensity_estimator.get_features(img_inp)
        imgL, imgR = torch.split(img, b, dim=0)
        ifeatL, ifeatR = torch.split(ifeats, b, dim=0)
#
        imgL = imgL * maskL
        imgR = imgR * maskR
        # imgL, imgR = batch["lim"], batch["rim"]
        L_gs_inp = torch.cat([depthL, imgL, ifeatL, leT], dim=1) # depthL, imgL, ifeatL, leT
        R_gs_inp = torch.cat([depthR, imgR, ifeatR, riT], dim=1)
        gs_inp = torch.cat([L_gs_inp, R_gs_inp], dim=0)
#
        rot, scale, opacity = self.regressor(gs_inp)

        return {
            "lview":{
                    "depth":depthL,
                    "mask":maskL,
                    "pts_valid":maskL.squeeze().reshape(b, -1),
                    "img": imgL,
                    "rot":rot[:b],
                    "scale":scale[:b],
                    "opacity":opacity[:b]
            },
            "rview":{
                    "depth":depthR,
                    "mask":maskR,
                    "pts_valid":maskR.squeeze().reshape(b, -1),
                    "img": imgR,
                    "rot":rot[b:],
                    "scale":scale[b:],
                    "opacity":opacity[b:]
            }
        }


        