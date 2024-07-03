from lib.config import cfg, args
from lib.dataset import EventDataloader
from lib.recorder import Logger, file_backup
from lib.network import model_loss_light, model_loss, EventGaussian
from lib.renderer import pts2render, depth2pc
from lib.utils import depth2img
import numpy as np
import imageio
import cv2
import os
from pathlib import Path
from tqdm import tqdm
import logging
import torch
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

cs = cfg.cs

class Trainer:
    def __init__(self) -> None:
        device = torch.device('cuda:{}'.format(cfg.local_rank))
        self.device = device
        which_test = "val"
        self.train_loader = EventDataloader(cfg.dataset.base_folder, split="train", num_workers=1,\
                                             batch_size=1, shuffle=False)
        
        self.val_loader = EventDataloader(cfg.dataset.base_folder, split=which_test, num_workers=1,\
                                             batch_size=1, shuffle=False)

        self.len_val = len(self.val_loader)
        self.model = EventGaussian().to(self.device)
        print(" Load warm up parameters ... ")
        d_warmup = False
        int_warmup = False
        if cfg.depth_warmup_ckpt is not None:
            self.model.depth_estimator.load_state_dict(torch.load(cfg.depth_warmup_ckpt)["network"])
            d_warmup = True
        if cfg.intensity_warmup_ckpt is not None:
            self.model.intensity_estimator.load_state_dict(torch.load(cfg.intensity_warmup_ckpt)["network"])
            int_warmup = True
        print(f" Using depth warm up {d_warmup} ; intensity warm up {int_warmup}")

        # self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=cfg.wdecay, eps=1e-8)
        dpt_params = list(map(id,self.model.depth_estimator.parameters())) + list(map(id,self.model.intensity_estimator.parameters()))
        rest_params = filter(lambda x:id(x) not in dpt_params,self.model.parameters())
        self.optimizer = optim.Adam([
            {'params':self.model.depth_estimator.parameters(), 'lr':0.00001},
            {'params':self.model.intensity_estimator.parameters(), 'lr':0.00001},
            {'params':rest_params, 'lr':0.0005},
        ], lr=0.0005, weight_decay=cfg.wdecay, eps=1e-8)
        
        # self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, 0.001, 1000000 + 100,
        #                                                pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=15000, gamma=0.9)
        
        self.logger = Logger(self.scheduler, cfg.record)

        self.total_steps = 0
        self.target_epoch = cfg.target_epoch

        if cfg.restore_ckpt:
            self.load_ckpt(cfg.restore_ckpt)

        self.model.train()

    def train(self):
        # self.model.eval()
        # self.run_eval()
        # self.model.train()
        for e in range(self.target_epoch):
            for idx, batch in enumerate(tqdm(self.train_loader)):
                batch = self.to_cuda(batch)

                ### model and loss computing ##
                gt = {}
                gt["cim"] = batch["cim"]

                gt["lim"], gt["rim"], gt["ldepth"], gt["rdepth"], gt["lmask"], gt["rmask"] \
                            = batch["lim"], batch["rim"], batch["ldepth"], batch["rdepth"], batch["lmask"], batch["rmask"]

                batch["left_event_tensor"] = torch.cat([batch["leframe"], batch["left_voxel"]], dim=1)
                batch["right_event_tensor"] = torch.cat([batch["reframe"], batch["right_voxel"]], dim=1)

                data = self.model(batch)
                
                data["target"] = {"H":batch["H"],
                "W":batch["W"],
                "FovX":batch["FovX"],
                "FovY":batch["FovY"],
                'world_view_transform': batch["world_view_transform"],
                'full_proj_transform': batch["full_proj_transform"],
                'camera_center': batch["camera_center"]}
                
                imgL, depthL, maskL = data["lview"]["img"], data["lview"]["depth"], data["lview"]["mask"]
                imgR, depthR, maskR = data["rview"]["img"], data["rview"]["depth"], data["rview"]["mask"]

                data["lview"]["pts"] = depth2pc(data["lview"]["depth"], torch.inverse(batch["lpose"]), batch["intrinsic"])
                data["rview"]["pts"] = depth2pc(data["rview"]["depth"], torch.inverse(batch["rpose"]), batch["intrinsic"])
                
                pred = pts2render(data, [0.,0.,0.])[:,0:1]
                loss = F.l1_loss(pred, gt["cim"])

                imgloss = torch.mean((imgL - gt["lim"])**2) + torch.mean((imgR - gt["rim"])**2)
                depthloss = F.l1_loss(depthL, gt["ldepth"]) + F.l1_loss(depthR, gt["rdepth"])
                maskloss = F.binary_cross_entropy(maskL.reshape(-1, 1), gt["lmask"].reshape(-1, 1).float()) + \
                    F.binary_cross_entropy(maskR.reshape(-1, 1), gt["rmask"].reshape(-1, 1).float())
                loss = loss + 0.33*imgloss + 0.33*depthloss + 0.33*maskloss
                # msk = torch.ones_like(gt).to(bool)
                metrics = {
                    "l1loss" : loss.item()
                }

                if self.total_steps and self.total_steps % cfg.record.loss_freq == 0:
                    self.logger.writer.add_scalar(f'lr', self.optimizer.param_groups[0]['lr'], self.total_steps)
                    print(f"{cfg.exp_name} epoch {e} step {self.total_steps} L1loss {loss.item()} lr {self.optimizer.param_groups[0]['lr']}")
                self.logger.push(metrics)

                if self.total_steps and self.total_steps % cfg.record.save_freq == 0:
                    self.save_ckpt(save_path=Path('%s/%d_%d.pth' % (cfg.record.ckpt_path, e, self.total_steps)), show_log=False)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                if self.total_steps and self.total_steps % cfg.record.eval_freq == 0:
                    self.model.eval()
                    self.run_eval()
                    self.model.train()

                self.total_steps += 1
                
        print("FINISHED TRAINING")
        self.logger.close()
        self.save_ckpt(save_path=Path('%s/%s_final.pth' % (cfg.record.ckpt_path, cfg.exp_name)))

    def to_cuda(self, batch):
        for k in batch:
            if isinstance(batch[k], tuple) or isinstance(batch[k], list):
                batch[k] = [b.to(self.device) for b in batch[k]]
            elif isinstance(batch[k], dict):
                batch[k] = {key: self.to_cuda(batch[k][key]) for key in batch[k]}
            else:
                batch[k] = batch[k].to(self.device)
        return batch
    
    def run_eval(self):
        print(f"Doing validation ...")
        torch.cuda.empty_cache()

        l1_list = []

        show_idx = [np.random.choice(list(range(self.len_val)), 1)]
        # show_idx = 0
        # show_idx = list(range(self.len_val))
        for idx, batch in enumerate(self.val_loader):
            with torch.no_grad():
                batch = self.to_cuda(batch)
                gt = batch["cim"]

                batch["left_event_tensor"] = torch.cat([batch["leframe"], batch["left_voxel"]], dim=1)
                batch["right_event_tensor"] = torch.cat([batch["reframe"], batch["right_voxel"]], dim=1)

                data = self.model(batch)
                
                data["target"] = {"H":batch["H"],
                "W":batch["W"],
                "FovX":batch["FovX"],
                "FovY":batch["FovY"],
                'world_view_transform': batch["world_view_transform"],
                'full_proj_transform': batch["full_proj_transform"],
                'camera_center': batch["camera_center"]}
                
                data["lview"]["pts"] = depth2pc(data["lview"]["depth"], torch.inverse(batch["lpose"]), batch["intrinsic"])
                data["rview"]["pts"] = depth2pc(data["rview"]["depth"], torch.inverse(batch["rpose"]), batch["intrinsic"])
                
                pred = pts2render(data, [0.,0.,0.])[:,0]
                loss = F.l1_loss(pred.squeeze(), gt.squeeze())
                l1_list.append(loss.item())

                # if idx == show_idx:
                if idx in show_idx:
                    print("show idx is ", idx)
                    tmp_gt = (gt[0]*255.0).cpu().numpy().astype(np.uint8).squeeze()
                    tmp_pred = (pred[0]*255.0).cpu().numpy().astype(np.uint8).squeeze()
                    # tmp_gt = tmp_pred
                    tmp_img_name = '%s/step%s_idx%d.jpg' % (cfg.record.show_path, self.total_steps, idx)       
                    imageio.imsave(tmp_img_name, np.concatenate([tmp_pred, tmp_gt], axis=0))
                   
        val_l1 = np.round(np.mean(np.array(l1_list)), 4)
        print(f"Validation Metrics ({self.total_steps}):, L1 {val_l1}")
        self.logger.write_dict({'val_l1': val_l1}, write_step=self.total_steps)
        torch.cuda.empty_cache()

    def save_ckpt(self, save_path, show_log=True):
        if show_log:
            print(f"Save checkpoint to {save_path} ...")
        torch.save({
            'total_steps': self.total_steps,
            'network': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }, save_path)
        
    def load_ckpt(self, load_path, load_optimizer=True, strict=True):
        assert os.path.exists(load_path)
        print(f"Loading checkpoint from {load_path} ...")
        ckpt = torch.load(load_path, map_location='cuda')
        self.model.load_state_dict(ckpt['network'], strict=strict)
        print(f"Parameter loading done")
        if load_optimizer:
            self.total_steps = ckpt['total_steps'] + 1
            self.logger.total_steps = self.total_steps
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.scheduler.load_state_dict(ckpt['scheduler'])
            print(f"Optimizer loading done")


if __name__ == "__main__":
    # L = torch.randn((1,3,640,480)).cuda()
    # R = torch.randn((1,3,640,480)).cuda()
    # # net = Net(int(5)).cuda()
    # net = ASNet_light(int(5)).cuda()
    # out = net(L,R)
    # print(len(out))
    # Input = torch.randn((1, 5, 640, 480)).cuda()
    # fnet = FireNet({"num_bins":5}).cuda()
    # out = fnet(Input, None)
    # print(out[0].shape)

    trainer = Trainer()
    trainer.train()



