from lib.config import cfg, args
from lib.dataset import EventDataloader
from lib.recorder import Logger, file_backup
from lib.network import model_loss_light, model_loss, EventGaussian
from lib.renderer import pts2render, depth2pc
from lib.utils import depth2img
from lib.losses import psnr, ssim
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
import lpips
import time

cs = cfg.cs

class Trainer:
    def __init__(self) -> None:
        device = torch.device('cuda:{}'.format(cfg.local_rank))
        self.device = device
        which_test = "val"
  
        self.train_loader =  None
        self.val_loader = EventDataloader(cfg.dataset.base_folder, split=which_test, num_workers=1,\
                                             batch_size=1, shuffle=False)

        self.len_val = len(self.val_loader)
        self.model = EventGaussian().to(self.device)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=cfg.wdecay, eps=1e-8)
        dpt_params = list(map(id,self.model.depth_estimator.parameters())) + list(map(id,self.model.intensity_estimator.parameters()))
        rest_params = filter(lambda x:id(x) not in dpt_params,self.model.parameters())
        self.optimizer = optim.Adam([
            {'params':self.model.depth_estimator.parameters(), 'lr':1},
            {'params':self.model.intensity_estimator.parameters(), 'lr':1},
            {'params':rest_params, 'lr':1},
        ], lr=0.001, weight_decay=cfg.wdecay, eps=1e-8)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10000, gamma=0.9)
        self.logger = Logger(self.scheduler, cfg.record)

        self.total_steps = 0
        self.target_epoch = cfg.target_epoch

        if cfg.restore_ckpt:
            self.load_ckpt(cfg.restore_ckpt)

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
        loss_fn_vgg = lpips.LPIPS(net='vgg').to(self.device)
        l1_list = []
        psnr_list = []
        ssim_list = []
        lpips_list = []
        show_idx = list(range(self.len_val))
        count = 0
        scene_num = 0
        os.makedirs(r'%s/%d/' % (cfg.record.show_path, scene_num), exist_ok=True)
        for idx, batch in enumerate(tqdm(self.val_loader)):
            if count == 201:
                scene_num += 1
                count = 0
                os.makedirs(r'%s/%d/' % (cfg.record.show_path, scene_num), exist_ok=True)
            with torch.no_grad():
                batch = self.to_cuda(batch)
                gt = batch["cim"]

                batch["left_event_tensor"] = torch.cat([batch["leframe"], batch["left_voxel"]], dim=1)
                batch["right_event_tensor"] = torch.cat([batch["reframe"], batch["right_voxel"]], dim=1)
                
                start_time = time.time()
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
                end_time = time.time()
                execution_time = end_time - start_time

                pred = pred[:,None]
                loss = F.l1_loss(pred.squeeze(), gt.squeeze())
                l1_list.append(loss.item())

                count += 1
                # if idx == show_idx:
                psnr_list.append(torch.mean(psnr(pred, gt)).item())
                ssim_list.append(ssim(pred, gt).item())
                lpips_list.append(torch.mean(loss_fn_vgg(pred*2-1, gt*2-1)).item())
                if idx in show_idx:
                    tmp_gt = (gt[0]*255.0).cpu().numpy().astype(np.uint8).squeeze()
                    tmp_pred = (pred[0]*255.0).cpu().numpy().astype(np.uint8).squeeze()
                    tmp_img_name = '%s/%d/step%s_idx%d.jpg' % (cfg.record.show_path, scene_num, self.total_steps, idx)       
                    imageio.imsave(tmp_img_name, np.concatenate([tmp_pred, tmp_gt], axis=0))
                   
        val_psnr = np.round(np.mean(np.array(psnr_list)), 8)
        val_ssim = np.round(np.mean(np.array(ssim_list)), 8)
        val_lpips = np.round(np.mean(np.array(lpips_list)), 8)
        print(f"Non masked and selected Metrics ({self.total_steps}):, psnr {val_psnr}, ssim {val_ssim}, lpips {val_lpips}")
        self.logger.write_dict({'NO masked psnr on val set': val_psnr}, write_step=self.total_steps)
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
    trainer = Trainer()
    trainer.load_ckpt(cfg.pretrain_ckpt, load_optimizer=False)
    trainer.model.eval()
    trainer.run_eval()



