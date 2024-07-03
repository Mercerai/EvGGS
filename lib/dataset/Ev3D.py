import sys
sys.path.append("/home/jiaxu/jx/EvGGS/")
from natsort import natsorted
import open3d as o3d
import h5py
import os
import numpy as np
import torch
from .utils import events_to_voxel_grid
from torch.utils.data import Dataset, DataLoader
import cv2
import glob
from tqdm import tqdm
from torch.utils.data import ConcatDataset
from lib.renderer.rend_utils import getProjectionMatrix, getWorld2View2, focal2fov
from lib.config import cfg, args

def depth2pc_np_ours(depth, extrinsic, intrinsic, isdisparity=False):
    H, W = depth.shape
    x_ref, y_ref = np.meshgrid(np.arange(0, W), np.arange(0, H))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    
    xyz_ref = np.matmul(np.linalg.inv(intrinsic[:3, :3]),
                        np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth.reshape([-1]))
    xyz_world = np.matmul(np.linalg.inv(extrinsic), np.vstack((xyz_ref, np.ones_like(x_ref))))[:3]
    
    return xyz_world.transpose((1, 0)).astype(np.float32)

def find_files(dir, exts):
    if os.path.isdir(dir):
        files_grabbed = []
        for ext in exts:
            files_grabbed.extend(glob.glob(os.path.join(dir, ext)))
        if len(files_grabbed) > 0:
            files_grabbed = sorted(files_grabbed)
        return files_grabbed
    else:
        return []

def parse_txt(filename, shape):
        assert os.path.isfile(filename)
        nums = open(filename).read().split()
        return np.array([float(x) for x in nums]).reshape(shape).astype(np.float32)

def concatenate_datasets_ratio(base_folders, dataset_type, split, dataset_kwargs={}):
    scene_lists = natsorted(os.listdir(os.path.join(base_folders, 'Event')))
    n_scenes = len(scene_lists)
    ratio = int(cfg.dataset.ratio * n_scenes)
    
    if split == "train":
        scene_lists = scene_lists[:ratio]
    elif split == "val":
        scene_lists = scene_lists[ratio:]
    
    dataset_list = []
    for i in range(len(scene_lists)):
        dataset_list.append(dataset_type(base_folders, scene_lists[i], **dataset_kwargs))
    return ConcatDataset(dataset_list)

def concatenate_datasets_split(base_folders, dataset_type, split, dataset_kwargs={}):
    if split == "train":
        scenes_path = os.path.join(base_folders, "train_scenes.txt")
    elif split == "test":
        scenes_path = os.path.join(base_folders, "test_scenes.txt")
    elif split == "val":
        scenes_path = os.path.join(base_folders, "val_scenes.txt")

    with open(scenes_path, 'r', encoding='utf-8') as file:
        scene_lists = [line.strip() for line in file.readlines()]
        
    dataset_list = []
    for i in range(len(scene_lists)):
        dataset_list.append(dataset_type(base_folders, scene_lists[i], **dataset_kwargs))
    return ConcatDataset(dataset_list)

T = np.array([[1,0,0,0],
            [0,-1,0,0],
            [0,0,-1,0],
            [0,0,0,1]])

class EventDataloader(DataLoader):
    def __init__(self, base_folders, split, num_workers, batch_size, shuffle=True):
        dataset = concatenate_datasets_split(base_folders, ReadEventFromH5, split=split)
        super().__init__(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=shuffle)

cs = cfg.cs

class ReadEventFromH5(Dataset):
    def __init__(self, base_folder, scene, polarity_offset=0):
        self.base_folder = base_folder
        self.scene = scene
        self.polarity_offset = polarity_offset
        self.H, self.W = 480, 640
        self.cropped_H, self.cropped_W = cs[1]-cs[0], cs[3] - cs[2]
        self.event_slices()

    def event_slices(self):
        ## load .h5 event files and generate event frames and voxels
        self.event_files_path = os.path.join(self.base_folder, "Event", self.scene)
        scene_files_path = os.path.join(self.base_folder, "Scenes", self.scene)
        self.pose_files = find_files('{}/Poses'.format(self.base_folder), exts=['*.txt'])
        self.num_views = len(self.pose_files)
        intrinsic_files = find_files('{}/Intrinsics'.format(self.base_folder), exts=['*.txt'])[:self.num_views]
        self.npz_files = find_files('{}'.format(scene_files_path), exts=["*.npz"])[:self.num_views]
        self.rgb_files = find_files('{}'.format(scene_files_path), exts=['*.png'])[:self.num_views]

        self.intrinsics = parse_txt(intrinsic_files[0], (4, 4))

    def __len__(self):
        return len(self.pose_files)
    
    def events_to_voxel(self, events):
        # generate a voxel grid from input events using temporal bilinear interpolation.
        x, y, t, p =  events
        x = x.astype(np.int32)
        y = y.astype(np.int32)
        p = p.astype(np.int32)
        mask_pos = p.copy()
        mask_neg = p.copy()
        mask_pos[p < 0] = 0
        mask_neg[p > 0] = 0
        frame1 = self.events_to_image(x, y, p * mask_pos)
        frame2 = self.events_to_image(x, y, p * mask_neg)
        frame3 = frame1 - frame2
        # cv2.imwrite('1.png', 128+frame1)
        # cv2.imwrite('2.png', 128-frame2)
        # cv2.imwrite('3.png', 128+frame3)
        return np.stack(((128+frame1)/255, (128-frame2)/255, (128 + frame3)/255), axis=2)
    
    def events_to_image(self, xs, ys, ps):
        # accumulate events into an image.
        img = np.zeros((self.H, self.W))
        np.add.at(img, (ys, xs), ps)
  
        # img = np.clip(img, -5, 5)
        # print(img)
        return img 

    def find_depth(self, npz_files, idx):
        npz = np.load(npz_files[idx], allow_pickle=True)
        depth = npz['depth_map']
        depth = self.prepare_depth(depth)
        return depth
    
    def find_pose(self, npz_files, idx):
        npz = np.load(npz_files[idx], allow_pickle=True)
        poses = npz['object_poses']
        for obj in poses:
            obj_name = obj['name']
            obj_mat = obj['pose']
            if obj_name == 'Camera':
                pose = obj_mat.astype(np.float32)
                break
        return pose @ T

    def prepare_depth(self, depth):
        # adjust depth maps generated by vision blender
        INVALID_DEPTH = -1
        depth[depth == INVALID_DEPTH] = 0

        return depth
        
    def accumulate_events_edited(self, events):
        x, y, t, p = events

    def events_to_frame(self, events):
        # generate a voxel grid from input events using temporal bilinear interpolation.
        x, y, t, p =  events
        x = x.astype(np.int32)
        y = y.astype(np.int32)
        p = p.astype(np.int32)
        mask_pos = p.copy()
        mask_neg = p.copy()
        mask_pos[p < 0] = 0
        mask_neg[p > 0] = 0
        frame1 = self.events_to_image(x, y, p * mask_pos)
        frame2 = self.events_to_image(x, y, p * mask_neg)
        frame3 = frame1 - frame2
 
        return np.stack(((128 + frame1)/255, (128-frame2)/255, (128 + frame3)/255), axis=2)
    
    def events_to_image(self, xs, ys, ps):
        # accumulate events into an image.
        img = np.zeros((self.H, self.W))
        np.add.at(img, (ys, xs), ps)
        # print(img.max(), img.min())
        # img = np.clip(img, -5, 5)
        # print(img)
        return img
        
    def accumulate_events(self, events, resolution_level=1, polarity_offset=0):
        x, y, t, p = events
        acc_frm = np.zeros((self.H, self.W))
        np.add.at(acc_frm, (y // resolution_level, x // resolution_level), p + polarity_offset)
        return acc_frm

    def __getitem__(self, idx):
        index = str(idx).zfill(4)
        left_event1 = np.load(os.path.join(self.event_files_path, '{}.npy'.format(index)))
        
        # left_event_voxel = events_to_voxel_grid(left_event.transpose((1,0)), cfg.model.num_bins, self.W, self.H)
        # left_pose = parse_txt(self.pose_files[idx], (4,4))
        
        left_pose = self.find_pose(self.npz_files, idx)
        left_depth_gt = self.find_depth(self.npz_files, idx)
        left_mask = (left_depth_gt > 0)
        left_img = cv2.cvtColor(cv2.imread(self.rgb_files[idx])[...,:3] * left_mask[..., np.newaxis], cv2.COLOR_BGR2GRAY) / 255.

        if idx + 1 < len(self.pose_files):
            left_event2 = np.load(os.path.join(self.event_files_path, '{}.npy'.format(str(idx+1).zfill(4))))
            center_depth_gt = self.find_depth(self.npz_files, idx+1)
            center_mask = (center_depth_gt > 0)
            # center_pose = parse_txt(self.pose_files[idx+1], (4,4))
            center_pose = self.find_pose(self.npz_files, idx+1)
            
            # try:
            int_img = cv2.cvtColor(cv2.imread(self.rgb_files[idx+1])[...,:3] * center_mask[..., np.newaxis], cv2.COLOR_BGR2GRAY) / 255.
            
        else:
            left_event2 = np.load(os.path.join(self.event_files_path, '{}.npy'.format(str(0).zfill(4))))
            center_depth_gt = self.find_depth(self.npz_files, 0)
            center_mask = (center_depth_gt > 0)
            # center_pose = parse_txt(self.pose_files[0], (4,4))
            center_pose = self.find_pose(self.npz_files, 0)
            int_img = cv2.cvtColor(cv2.imread(self.rgb_files[0])[...,:3] * center_mask[..., np.newaxis], cv2.COLOR_BGR2GRAY) /255.
        
        center_extrinsics = np.linalg.inv(center_pose)
    
        left_event_frame = self.events_to_frame(np.hstack((left_event1, left_event2)))
        left_event_voxel = events_to_voxel_grid(np.hstack((left_event1, left_event2)).transpose((1,0)), cfg.model.num_bins, self.W, self.H)
        if idx + 2 < len(self.pose_files):
            r_id = idx + 2
            r_index = str(r_id).zfill(4)
            right_event1 = np.load(os.path.join(self.event_files_path, '{}.npy'.format(r_index)))
            # right_pose = parse_txt(self.pose_files[r_id], (4,4))
            right_pose = self.find_pose(self.npz_files, r_id)
            right_depth_gt = self.find_depth(self.npz_files, r_id)
        else:
            r_id = (idx + 2) % len(self.pose_files)
            r_index = str(r_id).zfill(4)
            right_event1 = np.load(os.path.join(self.event_files_path, '{}.npy'.format(r_index)))
            # right_pose = parse_txt(self.pose_files[r_id], (4,4))
            right_pose = self.find_pose(self.npz_files, r_id)
            right_depth_gt = self.find_depth(self.npz_files, r_id)

        if idx + 3 < len(self.pose_files):
            r_id2 = idx + 3
            r_index2 = str(r_id2).zfill(4)
            right_event2 = np.load(os.path.join(self.event_files_path,'{}.npy'.format(r_index2)))
        else:
            r_id2 = (idx + 3) % len(self.pose_files)
            r_index2 = str(r_id2).zfill(4)
            right_event2 = np.load(os.path.join(self.event_files_path, '{}.npy'.format(r_index2)))


        # pr = depth2pc_np_ours(right_depth_gt, np.linalg.inv(right_pose), self.intrinsics)
        # pl = depth2pc_np_ours(left_depth_gt, np.linalg.inv(left_pose), self.intrinsics)
        # pc = np.concatenate([pr, pl], axis=0)
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(pc)
        # o3d.io.write_point_cloud("pts.ply", pcd)
        # print(np.hstack((right_event1, right_event2)).shape)
        right_event_frame = self.events_to_frame(np.hstack((right_event1, right_event2)))
        right_event_voxel = events_to_voxel_grid(np.hstack((right_event1, right_event2)).transpose((1,0)), cfg.model.num_bins, self.W, self.H)
        # right_event_voxel = events_to_voxel_grid(right_event.transpose((1,0)), cfg.model.num_bins, self.W, self.H)
        
        right_mask = (right_depth_gt > 0)
        right_img = cv2.cvtColor(cv2.imread(self.rgb_files[r_id])[...,:3] * right_mask[..., np.newaxis], cv2.COLOR_BGR2GRAY) / 255.

        center_event = np.hstack((left_event2, right_event1))
        center_event_frame = self.events_to_frame(center_event)
        center_event_voxel = events_to_voxel_grid(center_event.transpose((1,0)), cfg.model.num_bins, self.W, self.H)

        intrinsic = self.intrinsics  

        intrinsic[0,2] = (self.cropped_W - 1) / 2
        intrinsic[1,2] = (self.cropped_H - 1) / 2

        projection_matrix = getProjectionMatrix(znear=0.01, zfar=0.99, K=intrinsic, h=self.cropped_H, w=self.cropped_W).transpose(0, 1)
        world_view_transform = torch.tensor(getWorld2View2(center_extrinsics[:3,:3].reshape(3, 3).transpose(1, 0)\
            , center_extrinsics[:3, 3])).transpose(0, 1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]

        item = {
                'cim': int_img.astype(np.float32)[cs[0]:cs[1], cs[2]:cs[3]][np.newaxis],  #[1, H, W]
                'lim': left_img.astype(np.float32)[cs[0]:cs[1], cs[2]:cs[3]][np.newaxis],
                'rim': right_img.astype(np.float32)[cs[0]:cs[1], cs[2]:cs[3]][np.newaxis],
                'leframe': left_event_frame.transpose((2,0,1)).astype(np.float32)[:, cs[0]:cs[1], cs[2]:cs[3]],  #[3, H, W]
                'reframe': right_event_frame.transpose((2,0,1)).astype(np.float32)[:, cs[0]:cs[1], cs[2]:cs[3]], 
                'ceframe': center_event_frame.transpose((2,0,1)).astype(np.float32)[:, cs[0]:cs[1], cs[2]:cs[3]],
                'lmask': left_mask[cs[0]:cs[1], cs[2]:cs[3]],  #[H, W]
                'rmask': right_mask[cs[0]:cs[1], cs[2]:cs[3]],
                'cmask': center_mask[cs[0]:cs[1], cs[2]:cs[3]],
                'lpose': left_pose.astype(np.float32), #[4, 4]
                'rpose': right_pose.astype(np.float32),
                'intrinsic': intrinsic.astype(np.float32), #[4, 4]
                'ldepth': left_depth_gt.astype(np.float32)[cs[0]:cs[1], cs[2]:cs[3]], # #[H, W]
                'rdepth': right_depth_gt.astype(np.float32)[cs[0]:cs[1], cs[2]:cs[3]],
                'cdepth': center_depth_gt.astype(np.float32)[cs[0]:cs[1], cs[2]:cs[3]],
                'center_voxel':center_event_voxel[:, cs[0]:cs[1], cs[2]:cs[3]], #[5, H, W]
                'right_voxel':right_event_voxel[:, cs[0]:cs[1], cs[2]:cs[3]],
                'left_voxel':left_event_voxel[:, cs[0]:cs[1], cs[2]:cs[3]],
            ### target view rendering parameters ###
                "H":self.cropped_H,
                "W":self.cropped_W,
                "FovX":focal2fov(intrinsic[0, 0], self.cropped_W),
                "FovY":focal2fov(intrinsic[1, 1], self.cropped_H),
                'world_view_transform': world_view_transform,  #[4, 4]
                'full_proj_transform': full_proj_transform,   #[4, 4]
                'camera_center': camera_center  #[3]
                }
        return item
    




# if __name__ == "__main__":
#     dataset = ReadEventFromH5(r"/home/lsf_storage/dataset/EV3D5/", "AK47",0)
#     dataset[0]
#     # dataloader = EventDataloader(r"/home/lsf_storage/dataset/EV3D/", split="full", batch_size=1, num_workers=1, shuffle=False)
#     # print(len(dataloader))
    
#     # for idx, batch in enumerate(dataloader):
#     #     print(batch["ldepth"].shape)
#     #     break