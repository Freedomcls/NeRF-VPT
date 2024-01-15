import os, sys
import glob
import numpy as np
from skimage.io import imread
from torch.utils.data import Dataset
import cv2
import imageio
from imgviz import label_colormap
import numpy as np
import torch
import math
import torchvision.transforms as transforms
from datasets import augmentations
import random
from PIL import Image

def create_rays(num_rays, Ts_c2w, height, width, fx, fy, cx, cy, near, far, c2w_staticcam=None, depth_type="z",
            use_viewdirs=True, convention="opencv"):
    """
    convention: 
    "opencv" or "opengl". It defines the coordinates convention of rays from cameras.
    OpenCv defines x,y,z as right, down, forward while OpenGl defines x,y,z as right, up, backward (camera looking towards forward direction still, -z!)
    Note: Use either convention is fine, but the corresponding pose should follow the same convention.

    """
    # print('prepare rays')

    rays_cam = get_rays_camera(num_rays, height, width, fx, fy, cx, cy, depth_type=depth_type, convention=convention) # [N, H, W, 3]

    dirs_C = rays_cam.view(num_rays, -1, 3)  # [N, HW, 3]
    rays_o, rays_d = get_rays_world(Ts_c2w, dirs_C)  # origins: [B, HW, 3], dirs_W: [B, HW, 3]

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # c2w_staticcam: If not None, use this transformation matrix for camera,
            # while using other c2w argument for viewing directions.
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays_world(c2w_staticcam, dirs_C)  # origins: [B, HW, 3], dirs_W: [B, HW, 3]

        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True).float()

    near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)

    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)
    return rays

def get_rays_camera(B, H, W, fx, fy,  cx, cy, depth_type, convention="opencv"):

    assert depth_type == "z" or depth_type == "euclidean"
    i, j = torch.meshgrid(torch.arange(W), torch.arange(H))  # pytorch's meshgrid has indexing='ij', we transpose to "xy" moode

    i = i.t().float()
    j = j.t().float()

    size = [B, H, W]

    i_batch = torch.empty(size)
    j_batch = torch.empty(size)
    i_batch[:, :, :] = i[None, :, :]
    j_batch[:, :, :] = j[None, :, :]

    if convention == "opencv":
        x = (i_batch - cx) / fx
        y = (j_batch - cy) / fy
        z = torch.ones(size)
    elif convention == "opengl":
        x = (i_batch - cx) / fx
        y = -(j_batch - cy) / fy
        z = -torch.ones(size)
    else:
        assert False

    dirs = torch.stack((x, y, z), dim=3)  # shape of [B, H, W, 3]

    if depth_type == 'euclidean':
        norm = torch.norm(dirs, dim=3, keepdim=True)
        dirs = dirs * (1. / norm)

    return dirs

def get_rays_world(T_WC, dirs_C):
    R_WC = T_WC[:, :3, :3]  # Bx3x3
    dirs_W = torch.matmul(R_WC[:, None, ...], dirs_C[..., None]).squeeze(-1)
    origins = T_WC[:, :3, -1]  # Bx3
    origins = torch.broadcast_tensors(origins[:, None, :], dirs_W)[0]
    return origins, dirs_W

class ReplicaDatasetCache(Dataset):
    def __init__(self, root_dir, split='train', img_wh=(800, 800), is_crop=False, render='test', prompt_path='', if_vpt=False):

        data_dir = root_dir
        traj_file = os.path.join(data_dir, "traj_w_c.txt")
        self.rgb_dir = os.path.join(data_dir, "rgb")
        self.depth_dir = os.path.join(data_dir, "depth")  # depth is in mm uint
        self.semantic_class_dir = os.path.join(data_dir, "semantic_class")
        self.semantic_instance_dir = os.path.join(data_dir, "semantic_instance")
        if not os.path.exists(self.semantic_instance_dir):
            self.semantic_instance_dir = None

        total_num = 900
        
        self.split = split
        self.render = render
        self.if_vpt = if_vpt
        self.prompt_path = prompt_path
        
        step = 5 # 180 view
        
        train_ids = list(range(0, total_num, step))
        val_ids = [x+step//2 for x in train_ids]  
        if split == 'test' and self.render == 'train':
            # print(11111)
            val_ids = list(range(0, total_num, step))
        elif split == 'test' and self.render == 'test':
            ids = list(range(0, total_num, step))
            val_ids = [x+3 for x in ids] 

        self.train_ids = train_ids
        self.train_num = len(train_ids)
        self.val_ids = val_ids
        self.val_num = len(val_ids)

        self.img_w, self.img_h = img_wh
        self.set_params_replica()
        self.use_viewdir = False
        self.convention = "opencv"
        self.white_back = False
        self.split = split
        
        self.source_transform = transforms.Compose([
            # transforms.Resize((224, 224), interpolation=Image.NEAREST),
            augmentations.ToOneHot(99)])

        self.Ts_full = np.loadtxt(traj_file, delimiter=" ").reshape(-1, 4, 4)

        self.rgb_list = sorted(glob.glob(self.rgb_dir + '/rgb*.png'), key=lambda file_name: int(file_name.split("_")[-1][:-4]))
        self.depth_list = sorted(glob.glob(self.depth_dir + '/depth*.png'), key=lambda file_name: int(file_name.split("_")[-1][:-4]))
        self.semantic_list = sorted(glob.glob(self.semantic_class_dir + '/semantic_class_*.png'), key=lambda file_name: int(file_name.split("_")[-1][:-4]))
        
        if self.if_vpt is True:
            # print(11111)
            rgb_list = sorted(glob.glob(self.prompt_path + '_train/*.png'), key=lambda file_name: int(file_name.split("_")[-1][:-4]))
            rgb_lis = sorted(glob.glob(self.prompt_path + '_val/*.png'), key=lambda file_name: int(file_name.split("_")[-1][:-4]))
            if self.split == 'test':
                rgb_lis = sorted(glob.glob(self.prompt_path + '_' + self.render + '/*.png'), key=lambda file_name: int(file_name.split("_")[-1][:-4]))
        
        if self.semantic_instance_dir is not None:
            self.instance_list = sorted(glob.glob(self.semantic_instance_dir + '/semantic_instance_*.png'), key=lambda file_name: int(file_name.split("_")[-1][:-4]))

        self.train_samples = {'image': [], 'depth': [], 'nerf': [],
                          'semantic': [], 'T_wc': [],
                          'instance': []}

        self.test_samples = {'image': [], 'depth': [], 'nerf': [],
                          'semantic': [], 'T_wc': [],
                          'instance': []}

        if split == 'train':
        # training samples
            self.poses = []
            self.image_paths = []
            m=0
            for idx in train_ids:
                image = cv2.imread(self.rgb_list[idx])[:,:,::-1] / 255.0  # change from BGR uinit 8 to RGB float
                depth = cv2.imread(self.depth_list[idx], cv2.IMREAD_UNCHANGED) / 1000.0  # uint16 mm depth, then turn depth from mm to meter
                if self.if_vpt is True:
                    semantic = cv2.imread(rgb_list[m])[:,:,::-1] / 255.0
                    # print(rgb_list[m], self.rgb_list[idx])
                # semantic = cv2.imread(rgb_list[idx])[:,:,::-1] / 255.0
                m+=1
                # semantic = cv2.imread(self.rgb_list[idx])[:,:,::-1] / 255.0
                # semantic = cv2.cvtColor(semantic, cv2.COLOR_BGR2GRAY)

                if self.semantic_instance_dir is not None:
                    instance = cv2.imread(self.instance_list[idx], cv2.IMREAD_UNCHANGED) # uint16

                if (self.img_h is not None and self.img_h != image.shape[0]) or \
                        (self.img_w is not None and self.img_w != image.shape[1]):
                    # print('imageshape',self.img_h,self.img_w)
                    image = cv2.resize(image, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
                    depth = cv2.resize(depth, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
                    if self.if_vpt is True:
                        semantic = cv2.resize(semantic, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
                    # semantic = cv2.resize(cv2.imread(self.rgb_list[idx])[:,:,::-1] / 255.0, (480, 640), interpolation=cv2.INTER_LINEAR)
                    # semantic = semantic.convert('L')
                    # semantic = self.source_transform(semantic)
                    
                    # semantic = cv2.resize(semantic, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)
                    if self.semantic_instance_dir is not None:
                        instance = cv2.resize(instance, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)

                T_wc = self.Ts_full[idx]

                self.train_samples["image"].append(image)
                self.train_samples["depth"].append(depth)
                if self.if_vpt is True:
                    self.train_samples["semantic"].append(semantic)
                # self.train_samples["semantic"].append(image)
                if self.semantic_instance_dir is not None:
                    self.train_samples["instance"].append(instance)
                self.train_samples["T_wc"].append(T_wc)
                self.poses.append(T_wc[:3,:4])
                if idx % 9 == 0:
                    self.image_paths.append(self.rgb_list[idx])
            for key in self.train_samples.keys():  # transform list of np array to array with batch dimension
                self.train_samples[key] = np.asarray(self.train_samples[key])
            self.read_meta_train()

            print()
            print("Training Sample Summary:")
            for key in self.train_samples.keys(): 
                print("{} has shape of {}, type {}.".format(key, self.train_samples[key].shape, self.train_samples[key].dtype))
        else:
            # test samples
            i=0
            # self.index=train_ids
            self.index=val_ids
            for idx in val_ids:
                # j=random.choice(test_ids)
                image = cv2.imread(self.rgb_list[idx])[:,:,::-1] / 255.0  # change from BGR uinit 8 to RGB float
                depth = cv2.imread(self.depth_list[idx], cv2.IMREAD_UNCHANGED) / 1000.0  # uint16 mm depth, then turn depth from mm to meter
                # print(idx)
                # print(rgb_list)
                # print(self.rgb_list)
                # semantic = cv2.imread(rgb_lis[idx])[:,:,::-1] / 255.0
                if self.if_vpt is True:
                    semantic = cv2.imread(rgb_lis[i])[:,:,::-1] / 255.0
                # semantic = cv2.imread(self.rgb_list[idx])[:,:,::-1] / 255.0
                i=i+1

                
                if self.semantic_instance_dir is not None:
                    instance = cv2.imread(self.instance_list[idx], cv2.IMREAD_UNCHANGED) # uint16

                if (self.img_h is not None and self.img_h != image.shape[0]) or \
                        (self.img_w is not None and self.img_w != image.shape[1]):
                    image = cv2.resize(image, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
                    depth = cv2.resize(depth, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
                    if self.if_vpt is True:
                        semantic = cv2.resize(semantic, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
                    # semantic = cv2.resize(cv2.imread(self.rgb_list[idx])[:,:,::-1] / 255.0, (480, 640), interpolation=cv2.INTER_LINEAR)
                    # semantic = self.source_transform(semantic)
                    
                    # semantic = cv2.resize(semantic, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)
                    if self.semantic_instance_dir is not None:
                        instance = cv2.resize(instance, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)
                T_wc = self.Ts_full[idx]

                self.test_samples["image"].append(image)
                self.test_samples["depth"].append(depth)
                if self.if_vpt is True:
                    self.test_samples["semantic"].append(semantic)
                # self.test_samples["semantic"].append(image)
                if self.semantic_instance_dir is not None:
                    self.test_samples["instance"].append(instance)
                self.test_samples["T_wc"].append(T_wc)

            for key in self.test_samples.keys():  # transform list of np array to array with batch dimension
                self.test_samples[key] = np.asarray(self.test_samples[key])
            self.read_meta_test()

            # print()
            # print("Testing Sample Summary:")
            # for key in self.test_samples.keys(): 
            #     print("{} has shape of {}, type {}.".format(key, self.test_samples[key].shape, self.test_samples[key].dtype))

    def set_params_replica(self):
        self.H = self.img_h
        self.W = self.img_w

        self.n_pix = self.H * self.W
        self.aspect_ratio = self.W/self.H

        self.hfov = 90
        # the pin-hole camera has the same value for fx and fy
        self.fx = self.W / 2.0 / math.tan(math.radians(self.hfov / 2.0))
        # self.fy = self.H / 2.0 / math.tan(math.radians(self.yhov / 2.0))
        self.fy = self.fx
        self.focal = self.fx
        self.cx = (self.W - 1.0) / 2.0
        self.cy = (self.H - 1.0) / 2.0
        self.near, self.far = 0.1, 10.0
        self.bounds = np.array([self.near, self.far])
        self.c2w_staticcam = None

    def read_meta_train(self):
        self.train_Ts = torch.from_numpy(self.train_samples["T_wc"]).float()
        self.train_image = torch.from_numpy(self.train_samples["image"]).float().contiguous()
        if self.if_vpt is True:
            self.train_semantic = torch.from_numpy(self.train_samples["semantic"]).float().contiguous()
        self.train_depth = torch.from_numpy(self.train_samples["depth"]).float().contiguous()
        # self.train_semantic = torch.nn.functional.one_hot(self.train_semantic,99)
        
        self.all_rgbs = self.train_image.reshape(-1, self.train_image.shape[-1]) # [num_train*H*W, 8]
        if self.if_vpt is True:
            self.all_semantics = self.train_semantic.reshape(-1, self.train_semantic.shape[-1]) # [num_train*H*W, 99]
        # self.all_semantic = self.train_semantic.reshape(-1,self.train_semantic.shape[-1])
        self.all_rays = create_rays(self.train_num, self.train_Ts, self.H, self.W, self.fx, self.fy, self.cx, self.cy,
                        self.near, self.far, use_viewdirs=self.use_viewdir, convention=self.convention)
        num_img, num_ray, ray_dim = self.all_rays.shape
        # print("train_semantic", self.train_semantic.shape)
        print("all_ray", self.all_rays.shape)
        self.all_rays = self.all_rays.reshape(num_img*num_ray,ray_dim)

    def read_meta_test(self):
        self.test_Ts = torch.from_numpy(self.test_samples["T_wc"]).float()  # [num_test, 4, 4]
        self.test_image = torch.from_numpy(self.test_samples["image"]).float().contiguous()  # [num_test, H, W, 3]
        if self.if_vpt is True:
            self.test_semantic = torch.from_numpy(self.test_samples["semantic"]).float().contiguous()  # [num_test, H, W, 3]
        self.test_depth = torch.from_numpy(self.test_samples["depth"]).float().contiguous()  # [num_test, H, W, 3]
        # self.test_semantic = torch.from_numpy(self.train_samples["semantic"]).float().contiguous()  # [num_test, H, W, 3]
        # print("test_depth", self.test_depth.shape)
        self.test_depth = self.test_depth.unsqueeze(-1)
        # print("test_semantic", self.test_semantic.shape)
        # self.all_rays = create_rays(self.train_num, self.test_Ts, self.H, self.W, self.fx, self.fy,
        self.all_rays = create_rays(self.val_num, self.test_Ts, self.H, self.W, self.fx, self.fy,
                                self.cx, self.cy, self.near, self.far, use_viewdirs=self.use_viewdir, convention=self.convention)
        # print("test_semantic", self.test_semantic.shape)

        num_img, num_ray, ray_dim = self.all_rays.shape
        self.all_rgbs = self.test_image.reshape(num_img, num_ray, self.test_image.shape[-1]) # [num_test, H*W, 3]
        self.all_depths = self.test_depth.reshape(num_img, num_ray, self.test_depth.shape[-1]) # [num_test, H*W, 3]

        # self.all_semantics = self.test_semantic.reshape(num_img, 640*480, 3) # [num_test, H*W, 3]
        if self.if_vpt is True:
            self.all_semantics = self.test_semantic.reshape(num_img, num_ray, 3) # [num_test, H*W, 3]

    def __len__(self):
        return self.all_rays.shape[0]
            
    def __getitem__(self, idx):
        if self.split == 'train':
            # sample = {'rays':self.all_rays[idx], 'rgbs':self.all_rgbs[idx], 'semantic':self.all_semantics[idx],'nerf':self.all_nerf[idx]}
            if self.if_vpt is True:
                sample = {'rays':self.all_rays[idx], 'rgbs':self.all_rgbs[idx], 'semantic':self.all_semantics[idx]}
            else:
                sample = {'rays':self.all_rays[idx], 'rgbs':self.all_rgbs[idx]}

        else:
            rays = self.all_rays[idx]
            rgbs = self.all_rgbs[idx]
            # print("dataloader_index", idx)
            if self.if_vpt is True:
                semantic = self.all_semantics[idx]

            if self.if_vpt is True:
                sample = {'rays':rays, 'rgbs':rgbs, "semantic":semantic, "index":self.index[idx]} #"index":self.index
            else:
                sample = {'rays':rays, 'rgbs':rgbs, "index":self.index[idx]} #"index":self.index
            # sample = {'rays':rays, 'rgbs':rgbs, "semantic":semantic,"index":self.index[idx], 'depth':depth} #"index":self.index
        return sample