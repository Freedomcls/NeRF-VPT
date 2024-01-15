import torch
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import imageio
from argparse import ArgumentParser

from models.rendering import render_rays, render_semantic_rays
from models.nerf import *
# from models.nerf_cls import NeRF_3D
# from models.pointnets import PointNetDenseCls
# from models.ConvNetWork import *

from utils import load_ckpt, color_cls
import metrics

from datasets import dataset_dict
from datasets.depth_utils import *
import cv2
import ast
torch.backends.cudnn.benchmark = True

DEBUG = ast.literal_eval(os.environ.get("DEBUG", "False"))

def get_opts():
    parser = ArgumentParser()
    parser.add_argument('--mode', default="normal",
                        type=str, choices=['d3', 'd3_ib', 'normal'],
                        help='use which system')
    parser.add_argument("--nerf_model", default="NeRF", help="nerf model type")
    parser.add_argument('--root_dir', type=str,
                        default='/home/ubuntu/data/nerf_example_data/nerf_synthetic/lego',
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='blender',
                        choices=['blender', 'blender_cls_ib' ,'llff', "llff_cls", "llff_cls_ib", "replica"],
                        help='which dataset to validate')
    parser.add_argument('-sn', '--semantic_network', type=str, default='pointnet',
                        choices=['pointnet', 'conv3d'], 
                        help='use which network to extract semantic features')
    parser.add_argument('--scene_name', type=str, default='test',
                        help='scene name, used as output folder name')
    parser.add_argument('--split', type=str, default='test',
                        help='test or train')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[800, 800],
                        help='resolution (img_w, img_h) of the image')
    parser.add_argument('--spheric_poses', default=False, action="store_true",
                        help='whether images are taken in spheric poses (for llff)')

    parser.add_argument('--N_samples', type=int, default=64,
                        help='number of coarse samples')
    parser.add_argument('--N_importance', type=int, default=128,
                        help='number of additional fine samples')
    parser.add_argument('--use_disp', default=False, action="store_true",
                        help='use disparity depth sampling')
    parser.add_argument('--chunk', type=int, default=32*1024*4,
                        help='chunk size to split the input to avoid OOM')

    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='pretrained checkpoint path to load')

    parser.add_argument('--render', type=str, default='test',
                        help='if use vpt')
    
    parser.add_argument('--if_vpt', action='store_true',
                        help='vpt render which dataset')
    
    parser.add_argument('--prompt_path', type=str, default='',
                        help='view prompt path')

    parser.add_argument('--save_depth', default=False, action="store_true",
                        help='whether to save depth prediction')
    parser.add_argument('--depth_format', type=str, default='pfm',
                        choices=['pfm', 'bytes'],
                        help='which format to save')

    return parser.parse_args()


@torch.no_grad()
def batched_inference(models, embeddings,
                      rays, N_samples, N_importance, use_disp,
                      chunk,
                      white_back):
    """Do batched inference on rays using chunk."""
    B = rays.shape[0]
    chunk = 1024*32
    results = defaultdict(list)
    for i in range(0, B, chunk):
        rendered_ray_chunks = \
            render_rays(models,
                        embeddings,
                        rays[i:i+chunk],
                        N_samples,
                        use_disp,
                        0,
                        0,
                        N_importance,
                        chunk,
                        dataset.white_back,
                        test_time=True)

        for k, v in rendered_ray_chunks.items():
            results[k] += [v]

    for k, v in results.items():
        results[k] = torch.cat(v, 0)
    return results

@torch.no_grad()
def batched_semantic_inference(models, embeddings,
                      rays, semantic, N_samples, N_importance, use_disp,
                      chunk,
                      white_back,
                      render_func,
                      **kwargs,
                      ):
    """Do batched inference on rays using chunk."""
    B = rays.shape[0]
    chunk = 1024*32 # hard code 
    results = defaultdict(list)
    for i in range(0, B, chunk):
        # print(rays[i:i+chunk].shape, B)
        rendered_ray_chunks = \
            render_func(models,
                        embeddings,
                        rays[i:i+chunk],
                        semantic[i:i+chunk],
                        N_samples,
                        use_disp,
                        0,
                        0,
                        N_importance,
                        chunk,
                        dataset.white_back,
                        test_time=True,
                        **kwargs)

        for k, v in rendered_ray_chunks.items():
            results[k] += [v]

    for k, v in results.items():
        results[k] = torch.cat(v, 0)
    return results

if __name__ == "__main__":
    args = get_opts()
    w, h = args.img_wh
    # _cls = 6 # hard code

    kwargs = {'root_dir': args.root_dir,
              'split': args.split,
              'if_vpt': args.if_vpt,
              'prompt_path': args.prompt_path,
              'render': args.render,
              'img_wh': tuple(args.img_wh)}
    # print(args.if_vpt, 'args.if_vpt')
    if 'llff' in args.dataset_name:
        kwargs['spheric_poses'] = args.spheric_poses
    dataset = dataset_dict[args.dataset_name](**kwargs)

    embedding_xyz = Embedding(3, 10)
    embedding_dir = Embedding(3, 4)
    # nerf_coarse = SemanticNeRF()
    if 'NeRFVPT' in args.nerf_model:
        nerf_coarse = NeRFVPT()
        nerf_fine = NeRFVPT()
    else:
        nerf_coarse = NeRF()
        nerf_fine = NeRF()

    load_ckpt(nerf_coarse, args.ckpt_path, model_name='nerf_coarse')
    load_ckpt(nerf_fine, args.ckpt_path, model_name='nerf_fine')
    # load_ckpt(points, args.ckpt_path, model_name='points')
    
    nerf_coarse.cuda().eval()
    nerf_fine.cuda().eval()

    models = [nerf_coarse, nerf_fine]
    embeddings = [embedding_xyz, embedding_dir]

    imgs = []
    psnrs = []
    ssims = []
    depths = []
    dir_name = f'results/{args.dataset_name}/{args.scene_name}'
    os.makedirs(dir_name, exist_ok=True)
    render_func = render_semantic_rays
    # torch.cuda.synchronize()

    # for i in range(0,5):
    print(len(dataset),"len(dataset)")
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        rays = sample['rays'].cuda()
        if args.dataset_name == 'replica':
            index = sample["index"]
        if 'NeRFVPT' in args.nerf_model:
            semantic = sample['semantic'].cuda()
        if 'NeRFVPT' in args.nerf_model:
            render_func = render_semantic_rays
            results = batched_semantic_inference(models, embeddings, rays, semantic,
                                    args.N_samples, args.N_importance, args.use_disp,
                                    args.chunk,
                                    dataset.white_back,
                                    render_func=render_func,
                                    # _cls_num=_cls,
                                    )
        else:
            results = batched_inference(models, embeddings, rays,
                                        args.N_samples, args.N_importance, args.use_disp,
                                        args.chunk,
                                        dataset.white_back)
        img_pred = (results['rgb_fine']).view(h, w, 3).cpu().numpy()
        

        img_pred_ = (img_pred*255).astype(np.uint8)
        imgs += [img_pred_]
        # print('eval_index',index)
        if args.dataset_name == 'replica':
            imageio.imwrite(os.path.join(dir_name, f'rgb_{index}.png'), img_pred_)
        else:
            imageio.imwrite(os.path.join(dir_name, f'{i:03d}.png'), img_pred_)

        if 'rgbs' in sample:
            # depth = sample['depth']
            rgbs = sample['rgbs']
            # depth_gt = depth.view(h, w, 1)
            img_gt = rgbs.view(h, w, 3)
            psnrs += [metrics.psnr(img_gt, img_pred).item()]
        #     # depths += [metrics.cal_depth(depth_gt, depth_pred).item()]
        #     # ssims += [metrics.ssim(img_gt, img_pred).item()]
        # break
    
    # imageio.mimsave(os.path.join(dir_name, f'{args.scene_name}.gif'), imgs, fps=30)

    if psnrs:
        mean_psnr = np.mean(psnrs)
    #     # mean_depth = np.mean(depths)
        print()
        print(f'Mean PSNR : {mean_psnr:.2f}')
    #     # print(f'Mean DEPTH : {mean_depth:.2f}')
