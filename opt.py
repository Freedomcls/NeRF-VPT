import argparse

def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dir', type=str,
                        default='nerf_synthetic\\lego',
                        help='root directory of dataset')
    parser.add_argument('--ckpt', type=str, default=None,  help='pretrained checkpoint path to load')
    parser.add_argument("--nerf_model", default="NeRF", help="nerf model type")
    parser.add_argument("--nerf_args", nargs="+", help="nerf argments")
    parser.add_argument('--mode', default="normal",
                        type=str, choices=['d3', 'd3_ib', 'normal','eg3d'],
                        help='use which system')
    parser.add_argument('--dataset_name', type=str, default='blender',
                        choices=['blender', 'blender_cls_ib', 'llff', "llff_cls", "llff_cls_ib", "replica"], # ib mens batch as img
                        help='which dataset to train/val')
    parser.add_argument('-sn', '--semantic_network', type=str, default='pointnet',
                        choices=['pointnet', 'conv3d'], 
                        help='use which network to extract semantic features')
    parser.add_argument('--pretrained', type=str, default=None,
                        help="pretrained-model ckpt")
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
    parser.add_argument('--perturb', type=float, default=1.0,
                        help='factor to perturb depth sampling points')
    parser.add_argument('--noise_std', type=float, default=1.0,
                        help='std dev of noise added to regularize sigma')
        
    parser.add_argument('--loss_type', type=str, default='mse',
                        choices=['mse', "msece", "msenll", "perceptual", "pm"],
                        help='loss to use')

    parser.add_argument('--batch_size', type=int, default=1024,
                        help='batch size')
    parser.add_argument('--render_train', type=int, default=0, help='render_train')
    parser.add_argument('--chunk', type=int, default=32*1024,
                        help='chunk size to split the input to avoid OOM')
    parser.add_argument('--num_epochs', type=int, default=16,
                        help='number of training epochs')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus')

    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='pretrained checkpoint path to load')
    parser.add_argument('--prefixes_to_ignore', nargs='+', type=str, default=['loss'],
                        help='the prefixes to ignore in the checkpoint state dict')

    parser.add_argument('--optimizer', type=str, default='adam',
                        help='optimizer type',
                        choices=['sgd', 'adam', 'radam', 'ranger'])
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='learning rate momentum')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--lr_scheduler', type=str, default='steplr',
                        help='scheduler type',
                        choices=['steplr', 'cosine', 'poly'])
    #### params for warmup, only applied when optimizer == 'sgd' or 'adam'
    parser.add_argument('--warmup_multiplier', type=float, default=1.0,
                        help='lr is multiplied by this factor after --warmup_epochs')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='Gradually warm-up(increasing) learning rate in optimizer')
    ###########################
    #### params for steplr ####
    parser.add_argument('--decay_step', nargs='+', type=int, default=[20],
                        help='scheduler decay step')
    parser.add_argument('--decay_gamma', type=float, default=0.1,
                        help='learning rate decay amount')
    ###########################
    #### params for poly ####
    parser.add_argument('--poly_exp', type=float, default=0.9,
                        help='exponent for polynomial learning rate decay')
    ###########################
    parser.add_argument('--render', type=str, default='test',
                        help='vpt render which dataset')
    
    parser.add_argument('--prompt_path', type=str, default='',
                        help='view prompt path')
    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')
    parser.add_argument('--is_crop', type=bool, default=False, help='crop or not')
    parser.add_argument('--is_use_mixed_precision', type=bool, default=False, help='mixed precision or not')
    return parser.parse_args()
