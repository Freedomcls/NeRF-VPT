from pytorch_lightning import LightningModule
from models.nerf import Embedding, NeRF, NeRFVPT
from models.rendering import render_rays, render_semantic_rays
# from models.ConvNetWork import *

# optimizer, scheduler, visualization
from utils import *
from losses import loss_dict
from metrics import *
from torch.utils.data import DataLoader
from datasets import dataset_dict
from collections import defaultdict


class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super(NeRFSystem, self).__init__()
        self.hparams = hparams

        self.loss = loss_dict[hparams.loss_type]()
        self.weight_loss = loss_dict["weight"]()
        self.delta_loss = loss_dict["delta"]()

        self.prompt_path = hparams.prompt_path
        self.render = hparams.render
        
        self.embedding_xyz = Embedding(3, 10) # 10 is the default number
        self.embedding_dir = Embedding(3, 4) # 4 is the default number
        self.embeddings = [self.embedding_xyz, self.embedding_dir]

        try:
            init_args =  hparams.nerf_args if hparams.nerf_args else []
            nerf = eval(hparams.nerf_model)(*init_args) 
        except NameError as e:
            print(hparams.nerf_model, hparams.nerf_args)
            raise RuntimeError(e)

        if hparams.nerf_model == 'NeRF':
            self.if_vpt = False
        else:
            self.if_vpt = True
        
        self.nerf_coarse = nerf
        if hparams.ckpt:
            self.nerf_coarse = self.load_partial_model(self.nerf_coarse, hparams.ckpt, model_name='nerf_coarse')
        self.models = [self.nerf_coarse]
        if hparams.N_importance > 0:
            self.nerf_fine = nerf
            if hparams.ckpt:
                self.nerf_fine = self.load_partial_model(self.nerf_fine, hparams.ckpt, model_name='nerf_fine')
            self.models += [self.nerf_fine]
            
    def load_partial_model(self, model, checkpoint_path, model_name):
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint['state_dict']

        loaded_state_dict = {}
        for name in state_dict:
            if model_name in name:
                if 'xyz_encoding' in name or 'sigma' in name:
                    loaded_state_dict[name] = state_dict[name]

        # load
        model.load_state_dict(loaded_state_dict, strict=False)
        print('load finish')
        return model
    
    def decode_batch(self, batch):
        rays = batch['rays'] # (B, 8)
        rgbs = batch['rgbs'] # (B, 3)
        if 'NeRFVPT' in self.hparams.nerf_model:
            semantic = batch['semantic'] # (B, 3)
        # nerf = batch['nerf'] # (B, 3)
        # img = batch['img'] 
        # return rays, rgbs
            return rays, rgbs, semantic
        return rays, rgbs

    def forward(self, rays, semantic):
    # def forward(self, rays):
        """Do batched inference on rays using chunk."""
        rays = rays.reshape(-1,8) # todo
        B = rays.shape[0] # B is equal to H*W
        results = defaultdict(list)
        # self.models[0].requires_grad(enc_flag=True)
        # self.models[1].requires_grad(enc_flag=True)
        
        for i in range(0, B, self.hparams.chunk):
            # rendered_ray_chunks = render_rays(self.models,
            if 'NeRFVPT' in self.hparams.nerf_model:
                rendered_ray_chunks = render_semantic_rays(self.models,
                                self.embeddings,
                                rays[i:i+self.hparams.chunk],
                                # [x[i:i+self.hparams.chunk] for x in img],
                                semantic[i:i+self.hparams.chunk],
                                self.hparams.N_samples,
                                self.hparams.use_disp,
                                self.hparams.perturb,
                                self.hparams.noise_std,
                                self.hparams.N_importance,
                                self.hparams.chunk, # chunk size is effective in val mode
                                self.train_dataset.white_back)
            else:
                rendered_ray_chunks = render_rays(self.models,
                                self.embeddings,
                                rays[i:i+self.hparams.chunk],
                                self.hparams.N_samples,
                                self.hparams.use_disp,
                                self.hparams.perturb,
                                self.hparams.noise_std,
                                self.hparams.N_importance,
                                self.hparams.chunk, # chunk size is effective in val mode
                                self.train_dataset.white_back)

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    def prepare_data(self):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir,
                  'img_wh': tuple(self.hparams.img_wh),
                  }
        if 'llff' in self.hparams.dataset_name:
            kwargs['spheric_poses'] = self.hparams.spheric_poses
            kwargs['val_num'] = self.hparams.num_gpus

        # self.train_dataset = dataset(split='train', nerf_model=self.hparams.nerf_model, **kwargs)
        # self.val_dataset = dataset(split='val', nerf_model=self.hparams.nerf_model, **kwargs)
        self.train_dataset = dataset(split='train', render=self.render, if_vpt=self.if_vpt, prompt_path=self.prompt_path, **kwargs)
        self.val_dataset = dataset(split='val', render=self.render, if_vpt=self.if_vpt, prompt_path=self.prompt_path, **kwargs)

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams, self.models)
        scheduler = get_scheduler(self.hparams, self.optimizer)
        
        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                        #   shuffle=False,
                          shuffle=True,
                          num_workers=4,
                          batch_size=self.hparams.batch_size,  # slpit all rays (Num_img * H * W)
                          pin_memory=True) 

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1, # validate one image (H*W rays) at a time
                          pin_memory=True)
    
    def training_step(self, batch, batch_nb):
        log = {'lr': get_learning_rate(self.optimizer)}
        # rays, rgbs, img = self.decode_batch(batch)
        if 'NeRFVPT' in self.hparams.nerf_model:
            rays, rgbs, semantic = self.decode_batch(batch)
        else:
            rays, rgbs = self.decode_batch(batch)
            
        if self.hparams.is_use_mixed_precision:
            with torch.cuda.amp.autocast():
                if 'NeRFVPT' in self.hparams.nerf_model:
                    results = self(rays, semantic) # all pics rays concat
                else:
                    results = self(rays, 0) # all pics rays concat
                    
                # results = self(rays) # all pics rays concat
        else:
            # results = self(rays)
            if 'NeRFVPT' in self.hparams.nerf_model:
                results = self(rays, semantic) # all pics rays concat
            else:
                results = self(rays, 0) # all pics rays concat

        # print("result", results['rgb_coarse'],results['rgb_coarse'].shape)
        # print("rgbs", rgbs,rgbs.shape)
        
        log['train/mse_loss'] = loss = self.loss(results, rgbs)
        typ = 'fine' if 'rgb_fine' in results else 'coarse'

        with torch.no_grad():
            # psnr_ = psnr(results[f'rgb_{typ}'] + nerf, rgbs)
            psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
            log['train/psnr'] = psnr_

        return {'loss': loss,
                'progress_bar': {'train_psnr': psnr_},
                'log': log
               }

    def validation_step(self, batch, batch_nb):
        if 'NeRFVPT' in self.hparams.nerf_model:
            rays, rgbs, semantic = self.decode_batch(batch)
        else:
            rays, rgbs = self.decode_batch(batch)
            
        # rays, rgbs, semantic, nerf = self.decode_batch(batch)
        rays = rays.squeeze() # (H*W, 3)
        rgbs = rgbs.squeeze() # (H*W, 3)
        if 'NeRFVPT' in self.hparams.nerf_model:
            semantic = semantic.squeeze() # (H*W, 3)
            # nerf = nerf.squeeze() # (H*W, 3)
            results = self(rays,semantic)
        else:
            results = self(rays,0)
        
        # results = self(rays)
        log = {'val_loss': self.loss(results, rgbs)}
        typ = 'fine' if 'rgb_fine' in results else 'coarse'
    
        if batch_nb == 0:
            W, H = self.hparams.img_wh
            # img = (results[f'rgb_{typ}']+nerf).view(H, W, 3).cpu()
            img = results[f'rgb_{typ}'].view(H, W, 3).cpu()
            img = img.permute(2, 0, 1) # (3, H, W)
            img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
            depth = visualize_depth(results[f'depth_{typ}'].view(H, W)) # (3, H, W)
            stack = torch.stack([img_gt, img, depth]) # (3, 3, H, W)
            self.logger.experiment.add_images('val/GT_pred_depth',
                                               stack, self.global_step)
        # print("psnr_result", results[f'rgb_{typ}'].shape, rgbs.shape)
        log['val_psnr'] = psnr(results[f'rgb_{typ}'], rgbs)
        return log

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        # mean_loss = torch.stack([x['val_mse_loss'] for x in outputs]).mean()
        # mean_loss += torch.stack([x['val_weight_loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()

        return {'progress_bar': {'val_loss': mean_loss,
                                 'val_psnr': mean_psnr},
                'log': {'val/loss': mean_loss,
                        'val/psnr': mean_psnr}
               }



if __name__ == '__main__':

    eg3d_renderer = EG3D_Renderer()
    conditioning_params = -1
    # conditioning_params = torch.ones(1,12)
    rayo = torch.ones((128*128,3)).cuda()
    rayd = torch.ones((128*128,3)).cuda()
    image_dict = eg3d_renderer.render(conditioning_params, rayo, rayd)
    a = image_dict['rgb_fine']
    print(torch.max(a),torch.min(a))
    b = image_dict['depth_fine']
    c = image_dict['opacity_fine']
    print(a.shape,b.shape,c.shape)
    print(1)

    from opt import get_opts
    hparams = get_opts()
    print(hparams)
    print(hparams.dataset_name)
    system = EG3DSystem(hparams)
    
