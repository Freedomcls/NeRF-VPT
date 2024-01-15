import torch
from torch import nn
import numpy as np
from models import swin_encoder
from PIL import Image
from functools import partial
from opt import get_opts
from models.train_utils import requires_grad
import os
import torchvision.transforms as transforms
from datasets import augmentations
import random
import math
from torchvision.transforms import Resize

class Embedding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(Embedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels*(len(self.funcs)*N_freqs+1)

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...) 
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x: (B, self.in_channels)

        Outputs:
            out: (B, self.out_channels)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1)

class NeRF(nn.Module):
    def __init__(self,
                 D=8, W=256,
                 in_channels_xyz=63, in_channels_dir=27, 
                 skips=[4]):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        skips: add skip connection in the Dth layer
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.in_channels_mask = 3
        # self.in_channels_mask = 99
        self.skips = skips
        self.build_models()
        # self.semantic1x1 = nn.Conv2d(36800, 40000, kernel_size=1)

    def build_models(self):
        # xyz encoding layers
        # in_channel_semantic = 99
        in_channel_semantic = 3
        for i in range(self.D):
            if i == 0:
                layer = nn.Linear(self.in_channels_xyz, self.W)
            elif i in self.skips:
                layer = nn.Linear(self.W + self.in_channels_xyz, self.W)
            else:
                layer = nn.Linear(self.W, self.W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(self.W, self.W)

        # direction encoding layers
        self.dir_encoding = nn.Sequential(
                                nn.Linear(self.W + self.in_channels_dir, self.W//2),
                                nn.ReLU(True))

        # output layers
        self.sigma = nn.Linear(self.W, 1)
        self.rgb = nn.Sequential(
                        nn.Linear(self.W//2, 3),
                        nn.Sigmoid())

    def forward(self, x, sigma_only=False):
    # def forward(self, x, semantic, sigma_only=False):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
               the embedded vector of position and direction
            sigma_only: whether to infer sigma only. If True,
                        x is of shape (B, self.in_channels_xyz)

        Outputs:
            if sigma_ony:
                sigma: (B, 1) sigma
            else:
                out: (B, 4), rgb and sigma
        """

        # mask = semantic
        if not sigma_only:
            input_xyz, input_dir = \
                torch.split(x, [self.in_channels_xyz, self.in_channels_dir], dim=-1)
        else:
            input_xyz = x
        # print(semantic)
        xyz_ = input_xyz
        # if not sigma_only:
        #     xyz_ = torch.cat([input_xyz, semantic], -1)
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        sigma = self.sigma(xyz_)
        if sigma_only:
            return sigma

        xyz_encoding_final = self.xyz_encoding_final(xyz_)
        
        # print("semantic", semantic.shape)
        # if not sigma_only:
        #     input_dir = torch.cat([input_dir, semantic],-1)

        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir], -1)
        # if not sigma_only:
        #     dir_encoding = self.dir_encoding(dir_encoding_input)
        # else:
        dir_encoding = self.dir_encoding(dir_encoding_input)

        rgb = self.rgb(dir_encoding)

        out = torch.cat([rgb, sigma], -1)

        return out

class NeRFVPT(nn.Module):
    def __init__(self,
                 D=8, W=256,
                 in_channels_xyz=63, in_channels_dir=27, 
                 skips=[4]):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        skips: add skip connection in the Dth layer
        """
        super(NeRFVPT, self).__init__()
        self.D = D
        self.W = W
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.in_channels_mask = 3
        self.skips = skips
        self.build_models()
        # self.semantic1x1 = nn.Conv2d(36800, 40000, kernel_size=1)

    def build_models(self):
        # xyz encoding layers
        # in_channel_semantic = 99
        in_channel_semantic = 3
        for i in range(self.D):
            if i == 0:
                # layer = nn.Linear(self.in_channels_xyz + in_channel_semantic, self.W)
                layer = nn.Linear(self.in_channels_xyz, self.W)
            elif i in self.skips:
                layer = nn.Linear(self.W + self.in_channels_xyz, self.W)
            else:
                layer = nn.Linear(self.W, self.W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(self.W, self.W)

        # direction encoding layers
        # self.dir_encoding = nn.Sequential(
        #                         nn.Linear(self.W + self.in_channels_dir + in_channel_semantic, self.W//2),
        #                         nn.ReLU(True))
        self.dir_encoding = nn.Sequential(
                                # nn.Linear(self.W + self.in_channels_dir, self.W//2),
                                nn.Linear(self.W + self.in_channels_dir + in_channel_semantic, self.W//2),
                                nn.ReLU(True))

        # output layers
        self.sigma = nn.Linear(self.W, 1)
        self.rgb = nn.Sequential(
                        nn.Linear(self.W//2, 3),
                        nn.Sigmoid())

    def forward(self, x, sigma_only=False):
    # def forward(self, x, semantic, sigma_only=False):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
               the embedded vector of position and direction
            sigma_only: whether to infer sigma only. If True,
                        x is of shape (B, self.in_channels_xyz)

        Outputs:
            if sigma_ony:
                sigma: (B, 1) sigma
            else:
                out: (B, 4), rgb and sigma
        """

        # mask = semantic
        if not sigma_only:
            input_xyz, input_dir, semantic = \
                torch.split(x, [self.in_channels_xyz, self.in_channels_dir, self.in_channels_mask], dim=-1)
        else:
            input_xyz = x
        # print(semantic)
        xyz_ = input_xyz
        # if not sigma_only:
        #     xyz_ = torch.cat([input_xyz, semantic], -1)
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        sigma = self.sigma(xyz_)
        if sigma_only:
            return sigma

        xyz_encoding_final = self.xyz_encoding_final(xyz_)
        
        # print("semantic", semantic.shape)
        if not sigma_only:
            input_dir = torch.cat([input_dir, semantic],-1)

        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir], -1)
        # if not sigma_only:
        #     dir_encoding = self.dir_encoding(dir_encoding_input)
        # else:
        dir_encoding = self.dir_encoding(dir_encoding_input)

        rgb = self.rgb(dir_encoding)

        out = torch.cat([rgb, sigma], -1)

        return out
        


class NeRFVPT_siren(nn.Module):
    def __init__(self,
                 D=8, W=256,
                 in_channels_xyz=63, in_channels_dir=27, 
                 skips=[4]):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        skips: add skip connection in the Dth layer
        """
        super(NeRFVPT_siren, self).__init__()
        self.D = D
        self.W = W
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.in_channels_mask = 3
        self.skips = skips
        self.build_models()
    
    def build_models(self):
        first_omega_0 = 30
        hidden_omega_0 = 1
        in_channel_semantic = 3

        for i in range(self.D):
            if i == 0:
                layer = SineLayer(self.in_channels_xyz, self.W, 
                                  is_first=True, omega_0=first_omega_0)
            elif i in self.skips:
                layer = SineLayer(self.W + self.in_channels_xyz, self.W, 
                                      is_first=False, omega_0=hidden_omega_0)
            else:
                layer = SineLayer(self.W, self.W, 
                                      is_first=False, omega_0=hidden_omega_0)
            
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(self.W, self.W)

        self.dir_encoding = SineLayer(self.W + self.in_channels_dir + in_channel_semantic, self.W//2, 
                                      is_first=False, omega_0=hidden_omega_0)

        # output layers
        self.sigma = nn.Linear(self.W, 1)
        with torch.no_grad():
                self.sigma.weight.uniform_(-np.sqrt(6 / self.W) / hidden_omega_0, 
                                              np.sqrt(6 / self.W) / hidden_omega_0)

        self.rgb = nn.Linear(self.W//2, 3)
        with torch.no_grad():
                self.rgb.weight.uniform_(-np.sqrt(6 / self.W) / hidden_omega_0, 
                                              np.sqrt(6 / self.W) / hidden_omega_0)
        self.rgb = nn.Sequential(
                        self.rgb,
                        nn.Sigmoid())
        
    def forward(self, x, sigma_only=False):
        if not sigma_only:
            input_xyz, input_dir, semantic = \
                torch.split(x, [self.in_channels_xyz, self.in_channels_dir, self.in_channels_mask], dim=-1)
        else:
            input_xyz = x
        # print(semantic)
        xyz_ = input_xyz

        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        sigma = self.sigma(xyz_)
        if sigma_only:
            return sigma
        xyz_encoding_final = self.xyz_encoding_final(xyz_)
        
        # print("semantic", semantic.shape)
        if not sigma_only:
            input_dir = torch.cat([input_dir, semantic],-1)
        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir], -1)
        dir_encoding = self.dir_encoding(dir_encoding_input)
        rgb = self.rgb(dir_encoding)
        out = torch.cat([rgb, sigma], -1)

        return out    

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6/self.in_features) / self.omega_0, 
                                             np.sqrt(6/self.in_features) / self.omega_0)
        
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

# NeRF_Siren = partial(NeRFVPT,  D=8, W=256, in_channels_xyz=63, in_channels_dir=27, skips=[4], acti_type="siren")

class UniformBoxWarp(nn.Module):
    def __init__(self, sidelength):
        super().__init__()
        self.scale_factor = 2/sidelength

    def forward(self, coordinates):
        return coordinates * self.scale_factor

class FiLMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)

    # def forward(self, x):
    def forward(self, x, freq, phase_shift):
        x = self.layer(x)
        # print("xxxxxxxx",x,x.shape)
        # print("freq",freq,freq.shape)
        # print("phshi",phase_shift,phase_shift.shape)
        
        freq = freq.expand_as(x)
        phase_shift = phase_shift.expand_as(x)
        # freq = 30.
        # phase_shift = 0
        return torch.sin(freq * x + phase_shift)

def frequency_init(freq):
    def init(m):
        with torch.no_grad():
            if isinstance(m, nn.Linear):
                num_input = m.weight.size(-1)
                m.weight.uniform_(-np.sqrt(6 / num_input) / freq, np.sqrt(6 / num_input) / freq)
    return init
    
def first_layer_film_sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)
