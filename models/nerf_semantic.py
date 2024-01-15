import torch
import torch.nn as nn
from functools import partial
from .nerf import Embedding
from .nerf import NeRF  
import numpy as np



def frequency_init(freq):
    def init(m):
        with torch.no_grad():
            if isinstance(m, nn.Linear):
                num_input = m.weight.size(-1)
                m.weight.uniform_(-np.sqrt(6 / num_input) / freq, np.sqrt(6 / num_input) / freq)
    return init

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

    def forward(self, x):
    # def forward(self, x, freq, phase_shift):
        x = self.layer(x)
        # freq = freq.unsqueeze(1).expand_as(x)
        # phase_shift = phase_shift.unsqueeze(1).expand_as(x)
        freq = 30.
        phase_shift = 1
        return torch.sin(freq * x + phase_shift)

def first_layer_film_sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)

class SemanticNeRF(NeRF):
    def __init__(self, D=8, W=256, in_channels_xyz=63, in_channels_dir=27, skips=[4], acti_type="relu"):
        self.acti_type = acti_type
        super().__init__(D, W, in_channels_xyz, in_channels_dir, skips)

        self.network = nn.ModuleList([
            FiLMLayer(self.in_channels_xyz, self.W),
            FiLMLayer(self.W, self.W),
            FiLMLayer(self.W, self.W),
            FiLMLayer(self.W, self.W),
            FiLMLayer(self.W, self.W),
            FiLMLayer(self.W, self.W),
            FiLMLayer(self.W, self.W),
            FiLMLayer(self.W, self.W),
        ])
        self.xyz_encoding_final = nn.Linear(self.W, self.W)

        self.final_layer = nn.Linear(self.W, 1)

        self.color_layer_sine = FiLMLayer(self.W + self.in_channels_dir, self.W//2)
        self.color_layer_linear = nn.Sequential(nn.Linear(self.W//2, 3))

        # self.mapping_network = CustomMappingNetwork(z_dim, 256, (len(self.network) + 1)*hidden_dim*2)

        self.network.apply(frequency_init(25))
        self.final_layer.apply(frequency_init(25))
        self.color_layer_sine.apply(frequency_init(25))
        self.color_layer_linear.apply(frequency_init(25))
        self.network[0].apply(first_layer_film_sine_init)

        # self.gridwarper = UniformBoxWarp(51)  # Don't worry about this, it was added to ensure compatibility with another model. Shouldn't affect performance.

    
    def forward(self, x, sigma_only=False):
        # frequencies = frequencies*15 + 30

        # input = self.gridwarper(input)

        if not sigma_only:
            input_xyz, input_dir = \
                torch.split(x, [self.in_channels_xyz, self.in_channels_dir], dim=-1)
        else:
            input_xyz = x

        xyz_ = input_xyz
        # for index, layer in enumerate(self.network):
        for layer in enumerate(self.network):
            # start = index * self.hidden_dim
            # end = (index+1) * self.hidden_dim
            # xyz_ = layer(xyz_, frequencies[..., start:end], phase_shifts[..., start:end])
            xyz_ = layer(xyz_)

        sigma = self.final_layer(xyz_)
        if sigma_only:
            return sigma

        xyz_encoding_final = self.xyz_encoding_final(xyz_)

        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir], -1)
        dir_encoding = self.color_layer_sine(dir_encoding_input)
        # rgb = self.rgb(dir_encoding)
        rgb = torch.sigmoid(self.color_layer_linear(dir_encoding))
        
        out = torch.cat([rgb, sigma], -1)

        return out

    
    

# NeRF_Siren = partial(NeRF_Acti,  D=8, W=256, in_channels_xyz=3, in_channels_dir=3, skips=[4], acti_type="siren")
NeRF_Siren = partial(SemanticNeRF,  D=8, W=256, in_channels_xyz=63, in_channels_dir=27, skips=[4], acti_type="siren")


