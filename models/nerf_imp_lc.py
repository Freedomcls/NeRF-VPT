import torch
import torch.nn as nn
from functools import partial
from .nerf import Embedding
from .nerf import NeRF  
import numpy as np

class Sine(nn.Module):
    # refer : https://github.com/vsitzmann/siren/blob/4df34baee3f0f9c8f351630992c1fe1f69114b5f/modules.py#L28
    def __init__(self, alpha=30):
        super().__init__()
        self.alpha = alpha

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(self.alpha * input)

class Sine_1(nn.Module):
    # refer : https://github.com/vsitzmann/siren/blob/4df34baee3f0f9c8f351630992c1fe1f69114b5f/modules.py#L28
    def __init__(self, alpha=1):
        super().__init__()
        self.alpha = alpha

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(self.alpha * input)

def get_acti(name, inplace=True, **kwargs):
    name = name.lower()
    if name == 'relu':
        acti = nn.ReLU(inplace=inplace)
    elif name == "siren":
        acti = Sine(**kwargs)
    else:
        raise NotImplementedError(name)
    return acti


class NeRF_Acti(NeRF):
    def __init__(self, D=8, W=256, in_channels_xyz=63, in_channels_dir=27, skips=[4], acti_type="relu"):
        self.acti_type = acti_type
        super().__init__(D, W, in_channels_xyz, in_channels_dir, skips)
    
    def build_models(self):
        first_omega_0 = 30
        hidden_omega_0 = 1
        # hidden_omega_0 = 30.

        for i in range(self.D):
            if i == 0:
                # layer = nn.Linear(self.in_channels_xyz, self.W)
                layer = SineLayer(self.in_channels_xyz, self.W, 
                                  is_first=True, omega_0=first_omega_0)
            elif i in self.skips:
                # layer = nn.Linear(self.W + self.in_channels_xyz, self.W)
                layer = SineLayer(self.W + self.in_channels_xyz, self.W, 
                                      is_first=False, omega_0=hidden_omega_0)
            else:
                # layer = nn.Linear(self.W, self.W)
                layer = SineLayer(self.W, self.W, 
                                      is_first=False, omega_0=hidden_omega_0)
            
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(self.W, self.W)

        self.dir_encoding = SineLayer(self.W + self.in_channels_dir, self.W//2, 
                                      is_first=False, omega_0=hidden_omega_0)

        # output layers
        self.sigma = nn.Linear(self.W, 1)
        with torch.no_grad():
                self.sigma.weight.uniform_(-np.sqrt(6 / self.W) / hidden_omega_0, 
                                              np.sqrt(6 / self.W) / hidden_omega_0)
        # self.rgb = nn.Sequential(
        #                 nn.Linear(self.W//2, 3),
        #                 nn.Sigmoid())

        # m = nn.Sigmoid()
        self.rgb = nn.Linear(self.W//2, 3)
        with torch.no_grad():
                self.rgb.weight.uniform_(-np.sqrt(6 / self.W) / hidden_omega_0, 
                                              np.sqrt(6 / self.W) / hidden_omega_0)
        self.rgb = nn.Sequential(
                        self.rgb,
                        nn.Sigmoid())

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
    
    
class Siren(nn.Module):
    def __init__(self, in_features=2, out_features=3,
                 hidden_features=256, hidden_layers=4, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, x):
        return self.net(x)

# NeRF_Siren = partial(NeRF_Acti,  D=8, W=256, in_channels_xyz=3, in_channels_dir=3, skips=[4], acti_type="siren")
NeRF_Siren = partial(NeRF_Acti,  D=8, W=256, in_channels_xyz=63, in_channels_dir=27, skips=[4], acti_type="siren")
# NeRF_Siren = partial(Siren,  in_features=63, out_features=3,
#                  hidden_features=256, hidden_layers=4, outermost_linear=False, 
#                  first_omega_0=30, hidden_omega_0=30.)

