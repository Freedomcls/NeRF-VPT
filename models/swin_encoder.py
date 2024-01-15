# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from models.helpers import PatchEmbed, BasicLayer, PatchMerging
import torch.nn.functional as F

class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        self.head.weight.data.fill_(0)
        self.head.bias.data.fill_(0)

        # soft-view-pooling
        # in_channels = self.in_shape[1]
        # self.out_channels = in_channels

        # kernel = 3
        # bias = True
        # if kernel == 1:
        #     self.attn = nn.Sequential(
        #         nn.Conv2d(in_channels,
        #                   in_channels // 2,
        #                   kernel_size=kernel,
        #                   stride=1,
        #                   bias=bias), nn.ReLU(inplace=True),
        #         nn.Conv2d(in_channels // 2,
        #                   in_channels,
        #                   kernel_size=kernel,
        #                   stride=1,
        #                   bias=bias))
        # elif kernel == 3:
        #     self.attn = nn.Sequential(
        #         nn.Conv2d(
        #             in_channels,
        #             in_channels // 2,
        #             kernel_size=kernel,
        #             stride=2,
        #             padding=1,
        #             bias=bias,
        #         ), nn.ReLU(inplace=True),
        #         nn.ConvTranspose2d(in_channels // 2,
        #                            in_channels,
        #                            kernel_size=kernel,
        #                            stride=2,
        #                            padding=1,
        #                            output_padding=1,
        #                            bias=bias))
        # else:
        #     raise RuntimeError('[!] cfg.view_pool.kernel={} is not supported.'.format(kernel))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        # print("0", x.shape)
        # [1, 99, 224, 224]
        x = self.patch_embed(x)  #[1, 3136, 96]
        # print("1", x.shape)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        # print("2", x.shape)

        for layer in self.layers:
            x = layer(x)

        # print("3", x.shape)
        x = self.norm(x)  # B L C  [1, 49, 768]
        # print("1", x.shape)
        x = self.avgpool(x.transpose(1, 2))  # B C 1  [1, 768, 1]
        # print("before_flatten", x.shape)
        x = torch.flatten(x, 1)
        return x

    def soft_view_pooling(self, x):
        B, V, C, H, W = x.size()
        a = self.attn(x.view(B * V, C, H, W)) 
        a = F.softmax(a.view(B, V, C, H, W), dim=1)
        ax = torch.sum(a * x, dim=1)
            
        return ax

    def patch_drop(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        return x

    def forward(self, x):
        # x = [self.forward_features(mask) for mask in x]
        # x = torch.stack(x,0)
        # x = torch.max(x,0)[0]
        
        # x = self.soft_view_pooling(x)
        x = self.forward_features(x)
        # print("before_head", x.shape)
        x = self.head(x)
        x = self.feat_to_out_dict(x)
        return x

    def feat_to_out_dict(self, feat):
        out_dict = {}
        frequencies = feat[:, :feat.shape[-1] // 2]
        phase_shifts = feat[:, feat.shape[-1] // 2:]
        latent_code = [frequencies, phase_shifts]
        # print("frequencies", frequencies, frequencies.shape)
        out_dict['latent_code'] = latent_code
        return out_dict

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops
