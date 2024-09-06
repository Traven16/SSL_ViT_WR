# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import torch.nn.functional as F
import timm.models.vision_transformer


class StyleSampleViT(nn.Module):
    def __init__(self, vit):
        super(StyleSampleViT, self).__init__()
        self.vit = vit
    
    def forward(self, x):
        B, N, C, W, H = x.shape
        x = x.view(N*B, C, W, H)
        x = self.vit(x)
        BN, D = x.shape
        x = x.view(N,B,D)
        x = x.mean(dim=0)
        return x
        

class NetVLAD(nn.Module):
    """Net(R)VLAD layer implementation"""

    def __init__(self, num_clusters=100, dim=64, alpha=100.0, random=False):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            random : bool
                enables NetRVLAD, removes alpha-init and normalization

        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.random = random
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)

        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self._init_params()

    def _init_params(self):
        if not self.random:
            self.conv.weight = nn.Parameter(
                (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
            )
            self.conv.bias = nn.Parameter(
                - self.alpha * self.centroids.norm(dim=1)
            )

    def forward(self, x):
        N, C = x.shape[:2]

        if not self.random:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim
        
        x = x.transpose(2,1).unsqueeze(-1)
        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        # x = self.pool(x)
        x_flatten = x.reshape(N, C, -1).transpose(2,1)
        
        # calculate residuals to each clusters
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
            self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        
        residual *= soft_assign.unsqueeze(2)

        vlad = residual.sum(dim=-1)

        if not self.random:
            vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        
        vlad = vlad.view(x.size(0), -1)  # flatten

        if not self.random:
            vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad


class ViTVLAD(nn.Module):
    def __init__(self, model, vlad_clusters, vlad_dim, num_classes):
        super(ViTVLAD, self).__init__()
        self.model = model
        model_features = self.model.num_features
        self.vlad = NetVLAD(num_clusters=vlad_clusters, dim=vlad_dim, random=True).to("cuda")
        self.pred = nn.Linear(vlad_dim*vlad_clusters, num_classes).to("cuda")
    
    def forward(self, x):
        x = self.model.forward_features_all(x)
        x = self.vlad(x)
        x = self.pred(x)
        return x


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, n_fg=1, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        # use average pooling to get the foreground
        patch_size = kwargs["patch_size"]
        self.avg2d = torch.nn.AvgPool2d(patch_size, patch_size)
        self.n_fg = n_fg / (patch_size ** 2) # conversion from avg pool to sum pooling
            
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            #del self.norm  # remove the original norm
    
    def forward_features_all(self, x, return_foreground=False):
        
        #f = self.max2d(x).mean(dim=1) == 1
        f = self.avg2d(x).mean(dim=1) >= self.n_fg
        f = f.flatten(1)
        
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)
        
        if return_foreground:
            return x, f
        
        return x
    
    def forward_global_pool_and_cls(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x_cls = x[:, 1:, :].mean(dim=1)  # global pool without cls token
        global_pool = self.fc_rm(x_cls)
        
        x = self.norm(x)
        cls = x[:, 0]

        return cls, global_pool
        
    
    def get_last_selfattention(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)
    
    def forward_features_attn(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        
        return x[:, 0]
        

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome


def vit_base(**kwargs):
    model = VisionTransformer(
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_512_patch32(**kwargs):
    model = VisionTransformer(
        img_size=384,patch_size=32, embed_dim=512, depth=12, num_heads=16, qkv_bias=True,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_small_patch32_impl(**kwargs):
    model = VisionTransformer(
        img_size=224, patch_size=16, embed_dim=256, depth=8, num_heads=8,
        mlp_ratio=4, qkv_bias=True,norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)   
    return model

def vit_tiny_48(**kwargs):
    model = VisionTransformer(
        img_size=48, patch_size=8, embed_dim=64, depth=8, num_heads=8, qkv_bias=True,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)   
    return model

def vit_small(**kwargs):
    model = VisionTransformer(
        embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_custom(**kwargs):
    model = VisionTransformer(
        img_size=224, patch_size=16, embed_dim=256, depth=16, num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)   
    return model






        