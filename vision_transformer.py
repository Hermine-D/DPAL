from functools import partial
import os
import torch
import torch.nn as nn

import collections.abc as container_abcs
from itertools import repeat
from sapiens_vit import VisionTransformer as SapiensVisionTransformer

def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        qk = (q @ k.transpose(-2, -1)) * self.scale
        qk = qk.softmax(dim=-1)
        attn = self.attn_drop(qk)
        
        vv = ((v @ v.transpose(-2, -1)) * self.scale).softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, [qk, vv]

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads=num_heads,batch_first=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, key, value, return_attention=False):
        query = self.norm1(x)
        y, attn = self.attn(query, key, value)
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self, img_size=(224, 224), patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, num_heads_in_last_block=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.img_size = to_2tuple(img_size)
        self.patch_embed = PatchEmbed(
            img_size=self.img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth-1)]+[Block(
                dim=embed_dim, num_heads=num_heads_in_last_block, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[-1], norm_layer=norm_layer)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, h, w):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and self.img_size == (h,w):
            return self.pos_embed
        
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        OH = self.img_size[0] // self.patch_embed.patch_size
        OW = self.img_size[1] // self.patch_embed.patch_size
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, OH, OW, dim).permute(0, 3, 1, 2),
            scale_factor=(h0 / OH, w0 / OW),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-1] and int(h0) == patch_pos_embed.shape[-2]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        # print('x',x.shape)
        x = self.patch_embed(x)  # patch linear embedding
        # print('x1',x.shape)

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def forward(self, x):
        # print('begin:',x.shape)
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            x, atten = blk(x)
        x = self.norm(x)
        return x, atten

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x, _ = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


class ExpertViT(VisionTransformer):
    """ Vision Transformer """
    def __init__(self, pretrained=None, **kwargs):
        super().__init__(**kwargs)
        self._init_from_pretrained(pretrained)

    def _init_from_pretrained(self, pretrained=None):
        if pretrained:
            checkpoint_model = torch.load(pretrained, map_location='cpu')
            if 'model' in checkpoint_model:
                param_dict = checkpoint_model['model']
            elif 'state_dict' in checkpoint_model:
                param_dict = checkpoint_model['state_dict']
            elif 'student' in checkpoint_model: ### for dino
                print('load from student')
                param_dict = checkpoint_model["student"]
            else:
                param_dict = checkpoint_model
            param_dict = {k.replace("backbone.", ""): v for k, v in param_dict.items()}
            param_dict = {k.replace("module.", ""): v for k, v in param_dict.items()}
            
            msg = self.load_state_dict(param_dict, strict=False)
            print('Load from {}: {}'.format(pretrained, msg))

    def forward_backbone(self, x):
        # print('begin:',x.shape)
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            x, atten = blk(x)
        x = self.norm(x)
        
        return x, atten

    def forward(self, base_imgs, meta):
        bs = base_imgs.shape[0]
        msc_imgs = meta['region_imgs']
        b_ph, b_pw = base_imgs.shape[2]//self.patch_embed.patch_size, base_imgs.shape[3]//self.patch_embed.patch_size
        if meta['aug_img'] is not None:
            base_inputs = torch.cat([base_imgs, meta['aug_img']])
            len_base_inputs = 2
        else:
            base_inputs = base_imgs
            len_base_inputs = 1
        
        outputs = {}
        
        #  single-person
        b_feats, last_atten = self.forward_backbone(base_inputs)
        outputs['feats_from_teacher'] = list(torch.chunk(b_feats[:,:1,:], len_base_inputs, dim=0))
        outputs['feats_from_teacher_patch'] = list(torch.chunk(b_feats[:,1:,:], len_base_inputs, dim=0))
        qk_atten = torch.chunk(last_atten[0], len_base_inputs, dim=0)
        vv_atten = torch.chunk(last_atten[1], len_base_inputs, dim=0)
        outputs['qkv_atten'] = []
        for i in range(len_base_inputs):
            outputs['qkv_atten'].append([qk_atten[i], vv_atten[i]])
        # multi-person
        b_feats, last_atten = self.forward_backbone(meta['crowd_imgs'])
        outputs['feats_from_teacher_patch'].append(b_feats[:,1:,:])
        outputs['qkv_atten'].append(last_atten)
        # # single-person
        # if msc_imgs:
        #     len_msc_imgs= len(msc_imgs)
        #     b_feats, last_atten = self.forward_backbone(torch.cat(msc_imgs))
        #     outputs['feats_from_teacher'].extend(list(torch.chunk(b_feats[:,0,:], len_msc_imgs, dim=0)))
        #     outputs['feats_from_teacher_patch'].extend(list(torch.chunk(b_feats[:,1:,:], len_msc_imgs, dim=0)))
        #     qk_atten = torch.chunk(last_atten[0], len_msc_imgs, dim=0)
        #     vv_atten = torch.chunk(last_atten[1], len_msc_imgs, dim=0)
        #     for i in range(len_msc_imgs):
        #         outputs['qkv_atten'].append([qk_atten[i], vv_atten[i]])
            
        return outputs

######################################## vit ########################################
def expert_vit_base(patch_size=16, **kwargs):
    model = ExpertViT(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def expert_vit_large(patch_size=16, **kwargs):
    model = ExpertViT(
        patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, num_heads_in_last_block=16,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

######################################## solider ##########################################
from swin_transformer import SwinTransformer

class ExpertSolider(nn.Module):
    def __init__(self, img_size=224,drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0., pretrained=None, **kwargs):
        super().__init__()
        self.model = SwinTransformer(pretrain_img_size = img_size, patch_size=4, window_size=7, embed_dims=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), drop_path_rate=0.1, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, **kwargs)
        ckpt = torch.load(pretrained, map_location='cpu')['student']
        new_ckpt = {}
        for k, v in ckpt.items():
            if "decoder_module" in k:
                continue
            if "neck_module" in k:
                continue
            new_key = k.replace("module.backbone.", "")
            new_ckpt[new_key] = v
        missing_keys, unexpected_keys = self.model.load_state_dict(new_ckpt, strict=False)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
    
    def forward(self, base_imgs, meta):
        bs = base_imgs.shape[0]
        b_ph, b_pw = base_imgs.shape[2]//32, base_imgs.shape[3]//32
        base_inputs = base_imgs
        
        outputs = {}
            
        b_feats_global, b_feats = self.model(base_inputs)
        last_atten = None

        outputs['feats_from_teacher'] = [b_feats_global.unsqueeze(1)]
        outputs['feats_from_teacher_patch'] = [b_feats[-1].reshape(bs, 1024, b_ph*b_pw).permute(0,2,1)]
        outputs['qkv_atten'] =  last_atten
        return outputs