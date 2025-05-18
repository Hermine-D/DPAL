from functools import partial

import torch
import torch.nn as nn
import random
from vision_transformer import Block
from pretrain_moe_decoder import LossMOEDecoder, LossMOEDecoderV2
from mmpretrain.models import VisionTransformer
from einops import rearrange

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=(224, 224), patch_size=(16, 16), in_chans=3, embed_dim=768):
        super().__init__()
        
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class MaskedVisionTransformer(nn.Module):
    """ Masked Autoregressor with VisionTransformer backbone
    """
    def __init__(self, img_size=(224, 224), patch_size=(16,16), in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16, num_heads_in_last_block=3,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.patch_size = patch_size

        self.cls_token = nn.Parameter(torch.rand(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.rand(1, num_patches + 1, embed_dim), requires_grad=True) 

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth-1)]+[Block(embed_dim, num_heads_in_last_block, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)])
        self.norm = norm_layer(embed_dim)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, x, mask_ratio):
        # embed patches
        # h, w = x.size(2), x.size(3)
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        pos_embed = interpolate_pos_encoding(x, self.pos_embed)
        x = x + pos_embed[:,1:,:]

        # masking: length -> length * mask_ratio
        if mask_ratio > 0.:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)
        else:
            mask, ids_restore = None, None

        # append cls token
        cls_token = self.cls_token + pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

class DPALViT(MaskedVisionTransformer):
    """ Masked Autoregressor with VisionTransformer backbone
    """
    def __init__(self, img_size=(224, 224), patch_size=(16,16), in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16, num_heads_in_last_block=3,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, pretrained='',**args):
        super().__init__(img_size, patch_size, in_chans,
                 embed_dim, depth, num_heads, num_heads_in_last_block,
                 mlp_ratio, norm_layer)

        # --------------------------------------------------------------------------
        self.out_index = [6]
        self.inter_norm = norm_layer(embed_dim)
        self.anchor_size = (img_size[0]//patch_size[0], img_size[1]//patch_size[1])
        self.anchor_patches = self.anchor_size[0] * self.anchor_size[1]
        self.initialize_weights(pretrained)

    def initialize_weights(self, pretrained=''):
        # initialization
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)
        if pretrained != '':
            self.init_from_pretrain(pretrained)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_from_pretrain(self, pretrained=None):
        checkpoint_model = torch.load(pretrained, map_location='cpu')
        if 'model' in checkpoint_model:
            param_dict = checkpoint_model['model']
        elif 'state_dict' in checkpoint_model:
            param_dict = checkpoint_model['state_dict']
        elif 'student' in checkpoint_model: ### for dino
            param_dict = checkpoint_model["student"]
        else:
            param_dict = checkpoint_model
        param_dict = {k.replace("backbone.", ""): v for k, v in param_dict.items()}
        param_dict = {k.replace("module.", ""): v for k, v in param_dict.items()}
        count=0
        for k, v in param_dict.items():
            if k not in self.state_dict().keys():
                continue
            if 'head' in k or 'dist' in k or 'pre_logits' in k:
                continue
            if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
                # For old models that I trained prior to conv based patchification
                O, I, H, W = self.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
            elif k == 'pos_embed' and v.shape != self.pos_embed.shape:
                # To resize pos embedding when using model at different size from pretrained weights
                print('shape resize from :{}: param_dict{} to self.state_dict(){}'.format(k, v.shape, self.state_dict()[k].shape))
                b, l, d = self.state_dict()[k].size()
                pos_emb = v[:,1:,:]
                pos_emb = torch.nn.functional.interpolate(pos_emb.unsqueeze(0), size=(l-1,d),mode='bilinear')[0]
                v = torch.cat([v[:,0,:].unsqueeze(1), pos_emb], dim=1)
                param_dict[k] = v
            try:
                self.state_dict()[k].copy_(v)
                count +=1
            except:
                print('===========================ERROR=========================')
                print('shape do not match in k :{}: param_dict{} vs self.state_dict(){}'.format(k, v.shape, self.state_dict()[k].shape))
        print('Load %d / %d layers.'%(count,len(self.state_dict().keys())))
        msg = self.load_state_dict(param_dict, strict=False)
        print(msg)
        
    def get_mask(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        L = self.anchor_patches
        N = x.shape[0]
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 1 is keep, 0 is remove
        mask = torch.zeros([N, L], device=x.device)
        mask[:, :len_keep] = 1.
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return mask
    
    def random_masking_with_ref_anchor(self, masked_x, mask_ratio, hw_size):
        
        B, L, D = masked_x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        base_mask = self.get_mask(masked_x, mask_ratio)
        base_mask = base_mask.view(B, 1, self.anchor_size[0], self.anchor_size[1])
        mask = torch.nn.functional.interpolate(base_mask, hw_size)
        mask = mask.flatten(1)
        ids_shuffle = torch.argsort(1-mask, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(masked_x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        return x_masked, [base_mask, 1-mask], ids_restore

    def forward(self, x, mask_ratio, with_anchor_mask=False, get_intermediate_only=False, with_cls_token=True, return_resolution=False):
        # embed patches
        
        ph, pw = x.size(2)//self.patch_size[0], x.size(3)//self.patch_size[1]
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        pos_embed = interpolate_pos_encoding(x, self.pos_embed, x.shape[1])
        x = x + pos_embed[:,1:,:]

        # masking: length -> length * mask_ratio
        if mask_ratio > 0.:
            if with_anchor_mask:
                x, mask, ids_restore = self.random_masking_with_ref_anchor(x, mask_ratio, (ph, pw))
            else:
                x, mask, ids_restore = self.random_masking(x, mask_ratio)
        else:
            mask, ids_restore = None, None

        if with_cls_token:
            # append cls token
            cls_token = self.cls_token + pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        
        # apply Transformer blocks
        _outs = []
        
        for i, blk in enumerate(self.blocks):
            x, attn = blk(x)
            if i in self.out_index:
                _outs.append(self.inter_norm(x))
                if get_intermediate_only:
                    return _outs, mask, ids_restore, attn
            
        _outs.append(self.norm(x))
        
        if return_resolution:
            return _outs, mask, ids_restore, attn, (ph, pw)
        else:
            return _outs, mask, ids_restore, attn

class DPALKDViTMOEWrapper(nn.Module):
   
    def __init__(self, **args):
        super(DPALKDViTMOEWrapper, self).__init__()
        self.backbone = DPALViT(**args)
        self.adapter = LossMOEDecoder(**args)
    
    def forward(self, base_imgs, meta):
        B,_,_,_ = base_imgs.shape
        msc_imgs = meta['region_imgs']

        if meta['aug_img'] is not None:
            base_inputs = torch.cat([base_imgs, meta['aug_img']])
            len_base_inputs = 2
        else:
            base_inputs = base_imgs
            len_base_inputs = 1
        
        outputs = {}
        moe_loss = torch.Tensor([0]).cuda()
        experts_loss = torch.Tensor([0]).cuda()
        num_aux_loss = 0
        ########## Single-person cls + atten + patch
        # atten
        b_feats, _, _, aligned_atten = self.backbone(base_inputs, 0.)
        qk_atten = torch.chunk(aligned_atten[0], len_base_inputs, dim=0)
        vv_atten = torch.chunk(aligned_atten[1], len_base_inputs, dim=0)
        aligned_attens = []
        for i in range(len_base_inputs):
            aligned_attens.append([qk_atten[i], vv_atten[i]])
        # cls
        b_aligned_feats = b_feats[1]        
        b_aligned_feats, _, aux_loss, i_experts_loss = self.adapter(b_aligned_feats[:,:1,:], target_expert=0)
        moe_loss += aux_loss
        experts_loss += i_experts_loss
        num_aux_loss += 1
        aligned_cls_feats = list(torch.chunk(b_aligned_feats, len_base_inputs, dim=0))
        # patch
        b_aligned_feats = b_feats[1]
        b_aligned_feats, _, aux_loss, i_experts_loss = self.adapter(b_aligned_feats[:,1:,:], target_expert=1)
        moe_loss += aux_loss
        experts_loss += i_experts_loss
        num_aux_loss += 1
        aligned_patch_feats = list(torch.chunk(b_aligned_feats, len_base_inputs, dim=0))
        ########## Multi-person atten + patch
        # atten
        b_feats, _, _, aligned_atten = self.backbone(meta['crowd_imgs'], 0.)
        aligned_attens.append(aligned_atten)
        # patch
        b_aligned_feats = b_feats[1]
        b_aligned_feats, _, aux_loss, i_experts_loss = self.adapter(b_aligned_feats[:,1:,:], target_expert=2)
        moe_loss += aux_loss
        experts_loss += i_experts_loss
        num_aux_loss += 1
        aligned_patch_feats.append(b_aligned_feats)
        ########## Single person cls
        if msc_imgs:
            len_msc_imgs = len(msc_imgs)
            for i in range(len_msc_imgs):
                msc_feats, _, _, aligned_atten = self.backbone(msc_imgs[i], 0.)
                msc_aligned_feats = msc_feats[1]
                msc_aligned_feats, _, aux_loss, i_experts_loss = self.adapter(msc_aligned_feats[:,:1,:], target_expert=0)
                moe_loss += aux_loss
                experts_loss += i_experts_loss
                num_aux_loss += 1
                aligned_cls_feats.append(msc_aligned_feats)
                
        img_level_aligned_tokens = aligned_cls_feats
        outputs['aligned_cls_feats'] = img_level_aligned_tokens
        outputs['aligned_patch_feats'] = aligned_patch_feats
        outputs['qkv_atten'] =  aligned_attens
        moe_loss /= num_aux_loss
        experts_loss /= num_aux_loss
        return outputs, moe_loss, experts_loss

from swin_transformer import SwinTransformer
class DPALKDSWINMOEWrapper(nn.Module):
    def __init__(self, img_size, backbone="swin_tiny", **args):
        super(DPALKDSWINMOEWrapper, self).__init__()
        if backbone == "swin_tiny":
            self.backbone = SwinTransformer(pretrain_img_size = img_size, patch_size=4, window_size=7, embed_dims=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), drop_path_rate=0.1, drop_rate=0.0, attn_drop_rate=0.0)
        elif backbone == "swin_small":
            self.backbone = SwinTransformer(pretrain_img_size = img_size, patch_size=4, window_size=7, embed_dims=96, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24), drop_path_rate=0.1, drop_rate=0.0, attn_drop_rate=0.0)
        self.adapter = LossMOEDecoderV2(**args)
    
    def forward(self, base_imgs, meta):
        B,_,_,_ = base_imgs.shape
        msc_imgs = meta['region_imgs']

        if meta['aug_img'] is not None:
            base_inputs = torch.cat([base_imgs, meta['aug_img']])
            len_base_inputs = 2
        else:
            base_inputs = base_imgs
            len_base_inputs = 1
        
        outputs = {}
        moe_loss = torch.Tensor([0]).cuda()
        experts_loss = torch.Tensor([0]).cuda()
        num_aux_loss = 0
        ########## Single-person cls + patch
        avg_pool, patch  = self.backbone.forward_student(base_inputs) # b_feats[-1]
        b_feats = torch.cat([avg_pool.unsqueeze(1), patch[-1].reshape(B, 768, (256//32)*(128//32)).permute(0,2,1)], dim=1)
        # cls
        b_aligned_feats, _, aux_loss, i_experts_loss = self.adapter(b_feats[:,:1,:], target_expert=0)
        moe_loss += aux_loss
        experts_loss += i_experts_loss
        num_aux_loss += 1
        aligned_cls_feats = list(torch.chunk(b_aligned_feats, len_base_inputs, dim=0))
        # patch
        b_aligned_feats, _, aux_loss, i_experts_loss = self.adapter(b_feats[:,1:,:], target_expert=1)
        moe_loss += aux_loss
        experts_loss += i_experts_loss
        num_aux_loss += 1
        aligned_patch_feats = list(torch.chunk(b_aligned_feats, len_base_inputs, dim=0))
        # ########## Multi-person patch
        b_feats = self.backbone(meta['crowd_imgs'])
        # patch
        b_aligned_feats, _, aux_loss, i_experts_loss = self.adapter(b_feats[:,1:,:], target_expert=2)
        moe_loss += aux_loss
        experts_loss += i_experts_loss
        num_aux_loss += 1
        aligned_patch_feats.append(b_aligned_feats)        
        ########## Single-person cls
        if msc_imgs:
            len_msc_imgs = len(msc_imgs)
            for i in range(len_msc_imgs):
                avg_pool, patch = self.backbone.forward_student(msc_imgs[i])
                msc_feats = torch.cat([avg_pool.unsqueeze(1), patch[-1].reshape(B, 768, (128//32)*(96//32)).permute(0,2,1)], dim=1)
                msc_aligned_feats, _, aux_loss, i_experts_loss = self.adapter(msc_feats[:,:1,:], target_expert=0)
                moe_loss += aux_loss
                experts_loss += i_experts_loss
                num_aux_loss += 1
                aligned_cls_feats.append(msc_aligned_feats)
                
        img_level_aligned_tokens = aligned_cls_feats
        outputs['aligned_cls_feats'] = img_level_aligned_tokens
        outputs['aligned_patch_feats'] = aligned_patch_feats
        outputs['qkv_atten'] =  None
        moe_loss /= num_aux_loss
        experts_loss /= num_aux_loss
        return outputs, moe_loss, experts_loss

######################################## Student Model ######################################## 
########## vit_tiny
def DPAL_kd_vit_tiny_patch16_moe(**kwargs): 
    model = DPALKDViTMOEWrapper(patch_size=(16, 16), embed_dim=192, depth=12, 
        num_heads=6, num_heads_in_last_block=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        align_num_heads=16, out_dim=768, **kwargs)
    return model

def DPAL_kd_vit_tiny_patch16_moe_path_large(**kwargs): 
    print('saipv1_kd')
    model = DPALKDViTMOEWrapper(patch_size=(16, 16), embed_dim=192, depth=12, 
        num_heads=6, num_heads_in_last_block=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        align_num_heads=16, out_dim=1024, **kwargs)
    return model

########## vit_small
def DPAL_kd_vit_small_patch16_moe(**kwargs): 
    model = DPALKDViTMOEWrapper(patch_size=(16, 16), embed_dim=384, depth=12, 
        num_heads=6, num_heads_in_last_block=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        align_num_heads=16, out_dim=768, **kwargs)
    return model

########## vit_base
def DPAL_kd_vit_base_patch16_moe_path_large(**kwargs): 
    model = DPALKDViTMOEWrapper(patch_size=(16, 16), embed_dim=768, depth=12, 
        num_heads=12, num_heads_in_last_block=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        align_num_heads=16, out_dim=1024, **kwargs)
    return model

########## swin_tiny
def DPAL_kd_swin_tiny_patch16_moe(**kwargs):
    model = DPALKDSWINMOEWrapper(backbone = "swin_tiny", embed_dim=384, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        align_num_heads=16, **kwargs)
    return model

########## swin_small
def DPAL_kd_swin_small_patch16_moe(**kwargs):
    model = DPALKDSWINMOEWrapper(backbone = "swin_small", embed_dim=192, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        align_num_heads=16, **kwargs)
    return model