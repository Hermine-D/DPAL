from functools import partial

import torch
import torch.nn as nn
import random
from vision_transformer import Block
from pretrain_decoder import CSSDecoder, CSRDecoder, CSMDecoder, AlignDecoder, ViTAdapter, interpolate_pos_encoding, ViTLarge2SmallAdapter, LightConvFeatureTranslator, LightConvFeatureTranslatorV2, LightConvFeatureTranslatorV3, CrossAttenAdapter
from pretrain_moe_decoder import LossMOEDecoderV3, LossMOEDecoderV2
from tinyvit import tinyvit_5m
from mmpretrain.models import VisionTransformer
from unihcp import ViT as unihcp_vit
from einops import rearrange
from dinov2_vit import DinoVisionTransformer
from dinov2_layers import MemEffAttention
from dinov2_layers import NestedTensorBlock

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



class SAIPViT(MaskedVisionTransformer):
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
    
    def forward(self, x, mask_ratio, with_anchor_mask=False, get_intermediate_only=False):
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
        
        return _outs, mask, ids_restore, attn

class SAIPViTv2(SAIPViT):
    
    def __init__(self, **args):
        super().__init__(**args)

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

class CSLViTWrapper(nn.Module):
    
    def __init__(self, **args):
        super(CSLViTWrapper, self).__init__()
        self.norm_pix_loss = args['norm_pix_loss']
        self.scales = [0.75, 0.875, 1.125, 1.25]#, 1.5]
        self.backbone = SAIPViT(**args)
        self.csr_decoder = CSRDecoder(num_patches=self.backbone.anchor_patches, **args)
        self.csm_decoder = CSMDecoder(**args)
        self.css_decoder = CSSDecoder(**args)
        self.eps=1e-3
    
    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.backbone.patch_embed.patch_size[0]
        # assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0

        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x, hw_size):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.backbone.patch_embed.patch_size[0]
        h, w = hw_size
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, w * p))
        return imgs
    
    def forward(self, base_imgs, meta, is_student=True):
        
        bs = base_imgs.shape[0]
        target_s = random.choice(self.scales)
        ref_inputs = meta['ref_img']
        css_mask = meta['gt_mask']
        msc_imgs = meta['region_imgs']
        num_msc_imgs = len(msc_imgs)
        # msc_imgs_pre_half = msc_imgs[:num_msc_imgs//2]
        # msc_imgs_last_half = msc_imgs[num_msc_imgs//2:]

        base_inputs = torch.cat([base_imgs, meta['aug_img']])
        # r_cl_inputs = torch.cat(region_imgs)

        ph, pw = ref_inputs.size(2)//self.backbone.patch_size[0], ref_inputs.size(3)//self.backbone.patch_size[1]
        b_ph, b_pw = base_imgs.shape[2]//self.backbone.patch_size[0], base_imgs.shape[3]//self.backbone.patch_size[1]
        css_inst_mask = torch.nn.functional.interpolate(css_mask[0].unsqueeze(1), (ph, pw))
        css_roi_mask = torch.nn.functional.interpolate(css_mask[1].unsqueeze(1), (ph, pw))
        outputs = {}

        if is_student:

            b_feats, _, _, last_atten = self.backbone(base_inputs, 0.)
            
            # Image level learning
            # 1.1 learning invariant representation under cross scales
            # base_local_feats, _, _, _ = self.backbone(torch.cat(msc_imgs_pre_half), 0.)
            msc_outs = []
            base_local_feats, _, _, _ = self.backbone(torch.cat(msc_imgs[:num_msc_imgs-2]), 0.)
            msc_outs.append(base_local_feats[1][:, 0, :])
            for i, s_img in enumerate(msc_imgs):
                if i < num_msc_imgs-2:
                    continue
                target_s = random.choice(self.scales)
                csi_inputs = torch.nn.functional.interpolate(s_img, scale_factor=target_s)
                
                csi_feats, _, _, _ = self.backbone(csi_inputs, 0.)
                msc_outs.append(csi_feats[1][:, 0, :])

                if i == len(msc_imgs)-2:
                    css_region_feats = csi_feats[0]

            img_level_tokens = torch.cat([b_feats[1][:, 0, :]]+msc_outs)
            csm_outs = self.csm_decoder(img_level_tokens)
            outputs['csm_preds'] = csm_outs

            # Pixel level learning:
            # 2.1 learning to reconstruct masked details at another scale 
            csr_inputs = torch.nn.functional.interpolate(meta['aug_img'], scale_factor=1.25)
            c_ph, c_pw = csr_inputs.shape[2]//self.backbone.patch_size[0], csr_inputs.shape[3]//self.backbone.patch_size[1]

            masked_feats, mask, ids_restore, _ = self.backbone(csr_inputs, meta['mask_ratio'], with_anchor_mask=True, get_intermediate_only=True)
            csr_outs = self.csr_decoder(b_feats[0][:bs], masked_feats[0], ids_restore)

            csr_target = self.patchify(csr_inputs)
            if self.norm_pix_loss:
                mean = csr_target.mean(dim=-1, keepdim=True)
                var = csr_target.var(dim=-1, keepdim=True)
                csr_target = (csr_target - mean) / (var + 1.e-6)**.5

            csr_loss = (csr_outs - csr_target) ** 2
            csr_loss = csr_loss.mean(dim=-1)  # [N, L], mean loss per patch

            csr_loss = (csr_loss * mask[1]).sum() / mask[1].sum()  # mean loss on removed patches
            outputs['csr_preds'] = self.unpatchify(csr_outs, (c_ph, c_pw))
            outputs['csr_loss'] = csr_loss
            
            # 2.2 learning to match instance to reference region under cross scale setting
            # features of reference region come from 'teacher' model 
            r_feats = meta['ref_feats']
            inter_b_feats = b_feats[0]
            _r_feats = r_feats[:, 1:, :].view(bs, ph, pw, -1).permute(0, 3, 1, 2).contiguous()

            css_inst_preds, _ = self.css_decoder(inter_b_feats[:bs], _r_feats)
            css_inst_outs = torch.sigmoid(css_inst_preds.flatten(1))
            css_inst_mask = css_inst_mask.flatten(1)
            
            css_inst_loss = torch.nn.functional.binary_cross_entropy_with_logits(css_inst_preds.reshape(-1), css_inst_mask.reshape(-1), reduction='mean')

            css_roi_preds, _ = self.css_decoder(css_region_feats, _r_feats)
            css_roi_outs = torch.sigmoid(css_roi_preds.flatten(1))
            css_roi_mask = css_roi_mask.flatten(1)
            css_roi_loss = torch.nn.functional.binary_cross_entropy_with_logits(css_roi_preds.reshape(-1), css_roi_mask.reshape(-1), reduction='mean')
            
            outputs['css_inst_loss'] = css_inst_loss
            outputs['css_roi_loss'] = css_roi_loss
            outputs['css_inst_preds'] = css_inst_outs.reshape(-1, ph, pw)
            outputs['css_roi_preds'] = css_roi_outs.reshape(-1, ph, pw)
            outputs['last_atten'] = last_atten[0][:bs, :, 0, 1:].view(bs, -1, b_ph, b_pw)
            outputs['qkv_atten'] =  last_atten

        else:
            
            r_feats, _, _, _ = self.backbone(ref_inputs, 0.)
            
            b_feats, _, _, last_atten = self.backbone(base_inputs, 0.)

            img_level_tokens = b_feats[1][:, 0, :]
            csm_outs = self.csm_decoder(img_level_tokens)

            outputs['ref_feats'] = r_feats[0]
            outputs['csm_preds'] = csm_outs
            outputs['qkv_atten'] =  last_atten
        
        return outputs

class CSRSViTWrapper(nn.Module):
    """
    CSS+CSR
    """
    def __init__(self, **args):
        super(CSRSViTWrapper, self).__init__()
        self.norm_pix_loss = args['norm_pix_loss']
        self.scales = [0.75, 0.875, 1.125, 1.25]#, 1.5]
        self.backbone = SAIPViT(**args)
        self.csr_decoder = CSRDecoder(num_patches=self.backbone.anchor_patches, **args)
        self.dino_decoder = CSMDecoder(**args)
        self.eps=1e-3
    
    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.backbone.patch_embed.patch_size[0]
        # assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0

        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x, hw_size):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.backbone.patch_embed.patch_size[0]
        h, w = hw_size
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, w * p))
        return imgs
    
    def forward(self, base_imgs, meta, is_student=True):
        
        bs = base_imgs.shape[0]
        target_s = random.choice(self.scales)
        msc_imgs = meta['region_imgs']
        num_msc_imgs = len(msc_imgs)

        base_inputs = torch.cat([base_imgs, meta['aug_img']])

        b_ph, b_pw = base_imgs.shape[2]//self.backbone.patch_size[0], base_imgs.shape[3]//self.backbone.patch_size[1]
        
        outputs = {}

        if is_student:

            b_feats, _, _, last_atten = self.backbone(base_inputs, 0.)
            
            # Image level learning
            # 1.1 learning invariant representation under cross scales
            msc_outs = []
            base_local_feats, _, _, _ = self.backbone(torch.cat(msc_imgs[:num_msc_imgs-2]), 0.)
            msc_outs.append(base_local_feats[1][:, 0, :])
            for i, s_img in enumerate(msc_imgs):
                if i < num_msc_imgs-2:
                    continue
                target_s = random.choice(self.scales)
                csi_inputs = torch.nn.functional.interpolate(s_img, scale_factor=target_s)
                
                csi_feats, _, _, _ = self.backbone(csi_inputs, 0.)
                msc_outs.append(csi_feats[1][:, 0, :])

            img_level_tokens = torch.cat([b_feats[1][:, 0, :]]+msc_outs)
            dino_outs = self.dino_decoder(img_level_tokens)
            outputs['dino_preds'] = dino_outs

            # Pixel level learning:
            # 2.1 learning to reconstruct masked details at another scale 
            csr_inputs = torch.nn.functional.interpolate(meta['aug_img'], scale_factor=1.25)
            c_ph, c_pw = csr_inputs.shape[2]//self.backbone.patch_size[0], csr_inputs.shape[3]//self.backbone.patch_size[1]

            masked_feats, mask, ids_restore, _ = self.backbone(csr_inputs, meta['mask_ratio'], with_anchor_mask=True, get_intermediate_only=True)
            csr_outs = self.csr_decoder(b_feats[0][:bs], masked_feats[0], ids_restore)

            csr_target = self.patchify(csr_inputs)
            if self.norm_pix_loss:
                mean = csr_target.mean(dim=-1, keepdim=True)
                var = csr_target.var(dim=-1, keepdim=True)
                csr_target = (csr_target - mean) / (var + 1.e-6)**.5

            csr_loss = (csr_outs - csr_target) ** 2
            csr_loss = csr_loss.mean(dim=-1)  # [N, L], mean loss per patch

            csr_loss = (csr_loss * mask[1]).sum() / mask[1].sum()  # mean loss on removed patches
            outputs['csr_preds'] = self.unpatchify(csr_outs, (c_ph, c_pw))
            outputs['csr_loss'] = csr_loss
            
            
            outputs['last_atten'] = last_atten[0][:bs, :, 0, 1:].view(bs, -1, b_ph, b_pw)
            outputs['qkv_atten'] =  last_atten

        else:
            
            
            b_feats, _, _, last_atten = self.backbone(base_inputs, 0.)


            img_level_tokens = b_feats[1][:, 0, :] 
            dino_outs = self.dino_decoder(img_level_tokens)

            outputs['dino_preds'] = dino_outs
            outputs['qkv_atten'] =  last_atten
        
        return outputs
    
class KDCSLViTWrapper(nn.Module):
   
    def __init__(self, **args):
        super(KDCSLViTWrapper, self).__init__()
        self.scales = [0.75, 0.875, 1, 1.125]
        self.backbone = SAIPViT(**args)
        self.adapter = AlignDecoder(**args)
    
    def forward(self, base_imgs, meta):
        
   
        target_s = random.choice(self.scales)
        msc_imgs = meta['region_imgs']
        num_msc_imgs = len(msc_imgs)

        base_inputs = torch.cat([base_imgs, meta['aug_img']])
        
        outputs = {}

        b_feats, _, _, last_atten = self.backbone(base_inputs, 0.)
        
        # Image level learning
        # 1.1 learning invariant representation under cross scales
        msc_outs = [b_feats[1][:, 0, :]]
        base_local_feats, _, _, _ = self.backbone(torch.cat(msc_imgs[:num_msc_imgs-2]), 0.)
        msc_outs.append(base_local_feats[1][:, 0, :])
        for i, s_img in enumerate(msc_imgs):
            if i < num_msc_imgs-2:
                continue
            target_s = random.choice(self.scales)
            csi_inputs = torch.nn.functional.interpolate(s_img, scale_factor=target_s)
            
            csi_feats, _, _, _ = self.backbone(csi_inputs, 0.)
            msc_outs.append(csi_feats[1][:, 0, :])

        img_level_tokens = torch.cat(msc_outs)
        align_feats = self.adapter(img_level_tokens)
        outputs['align_feats'] = align_feats
        outputs['qkv_atten'] =  last_atten
        
        return outputs

class SAIPKDViTWrapper(nn.Module):
   
    def __init__(self, **args):
        super(SAIPKDViTWrapper, self).__init__()
        self.backbone = SAIPViT(**args)
        self.adapter = AlignDecoder(**args) #ViTAdapter(**args)
    
    def forward(self, base_imgs, meta):
        
        msc_imgs = meta['region_imgs']

        base_inputs = torch.cat([base_imgs, meta['aug_img']])
        
        outputs = {}

        b_feats, _, _, aligned_atten = self.backbone(base_inputs, 0.)
        # b_aligned_feats, aligned_atten = self.adapter(b_feats[1])
        b_aligned_feats = b_feats[1]
        b_aligned_feats = self.adapter(b_aligned_feats)
        
        aligned_cls_feats = [b_aligned_feats[:, 0, :]]
        aligned_patch_feats = b_aligned_feats[:, 1:, :]
        msc_feats, _, _, _ = self.backbone(torch.cat(msc_imgs), 0.)
        # msc_aligned_feats, _ = self.adapter(msc_feats[1])
        msc_aligned_feats = msc_feats[1]
        msc_aligned_feats = self.adapter(msc_aligned_feats, is_msc=True)
        aligned_cls_feats.append(msc_aligned_feats[:, 0, :])
        
        img_level_aligned_tokens = torch.cat(aligned_cls_feats)
        outputs['aligned_cls_feats'] = img_level_aligned_tokens
        outputs['aligned_patch_feats'] = aligned_patch_feats
        outputs['qkv_atten'] =  aligned_atten
        
        return outputs

class SAIPKDTinyViTWrapper(nn.Module):
   
    def __init__(self, **args):
        super(SAIPKDTinyViTWrapper, self).__init__()
        self.patch_size = args['patch_size']
        self.backbone = tinyvit_5m(**args)
        self.adapter = nn.Linear(192, 1024) #AlignDecoder(**args) #ViTAdapter(**args)
    
    def forward(self, base_imgs, meta):
        
        bs, _, bh, bw = base_imgs.shape
        msc_imgs = meta['region_imgs']

        base_inputs = torch.cat([base_imgs, meta['aug_img']])

        outputs = {}

        b_cls_token, b_feats = self.backbone(base_inputs) # 这里的b_cls_token是avgpool获得的，并不是真的cls_token
        B,D,H,W = b_feats[-1].shape
        b_aligned_feats = torch.cat([b_cls_token.unsqueeze(1), b_feats[-1].reshape(B,D,H*W).permute(0,2,1).contiguous()], dim=1)
        b_aligned_feats = self.adapter(b_aligned_feats)

        aligned_cls_feats = [b_aligned_feats[:, 0, :]]
        aligned_patch_feats = b_aligned_feats[:, 1:, :]
        if msc_imgs:
            ms_cls_token, ms_feats = self.backbone(torch.cat(msc_imgs))
            B,D,H,W = ms_feats[-1].shape
            msc_aligned_feats = torch.cat([ms_cls_token.unsqueeze(1), ms_feats[-1].reshape(B,D,H*W).permute(0,2,1).contiguous()], dim=1)
            msc_aligned_feats = self.adapter(msc_aligned_feats)
            aligned_cls_feats.append(msc_aligned_feats[:, 0, :])

        img_level_aligned_tokens = torch.cat(aligned_cls_feats)
        outputs['aligned_cls_feats'] = img_level_aligned_tokens
        outputs['aligned_patch_feats'] = aligned_patch_feats
        outputs['qkv_atten'] =  None
        
        return outputs

class SAIPKDViTMOEV2Wrapper(nn.Module):
   
    def __init__(self, **args):
        super(SAIPKDViTMOEV2Wrapper, self).__init__()
        self.backbone = SAIPViT(**args)
        self.adapter = LossMOEDecoderV2(**args) #ViTAdapter(**args) # DeepSeekMOEDecoder(**args)
    
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
        ########## 单人大图 cls + atten + patch
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
        ########## 多人大图 atten + patch
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
        ########## 单人小图 cls
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
                # aligned_cls_feats.extend(list(torch.chunk(msc_aligned_feats, len_msc_imgs, 0)))
                
        img_level_aligned_tokens = aligned_cls_feats
        outputs['aligned_cls_feats'] = img_level_aligned_tokens
        outputs['aligned_patch_feats'] = aligned_patch_feats
        outputs['qkv_atten'] =  aligned_attens
        moe_loss /= num_aux_loss
        experts_loss /= num_aux_loss
        return outputs, moe_loss, experts_loss
    
    def forward_v1(self, base_imgs, meta):
        B,_,_,_ = base_imgs.shape
        msc_imgs = meta['region_imgs']

        if meta['aug_img'] is not None:
            base_inputs = torch.cat([base_imgs, meta['aug_img']])
            len_base_inputs = 2
        else:
            base_inputs = base_imgs
            len_base_inputs = 1
        
        outputs = {}
        moe_loss = 0
        num_aux_loss = 0
        # 单人大图 cls + atten + patch
        b_feats, _, _, aligned_atten = self.backbone(base_inputs, 0.)
        b_aligned_feats = b_feats[1]
        b_aligned_feats, _, aux_loss = self.adapter(b_aligned_feats, update_pattern_queries=True, target_expert=0)
        
        moe_loss += aux_loss
        num_aux_loss += 1
        
        aligned_cls_feats = list(torch.chunk(b_aligned_feats[:,0,:], len_base_inputs, dim=0))
        aligned_patch_feats = list(torch.chunk(b_aligned_feats[:,1:,:], len_base_inputs, dim=0))
        qk_atten = torch.chunk(aligned_atten[0], len_base_inputs, dim=0)
        vv_atten = torch.chunk(aligned_atten[1], len_base_inputs, dim=0)
        aligned_attens = []
        for i in range(len_base_inputs):
            aligned_attens.append([qk_atten[i], vv_atten[i]])
        # 多人大图 atten + patch
        b_feats, _, _, aligned_atten = self.backbone(meta['crowd_imgs'], 0.)
        b_aligned_feats = b_feats[1]
        b_aligned_feats, _, aux_loss = self.adapter(b_aligned_feats, target_expert=1)
        
        moe_loss += aux_loss
        num_aux_loss += 1
        
        aligned_cls_feats.append(b_aligned_feats[:, 0, :])
        aligned_patch_feats.append(b_aligned_feats[:, 1:, :])
        aligned_attens.append(aligned_atten)
        # 单人小图 cls
        if msc_imgs:
            len_msc_imgs = len(msc_imgs)
            msc_feats, _, _, aligned_atten = self.backbone(torch.cat(msc_imgs), 0.)
            msc_aligned_feats = msc_feats[1]
            msc_aligned_feats, _, aux_loss = self.adapter(msc_aligned_feats, is_msc=True, target_expert=2)
            
            moe_loss += aux_loss
            num_aux_loss += 1
            
            aligned_cls_feats.extend(list(torch.chunk(msc_aligned_feats[:, 0, :], len_msc_imgs, 0)))
            aligned_patch_feats.extend(list(torch.chunk(msc_aligned_feats[:, 1:, :], len_msc_imgs, 0)))
            qk_atten = torch.chunk(aligned_atten[0], len_msc_imgs, dim=0)
            vv_atten = torch.chunk(aligned_atten[1], len_msc_imgs, dim=0)
            for i in range(len_msc_imgs):
                aligned_attens.append([qk_atten[i], vv_atten[i]])
                
        img_level_aligned_tokens = aligned_cls_feats
        outputs['aligned_cls_feats'] = img_level_aligned_tokens
        outputs['aligned_patch_feats'] = aligned_patch_feats
        outputs['qkv_atten'] =  aligned_attens
        moe_loss /= num_aux_loss
        return outputs, moe_loss
    
    def forward_visu_expert(self, base_imgs):
        B,_,_,_ = base_imgs.shape
        
        # 多人大图
        b_feats, _, _, aligned_atten = self.backbone(base_imgs, 0.)
        # b_aligned_feats, aligned_atten = self.adapter(b_feats[1])
        b_aligned_feats = b_feats[1]
        _, topk = self.adapter(b_aligned_feats, return_topk=True)
        
        return topk

class DinoKDViTWrapper(nn.Module):
   
    def __init__(self, **args):
        super(DinoKDViTWrapper, self).__init__()    
        self.backbone = DinoVisionTransformer(img_size=224, patch_size=14, embed_dim=192, 
                                              depth=12, num_heads=12, mlp_ratio=4, block_fn=partial(NestedTensorBlock, attn_class=MemEffAttention), num_register_tokens=0, init_values=1.0, ffn_layer='mlp', block_chunks=0,
                                              interpolate_antialias=False, interpolate_offset=0.1)
        self.adapter = AlignDecoder(**args) #ViTAdapter(**args)
    
    def forward(self, base_imgs, meta):
        bs = base_imgs.shape[0]
        b_ph, b_pw = base_imgs.shape[2]//self.backbone.patch_embed.patch_size[0], base_imgs.shape[3]//self.backbone.patch_embed.patch_size[1]
        msc_imgs = meta['region_imgs']

        base_inputs = torch.cat([base_imgs, meta['aug_img']])
        
        outputs = {}

        b_feats = self.backbone.forward_features(base_inputs, None)
        aligned_atten = b_feats['attention']
        # b_aligned_feats, aligned_atten = self.adapter(b_feats[1])
        b_aligned_feats = torch.cat([b_feats['x_norm_clstoken'].unsqueeze(1),b_feats['x_norm_patchtokens']], dim=1)
        b_aligned_feats = self.adapter(b_aligned_feats)
        
        aligned_cls_feats = [b_aligned_feats[:, 0, :]]
        aligned_patch_feats = b_aligned_feats[:, 1:, :]
        aligned_patch_feats = aligned_patch_feats.reshape(bs, b_ph, b_pw, -1).permute(0, 3, 1, 2)

        msc_feats = self.backbone.forward_features(torch.cat(msc_imgs), None)
        # msc_aligned_feats, _ = self.adapter(msc_feats[1])
        msc_aligned_feats = torch.cat([msc_feats['x_norm_clstoken'].unsqueeze(1),msc_feats['x_norm_patchtokens']], dim=1)
        msc_aligned_feats = self.adapter(msc_aligned_feats)
        aligned_cls_feats.append(msc_aligned_feats[:, 0, :])
        
        img_level_aligned_tokens = torch.cat(aligned_cls_feats)
        outputs['aligned_cls_feats'] = img_level_aligned_tokens
        outputs['aligned_patch_feats'] = aligned_patch_feats
        outputs['qkv_atten'] =  aligned_atten
        
        return outputs

from swin_transformer import SwinTransformer
class SAIPKDSWINMOEV2Wrapper(nn.Module):
    def __init__(self, img_size, backbone="swin_tiny", **args):
        super(SAIPKDSWINMOEV2Wrapper, self).__init__()
        if backbone == "swin_tiny":
            self.backbone = SwinTransformer(pretrain_img_size = img_size, patch_size=4, window_size=7, embed_dims=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), drop_path_rate=0.1, drop_rate=0.0, attn_drop_rate=0.0)
        elif backbone == "swin_small":
            self.backbone = SwinTransformer(pretrain_img_size = img_size, patch_size=4, window_size=7, embed_dims=96, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24), drop_path_rate=0.1, drop_rate=0.0, attn_drop_rate=0.0)
        self.adapter = LossMOEDecoderV3(**args) #ViTAdapter(**args) # DeepSeekMOEDecoder(**args)
    
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
        ########## 单人大图 cls + atten + patch
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
        # ########## 多人大图 atten + patch
        # b_feats = self.backbone(meta['crowd_imgs'])
        # # patch
        # b_aligned_feats, _, aux_loss, i_experts_loss = self.adapter(b_feats[:,1:,:], target_expert=2)
        # moe_loss += aux_loss
        # experts_loss += i_experts_loss
        # num_aux_loss += 1
        # aligned_patch_feats.append(b_aligned_feats)        
        ########## 单人小图 cls
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
                # aligned_cls_feats.extend(list(torch.chunk(msc_aligned_feats, len_msc_imgs, 0)))
                
        img_level_aligned_tokens = aligned_cls_feats
        outputs['aligned_cls_feats'] = img_level_aligned_tokens
        outputs['aligned_patch_feats'] = aligned_patch_feats
        outputs['qkv_atten'] =  None
        moe_loss /= num_aux_loss
        experts_loss /= num_aux_loss
        return outputs, moe_loss, experts_loss
    
class SAIPKDViTNoClsAttenWrapper(nn.Module):    

    def __init__(self, **args):
        super(SAIPKDViTNoClsAttenWrapper, self).__init__()
        self.backbone = SAIPViTv2(**args)
        self.adapter = ViTAdapter(**args)

    def forward(self, base_imgs, meta, with_cls_token=False, extract_regions=True):
        
        msc_imgs = meta['region_imgs']
        if extract_regions:
            base_inputs = torch.cat([base_imgs, meta['aug_img']])
        else:
            base_inputs = base_imgs
        
        outputs = {}

        b_feats, _, _, _ = self.backbone(base_inputs, 0., with_cls_token=with_cls_token)
        b_aligned_feats, aligned_atten = self.adapter(b_feats[1])
        
        if with_cls_token:
            aligned_cls_feats = [b_aligned_feats[:,:1,:]]
            aligned_patch_feats = b_aligned_feats[:, 1:, :]
        else:
            aligned_cls_feats = [b_aligned_feats.mean(dim=1)]
            aligned_patch_feats = b_aligned_feats[:, 0:, :]

        if extract_regions:
            msc_feats, _, _, _ = self.backbone(torch.cat(msc_imgs), 0., with_cls_token=with_cls_token)
            msc_aligned_feats, _ = self.adapter(msc_feats[1])
            if with_cls_token:
                aligned_cls_feats.append(msc_aligned_feats[:,:1,1])
            else:
                aligned_cls_feats.append(msc_aligned_feats.mean(dim=1))
        
        img_level_aligned_tokens = torch.cat(aligned_cls_feats)
        outputs['aligned_cls_feats'] = img_level_aligned_tokens
        outputs['aligned_patch_feats'] = aligned_patch_feats
        
        # aligned_atten_no_cls = []
        # for value in aligned_atten:
        #     aligned_atten_no_cls.append(value[:,:,1:,1:])
        
        outputs['qkv_atten'] =  aligned_atten
        
        return outputs
    
class SAIPKDNoClsAttenWrapper(nn.Module): # adapter是简单的线性变化
   
    def __init__(self, **args):
        super(SAIPKDNoClsAttenWrapper, self).__init__()
        self.backbone = SAIPViTv2(**args)
        self.adapter = AlignDecoder(**args)
    
    def forward(self, base_imgs, meta, with_cls_token=False, extract_regions=True):
        
        msc_imgs = meta['region_imgs']
        if extract_regions:
            base_inputs = torch.cat([base_imgs, meta['aug_img']])
        else:
            base_inputs = base_imgs
        
        outputs = {}

        b_feats, _, _, _ = self.backbone(base_inputs, 0., with_cls_token=with_cls_token)
        B,N,D= b_feats[1].shape
        b_feats_adapter = rearrange(b_feats[1], 'b n d -> (b n) d')
        b_aligned_feats= self.adapter(b_feats_adapter)
        b_aligned_feats = rearrange(b_aligned_feats, '(b n) d -> b n d', b=B, n=N)
        
        if with_cls_token:
            aligned_cls_feats = [b_aligned_feats[:,:1,:]]
            aligned_patch_feats = b_aligned_feats[:, 1:, :]
        else:
            aligned_cls_feats = [b_aligned_feats.mean(dim=1)]
            aligned_patch_feats = b_aligned_feats[:, 0:, :]

        if extract_regions:
            msc_feats, _, _, _ = self.backbone(torch.cat(msc_imgs), 0., with_cls_token=with_cls_token)
            msc_aligned_feats, _ = self.adapter(msc_feats[1])
            if with_cls_token:
                aligned_cls_feats.append(msc_aligned_feats[:,:1,1])
            else:
                aligned_cls_feats.append(msc_aligned_feats.mean(dim=1))
        
        img_level_aligned_tokens = torch.cat(aligned_cls_feats)
        outputs['aligned_cls_feats'] = img_level_aligned_tokens
        outputs['aligned_patch_feats'] = aligned_patch_feats
        
        # aligned_atten_no_cls = []
        # for value in aligned_atten:
        #     aligned_atten_no_cls.append(value[:,:,1:,1:])
        
        # outputs['qkv_atten'] =  aligned_atten
        
        return outputs
    
class SAIPKDViTNoClsAttenLarge2SmallWrapper(nn.Module):    

    def __init__(self, **args):
        super().__init__()
        self.backbone = SAIPViTv2(**args)
        self.adapter = ViTLarge2SmallAdapter(**args)
        
    def forward(self, base_imgs, meta, with_cls_token=False, extract_regions=True, interpolate=False):
        
        msc_imgs = meta['region_imgs']
        if extract_regions:
            base_inputs = torch.cat([base_imgs, meta['aug_img']])
        else:
            base_inputs = base_imgs
        
        outputs = {}

        b_feats, _, _, _, student_resolution = self.backbone(base_inputs, 0., with_cls_token=with_cls_token, return_resolution=True)
        b_aligned_feats, aligned_atten = self.adapter(b_feats[1], student_resolution, interpolate)
        
        aligned_cls_feats = [b_aligned_feats.mean(dim=1)]
        aligned_patch_feats = b_aligned_feats[:, 0:, :]

        if extract_regions:
            msc_feats, _, _, _ = self.backbone(torch.cat(msc_imgs), 0., with_cls_token=with_cls_token)
            msc_aligned_feats, _ = self.adapter(msc_feats[1])
            aligned_cls_feats.append(msc_aligned_feats.mean(dim=1))
        
        img_level_aligned_tokens = torch.cat(aligned_cls_feats)
        outputs['aligned_cls_feats'] = img_level_aligned_tokens
        outputs['aligned_patch_feats'] = aligned_patch_feats
        
        # aligned_atten_no_cls = []
        # for value in aligned_atten:
        #     aligned_atten_no_cls.append(value[:,:,1:,1:])
        
        outputs['qkv_atten'] =  aligned_atten
        
        return outputs
    
class SAIPKDViTNoClsAttenEMAWrapper(SAIPKDViTNoClsAttenLarge2SmallWrapper):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, base_imgs, meta, with_cls_token=False, is_student=True, extract_regions=True):
        
        msc_imgs = meta['region_imgs']
        outputs = {}
        if is_student: # All images
            if extract_regions: 
                base_inputs = torch.cat([base_imgs, meta['aug_img']])
            else:
                base_inputs = base_imgs

            b_feats, _, _, _, student_resolution = self.backbone(base_inputs, 0., with_cls_token=with_cls_token, return_resolution=True)
            b_aligned_feats, aligned_atten = self.adapter(b_feats[1], student_resolution, interpolate=True)
            
            if with_cls_token:
                aligned_cls_feats = [b_aligned_feats[:,:1,:]]
                aligned_patch_feats = b_aligned_feats[:, 1:, :]
            else:
                aligned_cls_feats = [b_aligned_feats.mean(dim=1)]
                aligned_patch_feats = b_aligned_feats
            
            if extract_regions:
                msc_feats, _, _, _ = self.backbone(torch.cat(msc_imgs), 0., with_cls_token=with_cls_token)
                msc_aligned_feats, _ = self.adapter(msc_feats[1])
                if with_cls_token:
                    aligned_cls_feats.append(msc_aligned_feats[:,:1,:])
                else:
                    aligned_cls_feats.append(msc_aligned_feats.mean(dim=1))
                
            img_level_aligned_tokens = torch.cat(aligned_cls_feats)
            outputs['aligned_cls_feats'] = img_level_aligned_tokens
            outputs['aligned_patch_feats'] = aligned_patch_feats        
            outputs['qkv_atten'] =  aligned_atten
            
            return outputs
        else:
            msc_feats, _, _, _ = self.backbone(torch.cat(msc_imgs), 0., with_cls_token=with_cls_token)
            msc_aligned_feats, _ = self.adapter(msc_feats[1])
            if with_cls_token:
                aligned_cls_feats=msc_aligned_feats[:,:1,1]
            else:
                aligned_cls_feats=msc_aligned_feats.mean(dim=1)
                
            outputs['aligned_cls_feats'] = aligned_cls_feats
            
            return outputs
        
class SAIPKDViTNoClsAttenConvWrapper(nn.Module):
    def __init__(self, **args):
        super().__init__()
        self.backbone = SAIPViTv2(**args)
        # self.backbone = VisionTransformer(arch='deit-tiny',
        #     img_size=(1024,768),
        #     patch_size=16,
        #     out_type='raw',
        #     with_cls_token=False
        # )
        # self.adapter = LightConvFeatureTranslator(backbone_feature_size=(192,16,12),
        #                                           target_feature_sizes=(1024,64,48),
        #                                           translator_hidden_size=1024,
        #                                           hidden_size_factor=1.0)
        self.adapter = LightConvFeatureTranslatorV2(backbone_feature_size=(192,32,24),
                                                  target_feature_sizes=(1024,64,48),
                                                  translator_hidden_size=1024,
                                                  hidden_size_factor=1.0)
        # self.adapter = AlignDecoder(embed_dim=192, out_dim=1024, nlayers=3, hidden_dim=2048, bottleneck_dim=256)

    def forward(self, base_imgs, meta, with_cls_token=False, is_student=True, extract_regions=True):
        
        msc_imgs = meta['region_imgs']
        
        # ## 可视化
        # from torchvision.utils import save_image
        # mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        # std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        # unnormalized_tensor = base_imgs[0].cpu() * std + mean
        # unnormalized_tensor = torch.clamp(unnormalized_tensor, 0, 1)
        # save_image(unnormalized_tensor, 'temp/output_image.png')
        # ## END
        
        outputs = {}
        if is_student: # All images
            if extract_regions: 
                base_inputs = torch.cat([base_imgs, meta['aug_img']])
            else:
                base_inputs = base_imgs

            b_feats, _, _, _ = self.backbone(base_inputs, 0., with_cls_token=with_cls_token)
            b_aligned_feats = self.adapter(b_feats[1], backbone_no_cls=True)
            # b_feats = self.backbone(base_inputs)
            # b_aligned_feats = self.adapter(b_feats[0], backbone_no_cls=True)
            
            if with_cls_token:
                aligned_cls_feats = [b_aligned_feats[:,:1,:]]
                aligned_patch_feats = b_aligned_feats[:, 1:, :]
            else:
                aligned_cls_feats = [b_aligned_feats.mean(dim=1)]
                aligned_patch_feats = b_aligned_feats
            
            if extract_regions:
                msc_feats, _, _, _ = self.backbone(torch.cat(msc_imgs), 0., with_cls_token=with_cls_token)
                msc_aligned_feats, _ = self.adapter(msc_feats[1])
                if with_cls_token:
                    aligned_cls_feats.append(msc_aligned_feats[:,:1,:])
                else:
                    aligned_cls_feats.append(msc_aligned_feats.mean(dim=1))
                
            img_level_aligned_tokens = torch.cat(aligned_cls_feats)
            outputs['aligned_cls_feats'] = img_level_aligned_tokens
            outputs['aligned_patch_feats'] = aligned_patch_feats
            
            return outputs
        else:
            msc_feats, _, _, _ = self.backbone(torch.cat(msc_imgs), 0., with_cls_token=with_cls_token)
            msc_aligned_feats, _ = self.adapter(msc_feats[1])
            if with_cls_token:
                aligned_cls_feats=msc_aligned_feats[:,:1,1]
            else:
                aligned_cls_feats=msc_aligned_feats.mean(dim=1)
                
            outputs['aligned_cls_feats'] = aligned_cls_feats
            
            return outputs
        
class MMKDViTNoClsAttenWrapper(nn.Module):    

    def __init__(self, **args):
        super(MMKDViTNoClsAttenWrapper, self).__init__()
        self.backbone = unihcp_vit(**args)
        self.adapter = ViTAdapter(**args)

    def forward(self, base_imgs, meta, with_cls_token=False, extract_regions=True):
        
        msc_imgs = meta['region_imgs']
        if extract_regions:
            base_inputs = torch.cat([base_imgs, meta['aug_img']])
        else:
            base_inputs = base_imgs
        
        outputs = {}
        input = {}
        input['image'] = base_inputs
        b_feats = self.backbone(input)
        b_aligned_feats, aligned_atten = self.adapter(b_feats['backbone_output'][0])
        
        if with_cls_token:
            aligned_cls_feats = [b_aligned_feats[:,:1,:]]
            aligned_patch_feats = b_aligned_feats[:, 1:, :]
        else:
            aligned_cls_feats = [b_aligned_feats.mean(dim=1)]
            aligned_patch_feats = b_aligned_feats[:, 0:, :]

        if extract_regions:
            msc_feats, _, _, _ = self.backbone(torch.cat(msc_imgs), 0., with_cls_token=with_cls_token)
            msc_aligned_feats, _ = self.adapter(msc_feats[1])
            if with_cls_token:
                aligned_cls_feats.append(msc_aligned_feats[:,:1,1])
            else:
                aligned_cls_feats.append(msc_aligned_feats.mean(dim=1))
        
        img_level_aligned_tokens = torch.cat(aligned_cls_feats)
        outputs['aligned_cls_feats'] = img_level_aligned_tokens
        outputs['aligned_patch_feats'] = aligned_patch_feats
        
        # aligned_atten_no_cls = []
        # for value in aligned_atten:
        #     aligned_atten_no_cls.append(value[:,:,1:,1:])
        
        outputs['qkv_atten'] =  aligned_atten
        
        return outputs
    
class CSLTinyViTWrapper(nn.Module):
   
    def __init__(self, **args):
        super(CSLTinyViTWrapper, self).__init__()
        self.norm_pix_loss = args['norm_pix_loss']
        self.scales = [0.75, 0.875, 1, 1.125]#, 1.5]
        self.patch_size = args['patch_size']
        self.backbone = tinyvit_5m(**args)
        self.csm_decoder = CSMDecoder(embed_dim=self.backbone.final_dims, **args)
        self.css_decoder = CSSDecoder(embed_dim=self.backbone.stage_dims[-1], **args)
        self.eps=1e-3
    
    def forward(self, base_imgs, meta, is_student=True):
        
        bs, _, bh, bw = base_imgs.shape
        target_s = random.choice(self.scales)
        ref_inputs = meta['ref_img']
        csm_mask = meta['gt_mask']
        msc_imgs = meta['region_imgs']
        num_msc_imgs = len(msc_imgs)

        base_inputs = torch.cat([base_imgs, meta['aug_img']])

        ph, pw = ref_inputs.size(2)//self.patch_size[0], ref_inputs.size(3)//self.patch_size[1]
        
        csm_inst_mask = torch.nn.functional.interpolate(csm_mask[0].unsqueeze(1), (ph, pw))
        csm_roi_mask = torch.nn.functional.interpolate(csm_mask[1].unsqueeze(1), (ph, pw))
        outputs = {}

        if is_student:

            b_cls_token, b_feats = self.backbone(base_inputs)
            
            # Image level learning
            # 1.1 learning invariant representation under cross scales
            msc_outs = [b_cls_token]
            ms_cls_token, _ = self.backbone(torch.cat(msc_imgs[:num_msc_imgs-2]))
            msc_outs.append(ms_cls_token)
            for i, s_img in enumerate(msc_imgs):
                if i < num_msc_imgs-2:
                    continue
                target_s = random.choice(self.scales)
                # target_h, target_w = int(bh * target_s), int(bw * target_s)
                # csi_inputs = torch.nn.functional.interpolate(s_img, (target_h, target_w))
                csi_inputs = torch.nn.functional.interpolate(s_img, scale_factor=target_s)
                
                csi_cls_token, csi_feats = self.backbone(csi_inputs)
                msc_outs.append(csi_cls_token)

                if i == len(msc_imgs)-2:
                    csm_region_feats = csi_feats[-2]

            img_level_tokens = torch.cat(msc_outs)
            dino_outs = self.dino_decoder(img_level_tokens)
            outputs['dino_preds'] = dino_outs

            # Pixel level learning:
            
            # learning to match instance to reference region under cross scale setting
            # features of reference region come from 'teacher' model 
            r_feats = meta['ref_feats']
            inter_b_feats = b_feats[-2][:bs]
            _r_feats = r_feats.view(bs, ph, pw, -1).permute(0, 3, 1, 2).contiguous()

            csm_inst_preds, _ = self.csm_decoder(inter_b_feats, _r_feats)
            csm_inst_outs = torch.sigmoid(csm_inst_preds.flatten(1))
            csm_inst_mask = csm_inst_mask.flatten(1)
            
            csm_inst_loss = torch.nn.functional.binary_cross_entropy_with_logits(csm_inst_preds.reshape(-1), csm_inst_mask.reshape(-1), reduction='mean')

            csm_roi_preds, _ = self.csm_decoder(csm_region_feats, _r_feats)
            csm_roi_outs = torch.sigmoid(csm_roi_preds.flatten(1))
            csm_roi_mask = csm_roi_mask.flatten(1)
            csm_roi_loss = torch.nn.functional.binary_cross_entropy_with_logits(csm_roi_preds.reshape(-1), csm_roi_mask.reshape(-1), reduction='mean')
            
            outputs['css_inst_loss'] = csm_inst_loss
            outputs['css_roi_loss'] = csm_roi_loss
            outputs['css_inst_preds'] = csm_inst_outs.reshape(-1, ph, pw)
            outputs['css_roi_preds'] = csm_roi_outs.reshape(-1, ph, pw)

        else:
            
            _, r_feats = self.backbone(ref_inputs)
            
            b_cls_token, b_feats = self.backbone(base_inputs)

            dino_outs = self.dino_decoder(b_cls_token)

            outputs['ref_feats'] = r_feats[-2]
            outputs['dino_preds'] = dino_outs
        
        return outputs

class SAIPKDCrossAttenWrapper(nn.Module): # 只有backbone没有adapter
   
    def __init__(self, **args):
        super(SAIPKDCrossAttenWrapper, self).__init__()
        self.backbone = SAIPViT(**args)
        self.adapter = CrossAttenAdapter(**args)
    
    def forward(self, base_imgs, pred_t):
        outputs = {}

        b_feats, _, _, aligned_atten = self.backbone(base_imgs, 0.)
        b_aligned_feats = b_feats[1]
        
        # teacher
        query = torch.cat([pred_t["feats_from_teacher"].unsqueeze(1), pred_t["feats_from_teacher_patch"]], dim=1)
        b_after_adapter_all, b_after_adapter_atten = self.adapter(x=b_aligned_feats, query=query)
        
        aligned_cls_feats = b_after_adapter_all[:, 0, :]
        aligned_patch_feats = b_after_adapter_all[:, 1:, :]

        outputs['aligned_cls_feats'] = aligned_cls_feats
        outputs['aligned_patch_feats'] = aligned_patch_feats
        outputs['qkv_atten'] =  b_after_adapter_atten
        
        # backbone
        outputs['aligned_cls_feats_backbone'] = b_aligned_feats[:, 0, :]
        outputs['aligned_patch_feats_backbone'] = b_aligned_feats[:, 1:, :]
        outputs['qkv_atten_backbone'] =  aligned_atten
        
        return outputs
######################################## saip ######################################## 
def csl_vit_tiny_patch16(**kwargs):
    model = CSLViTWrapper(
        patch_size=(16, 16), embed_dim=192, depth=12, num_heads=3,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def csrs_vit_tiny_patch16(**kwargs):
    model = CSRSViTWrapper(
        patch_size=(16, 16), embed_dim=192, depth=12, num_heads=3,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def csl_kd_vit_tiny_patch16(**kwargs):
    model = KDCSLViTWrapper(patch_size=(16, 16), embed_dim=192, depth=12, 
        num_heads=6, num_heads_in_last_block=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def saip_kd_vit_tiny_patch16(**kwargs):
    model = SAIPKDViTWrapper(patch_size=(16, 16), embed_dim=192, depth=12, 
        num_heads=3, num_heads_in_last_block=3,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def saipv1_kd_vit_tiny_patch16(**kwargs): 
    print('saipv1_kd')
    model = SAIPKDViTWrapper(patch_size=(16, 16), embed_dim=192, depth=12, 
        num_heads=6, num_heads_in_last_block=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def saipv1_kd_vit_tiny_patch16_cross_atten(**kwargs): 
    print('saipv1_kd')
    model = SAIPKDCrossAttenWrapper(patch_size=(16, 16), embed_dim=192, depth=12, 
        num_heads=6, num_heads_in_last_block=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def saipv1_kd_vit_tiny_patch16_moeV2(**kwargs): 
    print('saipv1_kd')
    model = SAIPKDViTMOEV2Wrapper(patch_size=(16, 16), embed_dim=192, depth=12, 
        num_heads=6, num_heads_in_last_block=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        align_num_heads=16, **kwargs)
    return model

def saipv1_kd_vit_tiny_patch16_moeV2_path_large(**kwargs): 
    print('saipv1_kd')
    model = SAIPKDViTMOEV2Wrapper(patch_size=(16, 16), embed_dim=192, depth=12, 
        num_heads=6, num_heads_in_last_block=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        align_num_heads=16, **kwargs)
    return model

def saipv1_kd_vit_base_patch16_moeV2_path_large(**kwargs): 
    print('saipv1_kd')
    model = SAIPKDViTMOEV2Wrapper(patch_size=(16, 16), embed_dim=768, depth=12, 
        num_heads=12, num_heads_in_last_block=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        align_num_heads=16, **kwargs)
    return model

def saipv1_kd_vit_small_patch16_moeV2(**kwargs): 
    print('saipv1_kd')
    model = SAIPKDViTMOEV2Wrapper(patch_size=(16, 16), embed_dim=384, depth=12, 
        num_heads=6, num_heads_in_last_block=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        align_num_heads=16, **kwargs)
    return model

def saipv1_kd_swin_tiny_patch16_moeV2(**kwargs):
    model = SAIPKDSWINMOEV2Wrapper(patch_size=(16, 16), embed_dim=192, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        align_num_heads=16, **kwargs)
    return model

def saipv1_kd_swin_small_patch16_moeV2(**kwargs):
    model = SAIPKDSWINMOEV2Wrapper(backbone = "swin_small",patch_size=(16, 16), embed_dim=192, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        align_num_heads=16, **kwargs)
    return model

def saipv1_kd_tiny_vit_patch16(**kwargs): 
    print('saipv1_kd')
    model = SAIPKDTinyViTWrapper(patch_size=(16, 16), embed_dim=256,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def saipv1_kd_vit_tiny_patch14(**kwargs): 
    print('saipv1_kd')
    model = SAIPKDViTWrapper(patch_size=(14, 14), embed_dim=192, depth=12, 
        num_heads=6, num_heads_in_last_block=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def saipv1_kd_vit_base_patch16(**kwargs): 
    # print('saipv1_kd')
    model = SAIPKDViTWrapper(patch_size=(16, 16), embed_dim=192, depth=12, 
        num_heads=12, num_heads_in_last_block=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def csl_tinyvit_5m_patch16(**kwargs):
    model = CSLTinyViTWrapper(
        patch_size=(16, 16),
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

######################################## sapiens ########################################
def saip_kd_vit_tiny_patch16_sapiens_1b(**kwargs):
    model = SAIPKDViTNoClsAttenWrapper(img_size=(1024, 768), patch_size=(16, 16), embed_dim=192, depth=12, 
        num_heads=3, num_heads_in_last_block=3,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        align_dim=1536, align_num_heads=24,
        **kwargs)
    return model

def saip_kd_vit_tiny_patch16_sapiens_0_3b(**kwargs):
    model = SAIPKDViTNoClsAttenWrapper(img_size=(1024, 768), patch_size=(16, 16), embed_dim=192, depth=12, 
        num_heads=3, num_heads_in_last_block=3,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        align_dim=1024, align_num_heads=16,
        **kwargs)
    return model

def saip_kd_vit_tiny_patch16_sapiens_0_3b_large2small(**kwargs):
    model = SAIPKDViTNoClsAttenLarge2SmallWrapper(img_size=(1024, 768), patch_size=(16, 16), embed_dim=192, depth=12, 
        num_heads=3, num_heads_in_last_block=3,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        align_dim=1024, align_num_heads=16,
        **kwargs)
    return model

def saip_kd_vit_tiny_patch16_sapiens_0_3b_ema(**kwargs):
    model = SAIPKDViTNoClsAttenEMAWrapper(img_size=(1024, 768), patch_size=(16, 16), embed_dim=192, depth=12, 
        num_heads=3, num_heads_in_last_block=3,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        align_dim=1024, align_num_heads=16,
        **kwargs)
    return model

def saip_kd_vit_tiny_patch16_sapiens_0_3b_conv(**kwargs):
    model = SAIPKDViTNoClsAttenConvWrapper(img_size=(1024, 768), patch_size=(16, 16), embed_dim=192, depth=12, 
        num_heads=3, num_heads_in_last_block=3,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        align_dim=1024, align_num_heads=16,
        **kwargs)
    return model

######################################## unihcp ########################################
def saip_kd_vit_tiny_patch16_unihcp(**kwargs):
    model = SAIPKDViTNoClsAttenWrapper(patch_size=(16, 16), embed_dim=192, depth=12, 
        num_heads=6, num_heads_in_last_block=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        align_dim=768,
        **kwargs)
    return model

def saip_kd_vit_small_patch16_unihcp(**kwargs):
    model = SAIPKDViTNoClsAttenWrapper(img_size=(1344, 1344), patch_size=(16, 16), embed_dim=384, depth=12, 
        num_heads=6, num_heads_in_last_block=6,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        align_dim=768, align_num_heads=12,
        **kwargs)
    return model

def mm_kd_vit_tiny_patch16_unihcp(**kwargs):
    model = MMKDViTNoClsAttenWrapper(img_size=1344, patch_size=16, embed_dim=192, depth=12, 
        num_heads=3, drop_path_rate=0.2, use_abs_pos_emb=True,
        mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        task_sp_list=['rel_pos_h', 'rel_pos_w'],
        lms_checkpoint_train='fairscale', window=False, test_pos_mode='learnable_interpolate', learnable_pos=True,
        align_dim=768, align_num_heads=12,
        **kwargs)
    return model

######################################## dinov2 ########################################
def saip_kd_vit_tiny_patch16_dinov2(**kwargs):
    model = SAIPKDNoClsAttenWrapper(img_size=(224, 224), patch_size=(16, 16), embed_dim=192, depth=12, 
        num_heads=12, num_heads_in_last_block=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        out_dim=1024, nlayers=3, hidden_dim=2048, bottleneck_dim=256,
        **kwargs)
    return model
 

#####################################  PATH ################################################
def saip_kd_vit_tiny_patch16_pathlarge(**kwargs):
    model = SAIPKDNoClsAttenWrapper(img_size=(224, 224), patch_size=(16, 16), embed_dim=192, depth=12, 
        num_heads=12, num_heads_in_last_block=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        out_dim=1024, nlayers=3, hidden_dim=2048, bottleneck_dim=256,
        **kwargs)
    return model
