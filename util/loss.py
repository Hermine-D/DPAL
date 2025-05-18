import torch

import torch.nn.functional as F
import torch.nn as nn


class CSMKDLoss(nn.Module):
    def __init__(self, ncrops):
        super().__init__()
        self.ncrops = ncrops

    def forward(self, s_feats, s_feats_patch, s_atten, t_feats, t_feats_patch,t_atten, nimages, ncrops):
      
        s_feats = s_feats.chunk(ncrops)
        s_feats_patch = s_feats_patch.chunk(nimages)
        t_feats = t_feats.detach().chunk(nimages)
        t_feats_patch = t_feats_patch.detach().chunk(nimages)

        s_qk_atten, s_vv_atten = s_atten
        s_qk_atten = s_qk_atten.chunk(nimages)
        s_vv_atten = s_vv_atten.chunk(nimages)
        
        t_qk_atten, t_vv_atten = t_atten
        t_qk_atten = t_qk_atten.detach().chunk(nimages)
        t_vv_atten = t_vv_atten.detach().chunk(nimages)

        rep_sim_loss = 0
        n_rep_loss_terms = 0
        att_sim_loss = 0
        n_att_loss_terms = 0
        for iq, q in enumerate(t_feats):
            for v in range(len(s_feats)):
                if v < 2 and v == iq:
                    # print(s_qk_atten[v].shape, t_qk_atten[iq].shape)
                    i_s_qk_atten = s_qk_atten[v].log()
                    i_s_vv_atten = s_vv_atten[v].log()
                    if s_qk_atten[v].shape != t_qk_atten[iq].shape:
                        i_s_qk_atten = F.interpolate(i_s_qk_atten, size=t_qk_atten[iq].shape[-2:])
                        i_s_vv_atten = F.interpolate(i_s_vv_atten, size=t_qk_atten[iq].shape[-2:])
                    qk_loss = nn.KLDivLoss(reduction="none")(i_s_qk_atten, t_qk_atten[iq]).sum(-1)
                    vv_loss = nn.KLDivLoss(reduction="none")(i_s_vv_atten, t_vv_atten[iq]).sum(-1)
                    att_sim_loss += (qk_loss.mean() + vv_loss.mean())
                    n_att_loss_terms += 1

                    norm_s = torch.nn.functional.normalize(s_feats_patch[v], dim=-1)
                    norm_t = torch.nn.functional.normalize(t_feats_patch[iq], dim=-1)
                    rep_sim_loss += torch.mean((-(norm_s * norm_t).sum(dim=-1)))
                    n_rep_loss_terms += 1
                else:
                    norm_s = torch.nn.functional.normalize(s_feats[v],dim=-1)
                    norm_t = torch.nn.functional.normalize(q, dim=-1)
                    rep_sim_loss += torch.mean((-(norm_s * norm_t).sum(dim=-1)))
                    n_rep_loss_terms += 1
        rep_sim_loss /= n_rep_loss_terms
        att_sim_loss /= n_att_loss_terms
        
        return {'align_rep_loss':rep_sim_loss,  'align_att_loss':att_sim_loss}

class TheiaLoss(nn.Module):
    def __init__(self, ncrops):
        super().__init__()
        self.ncrops = ncrops
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.SmoothL1Loss()
        self.cos_loss = nn.CosineEmbeddingLoss()
        self.cos_target = torch.ones((1), dtype=torch.int, requires_grad=False)
        self.target_loss_weights = 1
    
    def forward(self, s_feats, s_feats_patch, t_feats, t_feats_patch, nimages, ncrops):
        
        s_feats = s_feats.chunk(ncrops)
        s_feats_patch = s_feats_patch.chunk(nimages)
        t_feats = t_feats.detach().chunk(ncrops-nimages)
        # t_feats_patch = t_feats_patch.detach().chunk(nimages)
        
        mse_loss, cos_loss, l1_loss, n_loss = 0,0,0,0
        target = self.cos_target.repeat(s_feats[0].size(0)).to(s_feats[0].device)
        # For global images, compute patch loss
        for i, per_s_feats_patch in enumerate(s_feats_patch):
            mse_loss += self.mse_loss(per_s_feats_patch, t_feats_patch)
            
            l1_loss += self.l1_loss(per_s_feats_patch, t_feats_patch)
            
            per_s_feats_patch_norm = F.normalize(per_s_feats_patch.flatten(start_dim=1), dim=1, p=2)
            t_feats_patch_norm = F.normalize(t_feats_patch.flatten(start_dim=1), dim=1, p=2)
            cos_loss += self.cos_loss(per_s_feats_patch_norm, t_feats_patch_norm, target)
            
            n_loss += 1
            
        # For local images, compute class loss
        for i, per_s_feats in enumerate(s_feats):
            if i < 2:
                continue
            else:
                for j, per_t_feats in enumerate(t_feats):
                    mse_loss += self.mse_loss(per_s_feats, per_t_feats)
                
                    l1_loss += self.l1_loss(per_s_feats, per_t_feats)
                    
                    per_s_feats_norm = F.normalize(per_s_feats.flatten(start_dim=1), dim=1, p=2)
                    per_t_feats_norm = F.normalize(per_t_feats.flatten(start_dim=1), dim=1, p=2)
                    cos_loss += self.cos_loss(per_s_feats_norm, per_t_feats_norm, target)
                    n_loss += 1
        
        mse_loss_avg = mse_loss/n_loss
        cos_loss_avg = cos_loss/n_loss
        l1_loss_avg = l1_loss/n_loss
        
        return {
            "mse_loss": mse_loss_avg,
            "cos_loss": cos_loss_avg,
            "l1_loss": l1_loss_avg,
            # "mse_losses_per_model": mse_losses_per_model,
            # "cos_losses_per_model": cos_losses_per_model,
            # "l1_losses_per_model": l1_losses_per_model,
        }
        
class TheiaLossV2(nn.Module):
    def __init__(self, ncrops):
        super().__init__()
        self.ncrops = ncrops
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.SmoothL1Loss()
        self.cos_loss = nn.CosineEmbeddingLoss()
        self.cos_target = torch.ones((1), dtype=torch.int, requires_grad=False)
        self.target_loss_weights = 1
    
    def forward(self, s_feats_patch, t_feats_patch, nimages, ncrops):
        
        mse_loss, cos_loss, l1_loss = 0,0,0
        target = self.cos_target.repeat(s_feats_patch.size(0)).to(s_feats_patch.device)
        # For global images, compute patch loss
        mse_loss += self.mse_loss(s_feats_patch, t_feats_patch)
        
        l1_loss += self.l1_loss(s_feats_patch, t_feats_patch)
        
        s_feats_patch_norm = F.normalize(s_feats_patch.flatten(start_dim=1), dim=1, p=2)
        t_feats_patch_norm = F.normalize(t_feats_patch.flatten(start_dim=1), dim=1, p=2)
        cos_loss += self.cos_loss(s_feats_patch_norm, t_feats_patch_norm, target)
        
        mse_loss_avg = mse_loss
        cos_loss_avg = cos_loss
        l1_loss_avg = l1_loss
        
        return {
            "mse_loss": mse_loss_avg,
            "cos_loss": cos_loss_avg,
            "l1_loss": l1_loss_avg,
            # "mse_losses_per_model": mse_losses_per_model,
            # "cos_losses_per_model": cos_losses_per_model,
            # "l1_losses_per_model": l1_losses_per_model,
        }