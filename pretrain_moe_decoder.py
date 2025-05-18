from turtle import up
from cv2 import exp
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block
from functools import partial
from vision_transformer import CrossAttentionBlock, Block as ViTBlock
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import math
from vision_transformer import Attention, DropPath

class MoEGate(nn.Module):
    def __init__(self, top_k, num_experts, dim, scoring_func="softmax", aux_loss_alpha=0.001, seq_aux=True, norm_topk_prob=True):
        super().__init__()
        self.top_k = top_k
        self.n_routed_experts = num_experts

        self.scoring_func = scoring_func
        self.alpha = aux_loss_alpha
        self.seq_aux = seq_aux

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.moe_tokens = nn.Parameter(torch.randn(1, 1, dim))
        self.cross_attn = nn.MultiheadAttention(dim, num_heads=12,batch_first=True)
        
        self.norm_topk_prob = norm_topk_prob
        self.gating_dim = dim
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states, target_expert, return_scores = False):
        bsz, seq_len, h = hidden_states.shape

        x = self.cross_attn(self.norm(self.moe_tokens.expand(bsz, -1, -1)), hidden_states, hidden_states)[0] # torch.Size([32, 1, 192])
        # x = self.moe_tokens.expand(bsz, -1, -1) + self.cross_attn(self.norm(self.moe_tokens.expand(bsz, -1, -1)), hidden_states, hidden_states)[0] # torch.Size([32, 1, 192])
        x = x.squeeze(1)
        logits = F.linear(x, self.weight, None) # torch.Size([32, 3])
        scores = logits.softmax(dim=-1) # torch.Size([32, 3])
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
        topk_weight = topk_weight / denominator

        target_labels = torch.full((bsz,), target_expert, dtype=torch.long).cuda() 
        aux_loss = F.cross_entropy(logits, target_labels)
        
        if return_scores:
            return topk_idx, topk_weight, aux_loss, scores
        return topk_idx, topk_weight, aux_loss

class AfterExpert(nn.Module):
    def __init__(self, out_dim=768):
        super().__init__()
        self.relu = nn.GELU()
        self.norm = nn.LayerNorm(out_dim, eps=1e-6)

    def forward(self, x):
        # x = self.mlp(x)
        x = self.norm(self.relu(x))
        return x

class LossMOEDecoder(nn.Module): # Three patterns
    def __init__(self, embed_dim, align_num_heads, out_dim=768, norm_layer=nn.LayerNorm, drop=0., drop_path = 0. , **args):
        super().__init__()
        num_experts, top_k = 3, 1
        
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.num_experts = num_experts
        self.top_k = top_k

        ############# 1
        self.norm1 = norm_layer(embed_dim)
        self.attn = Attention(
            embed_dim, num_heads=align_num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = norm_layer(embed_dim)
        self.gate = MoEGate(top_k, num_experts, embed_dim)
        self.after_experts = AfterExpert(out_dim)

        self.pattern_queries = nn.Parameter(torch.zeros(1, 3, embed_dim))
        self.norm3 = norm_layer(embed_dim)
        self.attn2 = Attention(
            embed_dim, num_heads=align_num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=drop)
        
        self.norm4 = norm_layer(embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=align_num_heads,batch_first=True)
        
        self.norm5 = norm_layer(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int((embed_dim) * out_dim)),
            nn.GELU(),
            nn.LayerNorm(int((embed_dim) * out_dim), eps=1e-6)
        )
        self.expert_norm = nn.LayerNorm(out_dim, eps=1e-6)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def get_expert(self, bsz, x, scores=None):
        pattern_queries = self.pattern_queries.expand(bsz, -1, -1)
        # SA
        pattern_queries = pattern_queries + self.drop_path(self.attn2(pattern_queries)[0])
        # CA
        pattern_queries = pattern_queries + self.drop_path(self.cross_attn(self.norm4(pattern_queries), x, x)[0])
        # FFN
        experts = self.mlp(self.norm5(pattern_queries)).reshape(bsz, -1, self.embed_dim, self.out_dim).permute(1, 0, 2, 3).contiguous()
        if scores is not None:
            experts = experts * scores.permute(1,0).contiguous().unsqueeze(-1).unsqueeze(-1) # [topk, B, 192, 768] * [B, topk]
            experts = experts.sum(dim=0)
            experts = experts / len(experts)
        return experts
    
    def forward(self, x, target_expert, return_topk=False):
        bsz, seq_len, _ = x.shape
        identity_x = x

        y, attn = self.attn(x)
        x = x + self.drop_path(y)
        x = self.norm2(x)
        _, _, aux_loss, scores = self.gate(x, target_expert, return_scores=True) # [B, topk]
        
        experts = self.get_expert(bsz, identity_x, scores)
        weight_decay = 0.05 * 0.5
        experts_loss = weight_decay * (experts**2).sum(dim=(1,2)).mean()
        y = self.after_experts(torch.matmul(x, experts))

        return y, attn, aux_loss, experts_loss
    
class LossMOEDecoderV2(nn.Module): # Two patterns
    def __init__(self, embed_dim, align_num_heads, out_dim=1024, nlayers=3, hidden_dim=2048, bottleneck_dim=256, norm_layer=nn.LayerNorm, drop=0., drop_path = 0. , mlp_ratio = 4., **args):
        super().__init__()
        num_experts, top_k = 2, 1
        
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.down_dim = nn.Linear(embed_dim, 192)
        self.down_dim_relu = nn.GELU()
        embed_dim = 192
        self.embed_dim = embed_dim

        self.norm1 = norm_layer(embed_dim)
        self.attn = Attention(
            embed_dim, num_heads=align_num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = norm_layer(embed_dim)
        self.gate = MoEGate(top_k, num_experts, embed_dim)
        self.after_experts = AfterExpert(embed_dim, out_dim, nlayers, hidden_dim, bottleneck_dim)

        self.pattern_queries = nn.Parameter(torch.zeros(1, 2, embed_dim))
        self.norm3 = norm_layer(embed_dim)
        self.attn2 = Attention(
            embed_dim, num_heads=align_num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=drop)
        
        self.norm4 = norm_layer(embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=align_num_heads,batch_first=True)
        
        self.norm5 = norm_layer(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int((embed_dim) * out_dim)),
            nn.GELU(),
            nn.LayerNorm(int((embed_dim) * out_dim), eps=1e-6)
        )
        self.expert_norm = nn.LayerNorm(out_dim, eps=1e-6)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def get_expert(self, bsz, x, scores=None):
        pattern_queries = self.pattern_queries.expand(bsz, -1, -1)
        # SA
        pattern_queries = pattern_queries + self.drop_path(self.attn2(pattern_queries)[0])
        # CA
        pattern_queries = pattern_queries + self.drop_path(self.cross_attn(self.norm4(pattern_queries), x, x)[0])
        # FFN
        experts = self.mlp(self.norm5(pattern_queries)).reshape(bsz, -1, self.embed_dim, self.out_dim).permute(1, 0, 2, 3).contiguous()
        if scores is not None:
            experts = experts * scores.permute(1,0).contiguous().unsqueeze(-1).unsqueeze(-1) # [topk, B, 192, 768] * [B, topk]
            experts = experts.sum(dim=0)
            experts = experts / len(experts) # math.sqrt(len(experts))
        return experts
    
    def forward(self, x, target_expert, return_topk=False):
        bsz, seq_len, _ = x.shape
        x = self.norm1(self.down_dim_relu(self.down_dim(x)))
        identity_x = x

        y, attn = self.attn(x)
        x = x + self.drop_path(y)
        x = self.norm2(x)
        _, _, aux_loss, scores = self.gate(x, target_expert, return_scores=True)
        
        experts = self.get_expert(bsz, identity_x, scores)
        weight_decay = 0.05 * 0.5
        experts_loss = weight_decay * (experts**2).sum(dim=(1,2)).mean()
        y = self.after_experts(torch.matmul(x, experts))

        return y, None, aux_loss, experts_loss