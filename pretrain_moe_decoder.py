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

# class Expert(nn.Module):
#     def __init__(self, n_embd, dropout=0.1):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(n_embd, 4 * n_embd),  # 扩展维度
#             nn.ReLU(),                       # 激活函数
#             nn.Linear(4 * n_embd, n_embd),   # 恢复维度
#             nn.Dropout(dropout),             # 防止过拟合
#         )

#     def forward(self, x):
#         return self.net(x)

class Expert(nn.Module):
    def __init__(self, embed_dim, out_dim=768, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(embed_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(embed_dim, hidden_dim)]
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        
        self.last_layer = nn.Linear(bottleneck_dim, out_dim)

    def forward(self, x):
        x = self.mlp(x)
        x = self.last_layer(x)
        return x

#################################################### 没有负载均衡 ####################################################
class TopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(TopkRouter, self).__init__()
        self.top_k = top_k
        self.linear = nn.Linear(n_embed, num_experts)  # 将输入投影到专家数量
    
    def forward(self, mh_output):
        logits = self.linear(mh_output)  # (batch_size, tokens, num_experts)
        top_k_logits, indices = logits.topk(self.top_k, dim=-1)  # 获取前K个专家
        zeros = torch.full_like(logits, float('-inf'))  # 创建全'-inf'矩阵
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)  # 填充前K个专家的值
        router_output = F.softmax(sparse_logits, dim=-1)  # Softmax归一化
        return router_output, indices

class NoisyTopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = top_k
        self.topkroute_linear = nn.Linear(n_embed, num_experts)
        self.noise_linear = nn.Linear(n_embed, num_experts)  # 噪声层
    
    def forward(self, mh_output):
        logits = self.topkroute_linear(mh_output)
        noise_logits = self.noise_linear(mh_output)
        noise = torch.randn_like(logits) * F.softplus(noise_logits)  # 添加噪声
        noisy_logits = logits + noise
        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1) # topk的值，topk的idx
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits) # 根据 indices 所指定的位置，将 top_k_logits 中的值插入到 zeros 张量中，-1 代表操作的维度是最后一维
        router_output = F.softmax(sparse_logits, dim=-1) # 归一化
        return router_output, indices

class MOEDecoder(nn.Module):
    def __init__(self, embed_dim, out_dim=768, nlayers=3, hidden_dim=2048, bottleneck_dim=256, **args):
        super().__init__()
        self.out_dim = out_dim
        num_experts, top_k = 64, 8
        self.router = NoisyTopkRouter(embed_dim, num_experts, top_k)
        self.experts = nn.ModuleList([Expert(embed_dim, out_dim, nlayers, hidden_dim, bottleneck_dim) for _ in range(num_experts)])
        self.top_k = top_k
        
        self.decoder_norm = nn.LayerNorm(out_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x, is_msc=False):
        B, N, _ = x.shape
        gating_output, indices = self.router(x) # [B,N_patch, num_experts] 每个专家分配的权重，选的topk专家idx
        final_output = torch.zeros(B, N, self.out_dim).cuda()  # 初始化输出
        flat_x = x.view(-1, x.size(-1))  # 展平输入
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))  # 展平路由器输出

        for i, expert in enumerate(self.experts):
            expert_mask = (indices == i).any(dim=-1)  # 选择当前专家处理的tokens
            flat_mask = expert_mask.view(-1)
            if flat_mask.any():
                expert_input = flat_x[flat_mask]  # 获取输入
                expert_output = expert(expert_input)  # 专家处理
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)  # 获取权重
                weighted_output = expert_output * gating_scores  # 加权输出
                final_output[expert_mask] += weighted_output.squeeze(1)  # 累加结果
        
        final_output = self.decoder_norm(final_output)
        return final_output

#################################################### loss-free负载均衡 ####################################################
class LossFreeMOEDecoder(nn.Module):
    def __init__(self, embed_dim, out_dim=768, nlayers=3, hidden_dim=2048, bottleneck_dim=256, **args):
        super().__init__()
        num_experts, top_k = 64, 8
        
        self.out_dim = out_dim
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.experts = nn.ModuleList([Expert(embed_dim, out_dim, nlayers, hidden_dim, bottleneck_dim) for _ in range(num_experts)])
        
        self.gate = nn.Linear(embed_dim, num_experts, bias=False)
        self.bias = nn.Parameter(torch.zeros(num_experts))  # For load balancing
        self.bias_update_speed = 0.001
        self.register_buffer("expert_load", torch.zeros(num_experts))
        
        self.decoder_norm = nn.LayerNorm(out_dim)
        self.apply(self._init_weights)
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.02 / math.sqrt(hidden_dim))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x, is_msc=False):
        batch_size, seq_len, embed_dim = x.shape
        x_flat = x.view(-1, embed_dim)
        
        scores = torch.sigmoid(self.gate(x_flat))
        scores_for_choice = scores + self.bias.unsqueeze(0)
        top_scores, top_indices = scores_for_choice.topk(self.top_k, dim=-1)
        # _, top_indices = scores_for_choice.topk(self.top_k, dim=-1)
        # top_scores = scores.gather(1, top_indices)  # [batch_size * seq_len, top_k]

        top_scores = top_scores / (top_scores.sum(dim=-1, keepdim=True) + 1e-6)
  
        mask = F.one_hot(top_indices, self.num_experts).sum(dim=1).float()  # [bs * seq_len, num_experts]
        expert_load = mask.sum(dim=0)  # [num_experts]
        self.bias.data += self.bias_update_speed * (self.expert_load - expert_load)
        self.expert_load.lerp_(expert_load, 0.1)  # Exponential moving average

        combined = torch.zeros(batch_size*seq_len, self.out_dim).cuda()
        for i in range(self.top_k):
            expert_indices = top_indices[:, i]  # [batch_size * seq_len]
            coefficient = top_scores[:, i].unsqueeze(-1)  # [batch_size * seq_len, 1]

            for expert_idx, expert in enumerate(self.experts):
                mask = (expert_indices == expert_idx)
                if mask.any():
                    expert_inputs = x_flat[mask]
                    expert_outputs = expert(expert_inputs) * coefficient[mask]
                    combined.index_add_(0, torch.where(mask)[0], expert_outputs)
        combined = combined.view(batch_size, seq_len, self.out_dim)
        final_output = self.decoder_norm(combined)
        return final_output
    

#################################################### loss负载均衡 ####################################################
class MoEGate(nn.Module):
    def __init__(self, top_k, num_experts, dim, scoring_func="softmax", aux_loss_alpha=0.001, seq_aux=True, norm_topk_prob=True):
        super().__init__()
        self.top_k = top_k
        self.n_routed_experts = num_experts

        self.scoring_func = scoring_func
        self.alpha = aux_loss_alpha
        self.seq_aux = seq_aux

        self.norm_topk_prob = norm_topk_prob
        self.gating_dim = dim
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = 0
        return topk_idx, topk_weight, aux_loss

class LossMOEDecoder(nn.Module):
    def __init__(self, embed_dim, out_dim=768, nlayers=3, hidden_dim=2048, bottleneck_dim=256, **args):
        super().__init__()
        num_experts, top_k = 32, 2
        
        self.out_dim = out_dim
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.experts = nn.ModuleList([Expert(embed_dim, out_dim, nlayers, hidden_dim, bottleneck_dim) for _ in range(num_experts)])
        self.gate = MoEGate(top_k, num_experts, embed_dim)
        self.decoder_norm = nn.LayerNorm(out_dim)
        # self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x, is_msc=False, return_topk=False):
        identity = x
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape
        
        topk_idx, topk_weight, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        
        x = x.repeat_interleave(self.top_k, dim=0)
        # y = torch.empty_like(x, dtype=torch.float16)
        y = torch.empty([bsz*self.top_k*seq_len, self.out_dim], dtype=torch.float16).cuda()
        for i, expert in enumerate(self.experts):
            y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(y.dtype)  # 确保类型一致
        y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
        y = y.view(bsz, seq_len, self.out_dim)

        self.aux_loss = aux_loss
        
        if return_topk:
            return y, topk_idx.reshape(bsz, seq_len, self.top_k)
        return y
    
#################################################### 新adapter ####################################################
class MoEGateV2(nn.Module):
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

class ExpertV2(nn.Module):
    def __init__(self, embed_dim, out_dim=768, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        # self.mlp = nn.Linear(embed_dim, out_dim)
        self.relu = nn.GELU()
        self.norm = nn.LayerNorm(out_dim, eps=1e-6)

    def forward(self, x):
        # x = self.mlp(x)
        x = self.norm(self.relu(x))
        return x

class LossMOEDecoderV2(nn.Module): # 一个教师的版本
    def __init__(self, embed_dim, align_num_heads, out_dim=768, nlayers=3, hidden_dim=2048, bottleneck_dim=256, norm_layer=nn.LayerNorm, drop=0., drop_path = 0. , mlp_ratio = 4., **args):
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
        self.gate = MoEGateV2(top_k, num_experts, embed_dim)
        # self.after_experts = nn.ModuleList([ExpertV2(embed_dim, out_dim, nlayers, hidden_dim, bottleneck_dim) for _ in range(num_experts)])
        self.after_experts = ExpertV2(embed_dim, out_dim, nlayers, hidden_dim, bottleneck_dim)
        
        ############# 2
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
        
        # 更新参数
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
        # pattern_queries = pattern_queries + self.drop_path(self.attn2(self.norm3(pattern_queries))[0])
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
        identity_x = x

        y, attn = self.attn(x)
        x = x + self.drop_path(y)
        x = self.norm2(x)
        _, _, aux_loss, scores = self.gate(x, target_expert, return_scores=True) # [B, topk]
        
        experts = self.get_expert(bsz, identity_x, scores)
        # experts_loss = 0
        weight_decay = 0.05 * 0.5
        experts_loss = weight_decay * (experts**2).sum(dim=(1,2)).mean()
        y = self.after_experts(torch.matmul(x, experts)) # [B, N, L] * [B, 3, 129*768] -> [B, 3, 129*768]

        return y, attn, aux_loss, experts_loss
    
    # def forward(self, x, target_expert, return_topk=False):
    #     bsz, seq_len, _ = x.shape
        
    #     experts = self.get_expert(bsz, x)
        
    #     # y, attn = self.attn(self.norm1(x))
    #     y, attn = self.attn(x)
    #     x = x + self.drop_path(y)
    #     x = self.norm2(x)
    #     topk_idx, topk_weight, aux_loss = self.gate(x, target_expert) # [B, topk]
    #     topk_idx = topk_idx.view(-1) # [B*topk]
        
    #     x = x.repeat_interleave(self.top_k, dim=0) # [B*topk,N,L]
    #     y = torch.empty([bsz*self.top_k, seq_len, self.out_dim]).cuda()
    #     for i, expert in enumerate(experts):
    #         mask = (topk_idx == i)
    #         if mask.any():
    #             y[mask] = self.after_experts[i](torch.matmul(x[mask], expert[mask]))
    #     y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1) # [B, topk, 129*768]*[B, topk, 1] -> [B, 129*768]
    #     y = y.view(bsz, seq_len, self.out_dim)
        
    #     if return_topk:
    #         return y, topk_idx.reshape(bsz, self.top_k)
    #     return y, attn, aux_loss
    
class LossMOEDecoderV3(nn.Module): # 两个教师的版本
    def __init__(self, embed_dim, align_num_heads, out_dim=1024, nlayers=3, hidden_dim=2048, bottleneck_dim=256, norm_layer=nn.LayerNorm, drop=0., drop_path = 0. , mlp_ratio = 4., **args):
        super().__init__()
        num_experts, top_k = 2, 1
        
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.num_experts = num_experts
        self.top_k = top_k
        # self.down_dim = nn.Linear(768, 192)
        # self.down_dim_relu = nn.GELU()

        ############# 1
        self.norm1 = norm_layer(embed_dim)
        self.attn = Attention(
            embed_dim, num_heads=align_num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = norm_layer(embed_dim)
        self.gate = MoEGateV2(top_k, num_experts, embed_dim)
        # self.after_experts = nn.ModuleList([ExpertV2(embed_dim, out_dim, nlayers, hidden_dim, bottleneck_dim) for _ in range(num_experts)])
        self.after_experts = ExpertV2(embed_dim, out_dim, nlayers, hidden_dim, bottleneck_dim)
        
        ############# 2
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
        
        # 更新参数
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
        # pattern_queries = pattern_queries + self.drop_path(self.attn2(self.norm3(pattern_queries))[0])
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
        # x = self.norm1(self.down_dim_relu(self.down_dim(x)))
        identity_x = x

        y, attn = self.attn(x)
        x = x + self.drop_path(y)
        x = self.norm2(x)
        _, _, aux_loss, scores = self.gate(x, target_expert, return_scores=True) # [B, topk]
        
        experts = self.get_expert(bsz, identity_x, scores)
        # experts_loss = 0
        weight_decay = 0.05 * 0.5
        experts_loss = weight_decay * (experts**2).sum(dim=(1,2)).mean()
        y = self.after_experts(torch.matmul(x, experts)) # [B, N, L] * [B, 3, 129*768] -> [B, 3, 129*768]

        return y, None, aux_loss, experts_loss