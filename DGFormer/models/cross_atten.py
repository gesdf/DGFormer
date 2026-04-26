import torch
import torch.nn as nn
from models.modules.vision_transformer import Attention

# 确保 DropPath 类可用
try:
    from timm.models.layers import DropPath
except ImportError:
    class DropPath(nn.Module):
        """Drop paths (Stochastic Depth) per sample"""
        def __init__(self, drop_prob=None):
            super(DropPath, self).__init__()
            self.drop_prob = drop_prob

        def forward(self, x):
            if self.drop_prob == 0. or not self.training:
                return x
            keep_prob = 1 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()
            output = x.div(keep_prob) * random_tensor
            return output

# 确保 Mlp 类可用
try:
    from models.modules.vision_transformer import Mlp
except ImportError:
    class Mlp(nn.Module):
        def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
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


class ConditionalRelativePosEncoder(nn.Module):
    """
    条件性相对位置编码器：
    根据主客体是否重叠，动态选择不同的编码策略
    """
    
    def __init__(self, hidden_dim=256):
        super().__init__()
        

        self.overlap_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()
        )
        

        # self.non_overlap_encoder = nn.Sequential(
        #     nn.Linear(2, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim // 2),
        #     nn.ReLU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(hidden_dim // 2, 1),
        #     nn.Tanh()
        # )
    
    def forward(self, rel_pos, is_overlap):
        """
        Args:
            rel_pos: [B*N_q*N_k, 3] 相对位置 [Δy, Δx, Δz]
            is_overlap: 标量布尔值，True 表示主客体重叠
        
        Returns:
            bias: [B*N_q*N_k, 1] 条件性注意力偏置
        """
        # ⭐ 全局判断：主客体是否重叠
        if is_overlap:
            # 使用 3D 坐标编码
            bias = self.overlap_encoder(rel_pos)
        else:
            # 使用 2D 坐标编码
            xy_coords = rel_pos[:, :2]
            bias = self.non_overlap_encoder(xy_coords)
        
        return bias


class CrossAttention(nn.Module):
    """跨注意力模块（只掩码 Key）"""
    def __init__(self, dim, num_heads=12, qkv_bias=False, attn_drop=0.0, proj_drop=0.0, 
                 use_relative_pos=True):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.use_relative_pos = use_relative_pos

        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
         # ⭐ 新增：用于存储注意力的变量
        self.attention_scores = None


    def forward(self, query, key, value, 
                query_padding_mask=None,
                key_padding_mask=None,  # ⭐ 只接受 Key 掩码
                query_pos=None, key_pos=None, 
                rel_pos_bias_fn=None, is_overlap=None):
        """
        Args:
            query: [B, N_q, D]
            key: [B, N_k, D]
            value: [B, N_k, D]
            query_padding_mask: [B, N_q] True 表示填充位置 (非主体)
            key_padding_mask: [B, N_k] True 表示 Key 的填充位置
            query_pos: [B, 196, 3]
            key_pos: [B, 196, 3]
            rel_pos_bias_fn: 相对位置编码函数
            is_overlap: [B] 重叠标志
        """
        B, N_q, C = query.shape #[196]
        _, N_k, _ = key.shape   #[196]

        q = self.proj_q(query)
        k = self.proj_k(key)
        v = self.proj_v(value)

        # 重塑为多头格式
        q = q.reshape(B, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, N_k, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, N_k, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, 196, 196]

        # ========== 添加相对位置偏置 ==========
        if self.use_relative_pos and query_pos is not None and key_pos is not None and rel_pos_bias_fn is not None:

            rel_pos = query_pos.unsqueeze(2) - key_pos.unsqueeze(1)

            is_sub_token = ~query_padding_mask
            is_obj_token = ~key_padding_mask
            # interaction_mask: [B, N_q, N_k], True 表示 (主体token, 客体token) 对
            interaction_mask = is_sub_token.unsqueeze(2) & is_obj_token.unsqueeze(1)
            # 3. 筛选出需要计算偏置的相对位置
            rel_pos_to_compute = rel_pos[interaction_mask]

            if rel_pos_to_compute.numel() > 0:
                # overlap_flags_to_compute: [Num_Interactions]
                batch_indices = torch.where(interaction_mask)[0]
                overlap_flags_for_interactions = is_overlap[batch_indices]

                # 初始化偏置向量
                bias_values = torch.zeros(rel_pos_to_compute.shape[0], 1, device=rel_pos.device, dtype=rel_pos.dtype)

                # 分别处理重叠和非重叠情况
                overlap_indices = torch.where(overlap_flags_for_interactions)[0]
                non_overlap_indices = torch.where(~overlap_flags_for_interactions)[0]

                if len(overlap_indices) > 0:
                    bias_values[overlap_indices] = rel_pos_bias_fn.overlap_encoder(
                        rel_pos_to_compute[overlap_indices]
                    )
                
                if len(non_overlap_indices) > 0:
                    # bias_values[non_overlap_indices] = rel_pos_bias_fn.non_overlap_encoder(
                    #     rel_pos_to_compute[non_overlap_indices, :2] # 只取 xy
                    # )
                    # bias_values[overlap_indices] = rel_pos_bias_fn.overlap_encoder(
                    #     rel_pos_to_compute[overlap_indices]
                    # )
                    bias_values[overlap_indices] = rel_pos_bias_fn.overlap_encoder(
                        rel_pos_to_compute[overlap_indices]
                    )
                # 5. 创建一个全零的偏置矩阵，并将计算出的偏置值“散布”回去
                rel_pos_bias = torch.zeros(B, N_q, N_k, device=attn.device, dtype=attn.dtype)
                rel_pos_bias[interaction_mask] = bias_values.squeeze(-1)

                # 6. 扩展到多头维度并添加到 attn
                rel_pos_bias = rel_pos_bias.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
                attn = attn + rel_pos_bias
        if key_padding_mask is not None:
            key_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(key_mask, float('-inf'))

        attn = attn.softmax(dim=-1)
        # ⭐ 新增：存储注意力分数
        self.attention_scores = attn.detach()
        
        if torch.isnan(attn).any():
            raise ValueError("Attention contains NaN after softmax")

        attn = self.attn_drop(attn)

        output = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        output = self.proj(output)
        output = self.proj_drop(output)

        return output

class CrossAttentionBlock(nn.Module):
    """交叉注意力块（清零 Query 填充位置）"""
    def __init__(self, dim, num_heads, mlp_ratio=4, qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm_self = norm_layer(dim)
        self.self_attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        
        self.norm_cross = norm_layer(dim)
        self.cross_attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                                         attn_drop=attn_drop, proj_drop=drop, use_relative_pos=True)
        
        self.norm_mlp = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, query_tokens, context_tokens,
                query_padding_mask=None,  # ⭐ Query 掩码（用于清零）
                context_padding_mask=None,  # ⭐ Context (Key) 掩码
                query_pos=None, context_pos=None,
                rel_pos_bias_fn=None, is_overlap=None):
        """
        Args:
            query_tokens: [B, N_q, D]
            context_tokens: [B, N_k, D]
            query_padding_mask: [B, N_q] True 表示填充
            context_padding_mask: [B, N_k] True 表示填充
        """
        
        # ========== 自注意力 ==========
        # query_tokens_norm = self.norm_self(query_tokens)
        # self_attn_out = self.self_attn(query_tokens_norm)
        # query_tokens = query_tokens + self.drop_path(self_attn_out)
        
        # ⭐ 清零 Query 自注意力的填充位置
        if query_padding_mask is not None:
            query_tokens = query_tokens.masked_fill(
                query_padding_mask.unsqueeze(-1), 0.0
            )
        if context_padding_mask is not None:
            context_tokens = context_tokens.masked_fill(
                context_padding_mask.unsqueeze(-1), 0.0
            )
        
        # ========== 交叉注意力 ==========
        query_tokens_norm = self.norm_cross(query_tokens)
        
        # ⭐ 只传递 Key 掩码
        cross_attn_out = self.cross_attn(
            query=query_tokens_norm,
            key=context_tokens,
            value=context_tokens,
            query_padding_mask=query_padding_mask,
            key_padding_mask=context_padding_mask,  # ⭐ 只掩码 Key
            query_pos=query_pos,
            key_pos=context_pos,
            rel_pos_bias_fn=rel_pos_bias_fn,
            is_overlap=is_overlap
        )
        
        query_tokens = query_tokens + self.drop_path(cross_attn_out)
        
        # ⭐ 清零交叉注意力后的 Query 填充位置
        if query_padding_mask is not None:
            query_tokens = query_tokens.masked_fill(
                query_padding_mask.unsqueeze(-1), 0.0
            )
        # ========== MLP ==========
        query_tokens_norm = self.norm_mlp(query_tokens)
        mlp_out = self.mlp(query_tokens_norm)
        query_tokens = query_tokens + self.drop_path(mlp_out)
        
        # ⭐ 最后再清零一次
        if query_padding_mask is not None:
            query_tokens = query_tokens.masked_fill(
                query_padding_mask.unsqueeze(-1), 0.0
            )
        
        return query_tokens


class ModalityFusionDecoder(nn.Module):
    """模态融合解码器（支持批量处理）"""
    def __init__(self, dim, num_heads, depth=8, mlp_ratio=4., drop_path_rate=0.1):
        super().__init__()
        self.depth = depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        self.rgb_blocks = nn.ModuleList([
            CrossAttentionBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop_path=dpr[i])
            for i in range(depth)
        ])
    
    def forward(self, rgb_tokens, depth_tokens, 
                rgb_padding_mask=None, depth_padding_mask=None, is_overlap=None):
        """
        Args:
            rgb_tokens: [B, T, D]
            depth_tokens: [B, T, D]
            rgb_padding_mask: [B, T] True 表示填充
            depth_padding_mask: [B, T] True 表示填充
            is_overlap: [B] 重叠标志
        """
        output_rgb = rgb_tokens
        
        for i in range(self.depth):
            output_rgb = self.rgb_blocks[i](
                query_tokens=output_rgb,
                context_tokens=depth_tokens,
                query_padding_mask=rgb_padding_mask,  # ⭐ Query 掩码
                context_padding_mask=depth_padding_mask,  # ⭐ Key 掩码
                query_pos=None,
                context_pos=None,
                rel_pos_bias_fn=None,
                is_overlap=is_overlap
            )
        
        return output_rgb


class RelationDecoder(nn.Module):
    """关系解码器（支持批量处理）"""
    def __init__(self, dim, num_heads, depth=8, mlp_ratio=4., drop_path_rate=0.1):
        super().__init__()
        self.depth = depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.sub_blocks = nn.ModuleList([
            CrossAttentionBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop_path=dpr[i])
            for i in range(depth)
        ])

    def forward(self, sub_tokens, obj_tokens,
                sub_padding_mask=None, obj_padding_mask=None,
                sub_pos=None, obj_pos=None,
                rel_pos_bias_fn=None, is_overlap=None):
        """
        Args:
            sub_tokens: [B, N_sub, D]
            obj_tokens: [B, N_obj, D]
            sub_padding_mask: [B, N_sub] True 表示填充
            obj_padding_mask: [B, N_obj] True 表示填充
        """
        output_sub = sub_tokens
        
        for i in range(self.depth):
            output_sub = self.sub_blocks[i](
                query_tokens=output_sub,
                context_tokens=obj_tokens,
                query_padding_mask=sub_padding_mask,  # ⭐ Query 掩码
                context_padding_mask=obj_padding_mask,  # ⭐ Key 掩码
                query_pos=sub_pos,
                context_pos=obj_pos,
                rel_pos_bias_fn=rel_pos_bias_fn,
                is_overlap=is_overlap
            )
        
        return output_sub