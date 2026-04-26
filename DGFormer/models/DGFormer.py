import os
import math
import time
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import torchvision.models as models

from models.modules.vision_transformer import VisionTransformer, PatchEmbed, resize_pos_embed, trunc_normal_
from models.modules.read_net import PromptReadoutNet
from models.cross_atten import RelationDecoder, ModalityFusionDecoder,ConditionalRelativePosEncoder

import json
from spellchecker import SpellChecker
from Levenshtein import distance as lev_distance


class DepthFeatureExtractor(nn.Module):
    """深度特征提取器 - 使用可学习的卷积层映射"""
    def __init__(self):
        super(DepthFeatureExtractor, self).__init__()
        # 使用 1x1 卷积，将 1 通道深度图映射为 3 通道
        self.proj = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, depth_img):
        """
        Args:
            depth_img: (B, 1, H, W)
        Returns:
            (B, 3, H, W)
        """
        if depth_img is None:
            return None
        out = self.proj(depth_img)
        return out


class RelationQueryAttention(nn.Module):
    """
    堆叠的关系查询注意力模块
    核心：使用多层交叉注意力模块，逐层提取更深层次的关系信息
    """
    def __init__(self, feature_dim, num_relations, num_heads=12, hidden_dim=768, num_layers=4):
        super().__init__()
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.attention_scores = None
        
        self.relation_embed = nn.Embedding(num_relations, hidden_dim)
        trunc_normal_(self.relation_embed.weight, std=0.02)
        
        self.token_proj = nn.Linear(feature_dim, hidden_dim)
        
        self.attention_layers = nn.ModuleList([
            nn.ModuleDict({
                "cross_attn": nn.MultiheadAttention(hidden_dim, num_heads, dropout=0.0, batch_first=True),
                "norm_cross_query": nn.LayerNorm(hidden_dim),
                "norm_tokens": nn.LayerNorm(hidden_dim),
                "mlp": nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.Dropout(0.1), 
                    nn.GELU(),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.Dropout(0.1)  
                ),
                "norm_mlp": nn.LayerNorm(hidden_dim)
            })
            for _ in range(num_layers)
        ])
    
    def forward(self, tokens, predicate, padding_mask=None, pos_embed=None):
        B = tokens.shape[0]
        tokens_proj = self.token_proj(tokens) if tokens.shape[-1] != self.hidden_dim else tokens
        q = self.relation_embed(predicate).unsqueeze(1)
        
        for layer in self.attention_layers:
            tokens_norm = layer["norm_tokens"](tokens_proj)
            
            q_norm_cross = layer["norm_cross_query"](q)
            k_with_pos = tokens_norm + pos_embed if pos_embed is not None else tokens_norm
            
            q_cross, attn_weights = layer["cross_attn"](
                query=q_norm_cross, 
                key=k_with_pos, 
                value=tokens_norm,
                key_padding_mask=padding_mask
            )
            q = q + q_cross
            
            # --- FFN ---
            q_norm_mlp = layer["norm_mlp"](q)
            q = q + layer["mlp"](q_norm_mlp)
        self.attention_scores = attn_weights

        relation_feat = q.squeeze(1) 
        
        return relation_feat


class DGFormer(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_layer=PatchEmbed, 
                 pretrained="", use_attn_mask=False, prompt_emb_pool="max", drop=0.0,
                 predicate_dim=9, readnet_d_hidden=512, readnet_dropout=0.0, 
                 glove_path=None, category_map_path=None,
                 enable_spell_correction=True, enable_compound_split=True,
                 word_mappings_path="./word_mappings.json",
                 decoder_depth=2, decoder_drop_path=0.1, modality_fusion_depth=2,
                 use_resnet_fpn=False, feature_extractor_backbone="resnet50", pretrained_path=None,
                 *args, **kwargs):
        super(DGFormer, self).__init__(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, 
            embed_layer=embed_layer, *args, **kwargs
        )
        
        self.predicate_dim = predicate_dim
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim)
        self.depth_patch_embed= PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.cls_token=nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.enable_spell_correction = enable_spell_correction
        self.enable_compound_split = enable_compound_split
        self.word_mappings_path = word_mappings_path


        self.prompt_emb_pool = prompt_emb_pool
        self.num_tokens = 1
        self.use_attn_mask = use_attn_mask

        # --- 投影层 ---
        self.t_proj = nn.Linear(17, self.embed_dim)
        self.t_norm = nn.LayerNorm(self.embed_dim)
        self.combined_norm = nn.LayerNorm(self.embed_dim)
        # # --- GloVe 词嵌入相关 ---
        # self.use_glove = glove_path is not None and category_map_path is not None
        # self.category_map_path = category_map_path
        # if self.use_glove:
        #     self._initialize_glove(glove_path)

        # --- 位置编码 ---
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, self.embed_dim))
        trunc_normal_(self.pos_embed, std=self.init_std)
        # 关系查询位置编码
        self.combined_pos_embed = nn.Parameter(torch.zeros(1, 3, self.embed_dim))
        trunc_normal_(self.combined_pos_embed, std=0.02)

        self.absolute_pos_encoder = nn.Sequential(
            nn.Linear(2, self.embed_dim * 2),  # 输入 (y, x) 归一化坐标
            nn.ReLU(),
            nn.Linear(self.embed_dim *2, self.embed_dim)
        )
        
        self.relative_pos_encoder = ConditionalRelativePosEncoder(hidden_dim=256)
        self.rel_pos_scale = nn.Parameter(torch.tensor(1.00))  
        

        # --- 深度特征提取器 ---
        self.depth_extractor = DepthFeatureExtractor()

        # --- 加载预训练权重 ---
        self.load_pretrained(pretrained)
        self.decoder_depth=decoder_depth
        self.modality_fusion_depth=modality_fusion_depth

        
        # --- 关系解码器 ---
        self.relation_decoder = RelationDecoder(
            dim=self.embed_dim, 
            num_heads=self.num_heads, 
            depth=decoder_depth,
            drop_path_rate=decoder_drop_path
        )

        self.relation_query_attention = RelationQueryAttention(
            feature_dim=self.embed_dim,  # token的维度
            num_relations=self.predicate_dim,
            num_heads=12,
            hidden_dim=768,
            num_layers=4  # 
        )

        readout_head_input_dim = self.embed_dim
        self.readout_head = PromptReadoutNet(
            readout_head_input_dim, 
            readnet_d_hidden, 
            self.predicate_dim, 
            readnet_dropout
        )


    def load_pretrained(self, pretrained):
        if not pretrained:
            print("从头开始训练，不加载预训练权重")
            return
        
        print(f"🔄 开始加载预训练权重: {pretrained}")
        checkpoint = torch.load(pretrained, map_location="cpu")
        
        if 'model' in checkpoint:
            load_state_dict = checkpoint["model"]
        elif 'model_state' in checkpoint:
            load_state_dict = checkpoint["model_state"]
        elif 'state_dict' in checkpoint:
            load_state_dict = checkpoint['state_dict']
        else:
            load_state_dict = checkpoint

        # 1. 加载编码器权重
        filtered_dict = OrderedDict()
        for k, v in load_state_dict.items():
            if any(skip in k for skip in ['decoder', 'mask_token', 'readout_head', 'depth_extractor', 
                                        'relation_decoder', 'relation_query_attention']):
                continue
            new_k = k.replace('encoder.', '').replace('backbone.', '')
            filtered_dict[new_k] = v
        
        if "pos_embed" in filtered_dict:
            filtered_dict["pos_embed"] = resize_pos_embed(
                filtered_dict["pos_embed"], 
                self.pos_embed, 
                self.num_tokens, 
                self.patch_embed.grid_size
            )
        
        msg = self.load_state_dict(filtered_dict, strict=False)

        print(" 从 RGB patch_embed 初始化 depth_patch_embed...")
        try:
            self.depth_patch_embed.load_state_dict(self.patch_embed.state_dict())
            print("depth_patch_embed 初始化完成")
        except Exception as e:
            print(f"depth_patch_embed 初始化失败: {e}")
    
        print(f"编码器权重加载完成")
        print(f"   缺失的键: {len(msg.missing_keys)} 个")
        print(f"   意外的键: {len(msg.unexpected_keys)} 个")
        
        # 2. 从编码器派生初始化新模块
        self._init_new_modules_from_encoder()

    def _init_new_modules_from_encoder(self):
        """用编码器的参数初始化新增的注意力模块"""
        
        def split_qkv(qkv_linear):
            W = qkv_linear.weight.data
            B = qkv_linear.bias.data if qkv_linear.bias is not None else None
            dim = W.shape[0] // 3
            q_w, k_w, v_w = W[:dim].clone(), W[dim:2*dim].clone(), W[2*dim:].clone()
            q_b = B[:dim].clone() if B is not None else None
            k_b = B[dim:2*dim].clone() if B is not None else None
            v_b = B[2*dim:].clone() if B is not None else None
            return (q_w, q_b), (k_w, k_b), (v_w, v_b)
        
        def copy_linear_weights(dst_linear, src_w, src_b=None):
            try:
                if dst_linear.weight.shape == src_w.shape:
                    dst_linear.weight.data.copy_(src_w)
                    if dst_linear.bias is not None and src_b is not None:
                        dst_linear.bias.data.copy_(src_b)
            except Exception as e:
                pass
        
        def copy_norm_weights(dst_norm, src_norm):
            try:
                dst_norm.load_state_dict(src_norm.state_dict(), strict=False)
            except Exception as e:
                pass
        
        encoder_blocks = list(self.blocks)
        if len(encoder_blocks) == 0:
            return
        print("开始从编码器派生初始化交叉注意力模块...")
        
        def get_encoder_block(idx, total_layers):
            num_encoder_blocks = len(encoder_blocks)
            if total_layers <= num_encoder_blocks:
                return encoder_blocks[num_encoder_blocks - total_layers + idx]
            return encoder_blocks[idx % num_encoder_blocks]

        for dec_name in ['rgb_relation_decoder', 'depth_relation_decoder']:
            if hasattr(self, dec_name):
                decoder = getattr(self, dec_name)
                for i, blk in enumerate(decoder.sub_blocks): # sub_blocks 中包含了定制的 CrossAttention
                    src_block = get_encoder_block(i, self.decoder_depth)
                    
                    if hasattr(src_block, 'attn') and hasattr(src_block.attn, 'qkv'):
                        (q_w, q_b), (k_w, k_b), (v_w, v_b) = split_qkv(src_block.attn.qkv)
                        
                        # 初始化 custom 的 cross_attn
                        copy_linear_weights(blk.cross_attn.proj_q, q_w, q_b)
                        copy_linear_weights(blk.cross_attn.proj_k, k_w, k_b)
                        copy_linear_weights(blk.cross_attn.proj_v, v_w, v_b)
                        copy_linear_weights(blk.cross_attn.proj, src_block.attn.proj.weight.data, 
                                        src_block.attn.proj.bias.data if src_block.attn.proj.bias is not None else None)
                    
                    if hasattr(src_block, 'mlp'):
                        try:
                            blk.mlp.load_state_dict(src_block.mlp.state_dict(), strict=False)
                        except Exception:
                            pass
                            
                    if hasattr(src_block, 'norm1'):
                        copy_norm_weights(blk.norm_cross, src_block.norm1)
                    if hasattr(src_block, 'norm2'):
                        copy_norm_weights(blk.norm_mlp, src_block.norm2)


        if hasattr(self, 'relation_query_attention'):
            for i, layer in enumerate(self.relation_query_attention.attention_layers):
                src_block = get_encoder_block(i, self.relation_query_attention.num_layers)

                if hasattr(src_block, 'attn') and hasattr(src_block.attn, 'qkv'):
                    cross_attn = layer["cross_attn"]
                    try:
                        cross_attn.in_proj_weight.data.copy_(src_block.attn.qkv.weight.data)
                        if cross_attn.in_proj_bias is not None and src_block.attn.qkv.bias is not None:
                            cross_attn.in_proj_bias.data.copy_(src_block.attn.qkv.bias.data)
                        cross_attn.out_proj.weight.data.copy_(src_block.attn.proj.weight.data)
                        if cross_attn.out_proj.bias is not None and src_block.attn.proj.bias is not None:
                            cross_attn.out_proj.bias.data.copy_(src_block.attn.proj.bias.data)
                    except Exception as e:
                        print(f"Query Attention 初始化失败: {e}")
                
                # MLP 映射复制
                if hasattr(src_block, 'mlp'):
                    try:
                        layer["mlp"][0].weight.data.copy_(src_block.mlp.fc1.weight.data)
                        layer["mlp"][0].bias.data.copy_(src_block.mlp.fc1.bias.data)
                        layer["mlp"][2].weight.data.copy_(src_block.mlp.fc2.weight.data)
                        layer["mlp"][2].bias.data.copy_(src_block.mlp.fc2.bias.data)
                    except Exception:
                        pass

                # 初始化对应的 LayerNorm
                if hasattr(src_block, 'norm1'):
                    copy_norm_weights(layer["norm_cross_query"], src_block.norm1)
                    copy_norm_weights(layer["norm_tokens"], src_block.norm1)
                if hasattr(src_block, 'norm2'):
                    copy_norm_weights(layer["norm_mlp"], src_block.norm2)

            # 初始化条件查询的 Embedding Token
            with torch.no_grad():
                if hasattr(self, 'cls_token') and self.cls_token is not None:
                    cls_feat = self.cls_token.squeeze(0).squeeze(0)
                    if cls_feat.abs().sum() < 1e-6:
                        cls_feat = self.pos_embed[:, 0, :].squeeze(0)
                    
                    if hasattr(self.relation_query_attention, 'relation_embed'):
                        self.relation_query_attention.relation_embed.weight.data = cls_feat.unsqueeze(0).repeat(
                            self.predicate_dim, 1
                        )
                        print("relation_embed 初始化完成")

        print("交叉注意力模块派生初始化完成")

    def get_bbox_image_patches(self, full_im, bbox_s, bbox_o):
        """根据边界框提取图像patch"""
        batch_size, _, h, w = full_im.shape
        masked_img = torch.zeros_like(full_im)
        sub_img_mask_slice, obj_img_mask_slice = [], []
        
        for i, (sub_box, obj_box) in enumerate(zip(bbox_s, bbox_o)):
            sy1 = int(sub_box[0].item() * h)
            sy2 = int(sub_box[1].item() * h)
            sx1 = int(sub_box[2].item() * w)
            sx2 = int(sub_box[3].item() * w)
            
            oy1 = int(obj_box[0].item() * h)
            oy2 = int(obj_box[1].item() * h)
            ox1 = int(obj_box[2].item() * w)
            ox2 = int(obj_box[3].item() * w)
            
            masked_img[i, :, sy1:sy2, sx1:sx2] = full_im[i, :, sy1:sy2, sx1:sx2]
            masked_img[i, :, oy1:oy2, ox1:ox2] = full_im[i, :, oy1:oy2, ox1:ox2]
            
            sub_img_mask_slice.append([sy1, sy2, sx1, sx2])
            obj_img_mask_slice.append([oy1, oy2, ox1, ox2])
        
        return masked_img, torch.tensor(sub_img_mask_slice, device=full_im.device), torch.tensor(obj_img_mask_slice, device=full_im.device)

    def get_roi_mask_and_positions(self, img_mask_slice, full_depth=None):
        """
        根据边界框生成 ROI 掩码和位置信息（不提取 token）
        
        Args:
            img_mask_slice: [B, 4] 每个batch的边界框 [y1, y2, x1, x2]（像素坐标）
            full_depth: [B, 1, H, W] 深度图（如果为None，则不使用深度）
        
        Returns:
            roi_mask: [B, num_patches] 布尔掩码，True 表示该 patch 在 ROI 内
            roi_positions: [B, num_patches, 3] 所有 patch 的 3D 位置 (y, x, z)
        """
        batch_size = img_mask_slice.shape[0]
        device = img_mask_slice.device
        
        patch_size_h, patch_size_w = self.patch_embed.patch_size
        grid_h, grid_w = self.patch_embed.grid_size
        
        # 初始化掩码和位置
        roi_mask = torch.zeros(batch_size, self.num_patches, device=device, dtype=torch.bool)
        roi_positions = torch.zeros(batch_size, self.num_patches, 3, device=device)
        
        # 预先计算所有 patch 的归一化位置 (y, x)
        y_coords_norm = torch.arange(grid_h, device=device).float() / grid_h
        x_coords_norm = torch.arange(grid_w, device=device).float() / grid_w
        yy_norm, xx_norm = torch.meshgrid(y_coords_norm, x_coords_norm, indexing='ij')
        yy_flat = yy_norm.flatten()  # [num_patches]
        xx_flat = xx_norm.flatten()  # [num_patches]
        
        for b, mask_slice in enumerate(img_mask_slice):
            # 强制转为 Python int（避免 tensor 的除法/索引引发浮点/广播问题）
            y_start = int(mask_slice[0].item()) // int(patch_size_h)
            y_end = math.ceil(int(mask_slice[1].item()) / int(patch_size_h))
            x_start = int(mask_slice[2].item()) // int(patch_size_w)
            x_end = math.ceil(int(mask_slice[3].item()) / int(patch_size_w))
            
            # 确保边界框有效且至少包含一个 patch（避免产生全 False 的 roi_mask）
            if y_start >= y_end or x_start >=x_end:

                # 保证至少包含一个 patch
                y_start = max(0, min(grid_h - 1, y_start))
                y_end = max(y_start + 1, min(grid_h, y_end if y_end > y_start else y_start + 1))
                x_start = max(0, min(grid_w - 1, x_start))
                x_end = max(x_start + 1, min(grid_w, x_end if x_end > x_start else x_start + 1))

            # 计算深度信息
            if full_depth is not None:
                depth_img = full_depth[b, 0]  # [H, W]
                depth_values = []
                
                for py in range(grid_h):
                    for px in range(grid_w):
                        y_pix_start = py * patch_size_h
                        y_pix_end = min((py + 1) * patch_size_h, depth_img.shape[0])
                        x_pix_start = px * patch_size_w
                        x_pix_end = min((px + 1) * patch_size_w, depth_img.shape[1])
                        
                        patch_depth = depth_img[y_pix_start:y_pix_end, x_pix_start:x_pix_end]
                        avg_depth = patch_depth.mean().item() if patch_depth.numel() > 0 else 0.0
                        depth_values.append(avg_depth)
                
                depth_values = torch.tensor(depth_values, device=device).float()
                
                roi_depth_values = depth_values.view(grid_h, grid_w)[y_start:y_end, x_start:x_end].flatten()
                if roi_depth_values.numel() > 0:
                    depth_min = roi_depth_values.min()
                    depth_max = roi_depth_values.max()
                    depth_range = depth_max - depth_min
                    xy_range = max((y_end - y_start) / grid_h, (x_end - x_start) / grid_w)
                    
                    if depth_range > 1e-6:
                        depth_scale = xy_range / depth_range
                        z_coords_norm = (depth_values - depth_min) * depth_scale
                    else:
                        z_coords_norm = (depth_values - depth_min) * 0.3
                else:
                    z_coords_norm = torch.zeros_like(depth_values)
            else:
                z_coords_norm = torch.zeros(self.num_patches, device=device)
            
            # 组合 3D 坐标 [num_patches, 3]
            # 保证 roi_positions 大小匹配
            roi_positions[b] = torch.stack([yy_flat, xx_flat, z_coords_norm], dim=-1)
            
            # 设置 ROI 掩码
            for py in range(y_start, y_end):
                for px in range(x_start, x_end):
                    idx = py * grid_w + px
                    if 0 <= idx < self.num_patches:  # 
                        roi_mask[b, idx] = True
        
        return roi_mask, roi_positions
    def compute_all_patch_positions(self, full_depth, roi_slice_for_scaling):
        """
        为所有 patch 计算归一化的 3D 位置 (y, x, z)。
        深度 z 的缩放基于指定的 ROI 区域。

        Args:
            full_depth: [B, 1, H, W] 完整的深度图。
            roi_slice_for_scaling: [B, 4] 用于确定深度缩放范围的 ROI 边界框。

        Returns:
            positions: [B, num_patches, 3] 所有 patch 的 3D 位置 (y, x, z)。
        """
        batch_size = full_depth.shape[0]
        device = full_depth.device
        patch_size_h, patch_size_w = self.patch_embed.patch_size
        grid_h, grid_w = self.patch_embed.grid_size

        y_coords_norm = torch.arange(grid_h, device=device, dtype=torch.float32) / grid_h
        x_coords_norm = torch.arange(grid_w, device=device, dtype=torch.float32) / grid_w
        yy_norm, xx_norm = torch.meshgrid(y_coords_norm, x_coords_norm, indexing='ij')
        # [num_patches, 2]
        yx_coords_flat = torch.stack([yy_norm.flatten(), xx_norm.flatten()], dim=-1)
        full_depth = full_depth[:, 0:1, :, :]
        depth_values_grid = F.avg_pool2d(
            full_depth, kernel_size=(patch_size_h, patch_size_w), stride=(patch_size_h, patch_size_w)
        )
        # [B, num_patches]
        depth_values_flat = depth_values_grid.flatten(start_dim=1)

        z_coords_norm = torch.zeros_like(depth_values_flat)
        for b in range(batch_size):
            # 获取用于缩放的 ROI 边界 (patch 级别)
            y_start = int(roi_slice_for_scaling[b, 0].item()) // patch_size_h
            y_end = math.ceil(roi_slice_for_scaling[b, 1].item() / patch_size_h)
            x_start = int(roi_slice_for_scaling[b, 2].item()) // patch_size_w
            x_end = math.ceil(roi_slice_for_scaling[b, 3].item() / patch_size_w)

            # 确保边界有效
            y_start = max(0, min(grid_h - 1, y_start))
            y_end = max(y_start + 1, min(grid_h, y_end if y_end > y_start else y_start + 1))
            x_start = max(0, min(grid_w - 1, x_start))
            x_end = max(x_start + 1, min(grid_w, x_end if x_end > x_start else x_start + 1))

            # 提取 ROI 内的深度值以确定缩放范围
            roi_depth_values = depth_values_grid[b, 0, y_start:y_end, x_start:x_end].flatten()
            
            if roi_depth_values.numel() > 0:
                depth_min = roi_depth_values.min()
                depth_max = roi_depth_values.max()
                depth_range = depth_max - depth_min
                # 使用 patch 级别的 ROI 尺寸计算 xy_range
                xy_range = max((y_end - y_start) / grid_h, (x_end - x_start) / grid_w)

                if depth_range > 1e-6:
                    depth_scale = xy_range / depth_range
                    # 对当前样本的所有 patch 应用相同的缩放
                    z_coords_norm[b] = (depth_values_flat[b] - depth_min) * depth_scale
                else:
                    # 如果 ROI 内深度几乎不变，使用一个小的固定缩放
                    z_coords_norm[b] = (depth_values_flat[b] - depth_min) * 0.3
        
        positions = torch.cat([
            yx_coords_flat.unsqueeze(0).expand(batch_size, -1, -1),
            z_coords_norm.unsqueeze(-1)
        ], dim=-1)

        return positions
    def compute_iou_and_overlap(self, bbox_s, bbox_o):
        """
        计算主客体之间的 IoU 和是否重叠
        
        Args:
            bbox_s: [B, 4] 主体边界框 [y1, y2, x1, x2]
            bbox_o: [B, 4] 客体边界框 [y1, y2, x1, x2]
        
        Returns:
            is_overlap: [B] 布尔张量，True 表示主客体重叠
            iou: [B] IoU 值
        """
        batch_size = bbox_s.shape[0]
        device = bbox_s.device
        
        # 计算交集
        y1_inter = torch.max(bbox_s[:, 0], bbox_o[:, 0])
        y2_inter = torch.min(bbox_s[:, 1], bbox_o[:, 1])
        x1_inter = torch.max(bbox_s[:, 2], bbox_o[:, 2])
        x2_inter = torch.min(bbox_s[:, 3], bbox_o[:, 3])
        
        # 交集面积
        inter_area = torch.clamp(y2_inter - y1_inter, min=0) * torch.clamp(x2_inter - x1_inter, min=0)
        
        # 计算并集
        area_s = (bbox_s[:, 1] - bbox_s[:, 0]) * (bbox_s[:, 3] - bbox_s[:, 2])
        area_o = (bbox_o[:, 1] - bbox_o[:, 0]) * (bbox_o[:, 3] - bbox_o[:, 2])
        union_area = area_s + area_o - inter_area
        
        # 计算 IoU
        iou = inter_area / (union_area + 1e-6)
        
        # 判断是否重叠（IoU > 0）
        is_overlap = iou > 0
        
        return is_overlap, iou
    
    def compute_depth_relative_embeddings(self, full_depth, bbox_s, bbox_o):
        batch_size, _, h, w = full_depth.shape
        
        sub_avg_depths = []
        obj_avg_depths = []
        sub_depth_ranges = []
        obj_depth_ranges = []
        
        epsilon = 1e-6  # 防止除零错误的小常数
    
        for b in range(batch_size):
            # 获取主体和客体的边界框
            y1_s, y2_s, x1_s, x2_s = bbox_s[b]
            y1_o, y2_o, x1_o, x2_o = bbox_o[b]
            
            # 转换为像素坐标
            y1_s_px, y2_s_px = int(y1_s * h), int(y2_s * h)
            x1_s_px, x2_s_px = int(x1_s * w), int(x2_s * w)
            y1_o_px, y2_o_px = int(y1_o * h), int(y2_o * h)
            x1_o_px, x2_o_px = int(x1_o * w), int(x2_o * w)
            
            # 确保边界框在有效范围内
            y1_s_px, y2_s_px = max(0, y1_s_px), min(h, y2_s_px)
            x1_s_px, x2_s_px = max(0, x1_s_px), min(w, x2_s_px)
            y1_o_px, y2_o_px = max(0, y1_o_px), min(h, y2_o_px)
            x1_o_px, x2_o_px = max(0, x1_o_px), min(w, x2_o_px)
            
            # 提取深度区域
            sub_region_depth = full_depth[b, 0, y1_s_px:y2_s_px, x1_s_px:x2_s_px]
            obj_region_depth = full_depth[b, 0, y1_o_px:y2_o_px, x1_o_px:x2_o_px]
            
            # 检查 sub_region_depth 是否为空
            if sub_region_depth.numel() > 0:
                sub_avg_depth = sub_region_depth.mean()
                sub_depth_range = sub_region_depth.max() - sub_region_depth.min()
            else:
                sub_avg_depth = torch.tensor(0.0, device=full_depth.device)
                sub_depth_range = torch.tensor(0.0, device=full_depth.device)
            
            # 检查 obj_region_depth 是否为空
            if obj_region_depth.numel() > 0:
                obj_avg_depth = obj_region_depth.mean()
                obj_depth_range = obj_region_depth.max() - obj_region_depth.min()
            else:
                obj_avg_depth = torch.tensor(0.0, device=full_depth.device)
                obj_depth_range = torch.tensor(0.0, device=full_depth.device)
            
            sub_avg_depths.append(sub_avg_depth)
            obj_avg_depths.append(obj_avg_depth)
            sub_depth_ranges.append(sub_depth_range)
            obj_depth_ranges.append(obj_depth_range)
        
        # 转换为张量
        sub_avg_depths = torch.stack(sub_avg_depths)  # [B]
        obj_avg_depths = torch.stack(obj_avg_depths)  # [B]
        sub_depth_ranges = torch.stack(sub_depth_ranges)  # [B]
        obj_depth_ranges = torch.stack(obj_depth_ranges)  # [B]
        
        # 计算深度相对特征
        depth_diff = sub_avg_depths - obj_avg_depths  # 深度差异
        depth_log_ratio = torch.log((sub_avg_depths + epsilon) / (obj_avg_depths + epsilon))  # 深度比的对数
        is_in_front = (depth_diff < 0).float()  # 主体是否在客体前方（标志位）
        is_behind = (depth_diff > 0).float()  # 主体是否在客体后方（标志位）
        
        # 拼接所有深度特征
        depth_embeddings = torch.stack([
            depth_diff,      # 深度差异 [B]
            depth_log_ratio, # 深度比的对数 [B]
            sub_depth_ranges,  # 主体深度范围 [B]
            obj_depth_ranges,  # 客体深度范围 [B]
            is_in_front,     # 主体是否在客体前方 [B]
            is_behind        # 主体是否在客体后方 [B]
        ], dim=1)  # 拼接为 [B, 6]
    
        return depth_embeddings.to(full_depth.device)  # 返回 [B, 6]



    def prompt_pooling(self, prompt_embedding, img_mask_slice=None):
        if self.prompt_emb_pool == 'max':
            pooled_embedding = torch.max(prompt_embedding, dim=1).values
        elif self.prompt_emb_pool == 'max-in-roi':
            assert img_mask_slice is not None
            rearrange_embedding = rearrange(prompt_embedding, "b (h w) c -> b h w c", h=self.patch_embed.grid_size[0], w=self.patch_embed.grid_size[1])
            pooled_embedding = []
            patch_size_h, patch_size_w = self.patch_embed.patch_size[0], self.patch_embed.patch_size[1]
            for i, (embedding, mask_slice) in enumerate(zip(rearrange_embedding, img_mask_slice)):
                roi_tokens = embedding[mask_slice[0] // patch_size_h: math.ceil(mask_slice[1] / patch_size_h),
                                   mask_slice[2] // patch_size_w: math.ceil(mask_slice[3] / patch_size_w), :].flatten(0, 1)
                if roi_tokens.shape[0] == 0:
                    roi_tokens = torch.zeros(1, embedding.shape[-1], device=embedding.device)
                pooled_embedding.append(torch.max(roi_tokens, dim=0).values)
            pooled_embedding = torch.stack(pooled_embedding, dim=0)
        elif self.prompt_emb_pool == 'avg':
            pooled_embedding = torch.mean(prompt_embedding, dim=1)
        elif self.prompt_emb_pool == 'avg-in-roi':
            assert img_mask_slice is not None
            rearrange_embedding = rearrange(prompt_embedding, "b (h w) c -> b h w c", h=self.patch_embed.grid_size[0], w=self.patch_embed.grid_size[1])
            pooled_embedding = []
            patch_size_h, patch_size_w = self.patch_embed.patch_size[0], self.patch_embed.patch_size[1]
            for i, (embedding, mask_slice) in enumerate(zip(rearrange_embedding, img_mask_slice)):
                pooled_embedding.append(
                    torch.mean(
                        embedding[mask_slice[0] // patch_size_h: math.ceil(mask_slice[1] / patch_size_h),
                        mask_slice[2] // patch_size_w: math.ceil(mask_slice[3] / patch_size_w), :].flatten(0, 1), dim=0
                    )
                )
            pooled_embedding = torch.stack(pooled_embedding, dim=0)
        elif self.prompt_emb_pool == 'log-sum-exp':
            pooled_embedding = torch.logsumexp(prompt_embedding, dim=1)
        elif self.prompt_emb_pool == 'logsumexp-in-roi':
            assert img_mask_slice is not None
            rearrange_embedding = rearrange(prompt_embedding, "b (h w) c -> b h w c", h=self.patch_embed.grid_size[0],
                                            w=self.patch_embed.grid_size[1])
            pooled_embedding = []
            patch_size_h, patch_size_w = self.patch_embed.patch_size[0], self.patch_embed.patch_size[1]
            for i, (embedding, mask_slice) in enumerate(zip(rearrange_embedding, img_mask_slice)):
                pooled_embedding.append(
                    torch.logsumexp(
                        embedding[mask_slice[0] // patch_size_h: math.ceil(mask_slice[1] / patch_size_h),
                        mask_slice[2] // patch_size_w: math.ceil(mask_slice[3] / patch_size_w), :].flatten(0, 1), dim=0
                    )
                )
            pooled_embedding = torch.stack(pooled_embedding, dim=0)
        else:
            raise ValueError
        return pooled_embedding
    


    def masked_pooling(self, tokens, mask, mode='max'):
        """
        带掩码的池化
        
        Args:
            tokens: [B, N, D]
            mask: [B, N] True 表示填充位置
            mode: 'max' 或 'avg'
        """
        # 只做简短报告（避免打印整个 mask 导致大量日志）
        fully_padded = (mask.sum(dim=1) == mask.shape[1])
        if fully_padded.any():
            idxs = torch.where(fully_padded)[0].tolist()
            print(f"警告：检测到 {len(idxs)} 个完全填充的样本，索引: {idxs}")

        if mode == 'max':
            tokens_masked = tokens.masked_fill(mask.unsqueeze(-1), float('-inf'))
            pooled, _ = torch.max(tokens_masked, dim=1)
            # 处理全填充的情况：将 -inf 替换为 0
            pooled = torch.where(
                fully_padded.unsqueeze(-1),
                torch.zeros_like(pooled),
                pooled
            )
        elif mode == 'avg':
            tokens_masked = tokens.masked_fill(mask.unsqueeze(-1), 0.0)
            valid_counts = (~mask).sum(dim=1, keepdim=True).float()
            pooled = tokens_masked.sum(dim=1) / (valid_counts + 1e-6)
        else:
            raise ValueError(f"Unknown pooling mode: {mode}")
        
        return pooled


    def forward_features(self, full_im, bbox_s, bbox_o, predicate, full_depth=None, 
                        subject_label=None, object_label=None, subject_t=None, object_t=None,union_bbox=None, return_attention=False):
        batch_size = full_im.shape[0]
        device = full_im.device

        # # # 语义嵌入
        # semantic_embed = torch.zeros(batch_size, 1, self.embed_dim, device=device)
        # if self.use_glove and subject_label is not None and object_label is not None:
        #     sub_semantic_embed = self._get_semantic_embedding(subject_label, device)
        #     obj_semantic_embed = self._get_semantic_embedding(object_label, device)
        #     # semantic_embed = torch.cat((sub_semantic, obj_semantic), dim=2)
        # else:
        #     sub_semantic_embed = torch.zeros(batch_size, 1, self.embed_dim, device=device)
        #     obj_semantic_embed = torch.zeros(batch_size, 1, self.embed_dim, device=device)
        # 几何嵌入
        if subject_t is not None and object_t is not None:
            depth_t = self.compute_depth_relative_embeddings(full_depth, bbox_s, bbox_o)
            t_embed = torch.cat((subject_t, depth_t), dim=1)
            t_embed = self.t_proj(t_embed).unsqueeze(1)
            # t_embed = self.t_norm(t_embed)
        else:
            t_embed = torch.zeros(batch_size, 1, self.embed_dim, device=device)

        # 深度特征提取
        if full_depth is not None:
            full_depth = self.depth_extractor(full_depth)
        else:
            full_depth = torch.zeros_like(full_im)


        # 获取掩码图像和边界框切片
        masked_img, sub_img_mask_slice, obj_img_mask_slice = self.get_bbox_image_patches(full_im, bbox_s, bbox_o)
        dep_masked_img, _, _ = self.get_bbox_image_patches(full_depth, bbox_s, bbox_o)
        # 视觉Token
        rgb_prompts = self.patch_embed(masked_img) + self.pos_embed[:, 1:, :]
        depth_prompts = self.depth_patch_embed(dep_masked_img) + self.pos_embed[:, 1:, :]
        x_batched = torch.cat((rgb_prompts, depth_prompts), dim=0) 
        x_batched = self.pos_drop(x_batched)

        for block in self.blocks:
            x_batched = block(x_batched)
            
        x_batched = self.norm(x_batched)

        # 重新分离出 RGB 和 Depth
        encoded_rgb_prompts, encoded_depth_prompts = torch.chunk(x_batched, 2, dim=0)
        sub_rgb_mask, sub_rgb_pos = self.get_roi_mask_and_positions(sub_img_mask_slice, full_depth)
        obj_rgb_mask, obj_rgb_pos = self.get_roi_mask_and_positions(obj_img_mask_slice, full_depth)
        all_patch_positions = self.compute_all_patch_positions(full_depth, roi_slice_for_scaling=sub_img_mask_slice)

        is_overlap, iou = self.compute_iou_and_overlap(bbox_s, bbox_o)

        combined_rgb_mask = sub_rgb_mask | obj_rgb_mask      # [B, num_patches]
        sub_rgb_padding_mask = ~sub_rgb_mask      # [B, num_patches]
        obj_rgb_padding_mask = ~obj_rgb_mask      # [B, num_patches]

            # ========== 批量调用 RelationDecoder ==========
        sub_rgb_padded=encoded_rgb_prompts
        obj_rgb_padded=encoded_rgb_prompts
        sub_depth_padded=encoded_depth_prompts
        obj_depth_padded=encoded_depth_prompts
        # RGB 关系建模
        rgb_feat = self.relation_decoder(
            sub_tokens=sub_rgb_padded,
            obj_tokens=obj_rgb_padded,
            sub_padding_mask=sub_rgb_padding_mask,
            obj_padding_mask=obj_rgb_padding_mask,
            sub_pos=all_patch_positions,
            obj_pos=all_patch_positions,
            rel_pos_bias_fn=self.relative_pos_encoder,
            is_overlap=is_overlap
        )
        if return_attention:
            temp_rgb_attn_scores = self.relation_decoder.sub_blocks[-1].cross_attn.attention_scores.clone()
        # 深度关系建模
        depth_feat = self.relation_decoder(
            sub_tokens=sub_depth_padded,
            obj_tokens=obj_depth_padded,
            sub_padding_mask=sub_rgb_padding_mask,
            obj_padding_mask=obj_rgb_padding_mask,
            sub_pos=all_patch_positions,
            obj_pos=all_patch_positions,
            rel_pos_bias_fn=self.relative_pos_encoder,
            is_overlap=is_overlap
        )
        if return_attention:
            temp_depth_attn_scores = self.relation_decoder.sub_blocks[-1].cross_attn.attention_scores.clone()

        rgb_feat_pooled = self.masked_pooling(rgb_feat, sub_rgb_padding_mask, mode='max')
        depth_feat_pooled = self.masked_pooling(depth_feat, sub_rgb_padding_mask, mode='max')


        combined_tokens = torch.cat([
                rgb_feat_pooled.unsqueeze(1),   # [B, N_fusion, D]
                depth_feat_pooled.unsqueeze(1),
                t_embed           # [B, 1, D]
            ], dim=1)  # [B, N_total, D]
            
        
        combined_tokens = self.combined_norm(combined_tokens)

        relation_feat = self.relation_query_attention(
            tokens=combined_tokens,
            predicate=predicate,
            pos_embed=self.combined_pos_embed 
        )
        if return_attention:
                # 分别对多头注意力求平均
                avg_rgb_attn = temp_rgb_attn_scores.mean(dim=1)
                avg_depth_attn = temp_depth_attn_scores.mean(dim=1)
                
                sub_indices = (~sub_rgb_padding_mask).nonzero(as_tuple=False)
                b = 0 
                b_sub_indices = sub_indices[sub_indices[:, 0] == b][:, 1]

                if b_sub_indices.numel() > 0:
                    # 分别提取 RGB 和 Depth 的 token 级注意力
                    per_token_rgb = avg_rgb_attn[b, b_sub_indices, :]
                    per_token_depth = avg_depth_attn[b, b_sub_indices, :]
                    
                    grid_h, grid_w = self.patch_embed.grid_size
                    subject_token_coords_y = b_sub_indices // grid_w
                    subject_token_coords_x = b_sub_indices % grid_w
                    subject_token_coords = torch.stack([subject_token_coords_y, subject_token_coords_x], dim=1)
                    
                    # 变形为 2D 图像网格
                    rgb_maps_2d = per_token_rgb.reshape(-1, grid_h, grid_w)
                    depth_maps_2d = per_token_depth.reshape(-1, grid_h, grid_w)
                else:
                    rgb_maps_2d = torch.empty(0, *self.patch_embed.grid_size, device=full_im.device)
                    depth_maps_2d = torch.empty(0, *self.patch_embed.grid_size, device=full_im.device)
                    subject_token_coords = torch.empty(0, 2, device=full_im.device)
                
                cross_attention_data = {
                    "rgb_maps": rgb_maps_2d,
                    "depth_maps": depth_maps_2d,
                    "coords": subject_token_coords
                }

                q_embed = self.relation_query_attention.relation_embed(predicate) # [B, D]
                cos_sim_rgb = torch.cosine_similarity(q_embed, combined_tokens[:, 0, :], dim=-1)
                cos_sim_depth = torch.cosine_similarity(q_embed, combined_tokens[:, 1, :], dim=-1)
                cos_sim_geo = torch.cosine_similarity(q_embed, combined_tokens[:, 2, :], dim=-1)
                
                raw_sim = torch.stack([cos_sim_rgb, cos_sim_depth, cos_sim_geo], dim=1) 
                real_query_attn_weights = F.softmax(raw_sim * 5.0, dim=-1) 
                return relation_feat, (cross_attention_data, real_query_attn_weights)

        return relation_feat


    def forward_head(self, x, predicate):
        """
        预测头（修改）
        Args:
            pooled_feat: [B, embed_dim * 2] 池化后的特征
            combined_tokens: [B, N, embed_dim] 主客体拼接的token序列
            combined_mask: [B, N] 填充掩码
            predicate: [B] 候选关系类型
        """
        
        rel_dists = self.readout_head(x)  # [B, predicate_dim]
        
        batch_indices = torch.arange(len(predicate), device=predicate.device)
        output = rel_dists[batch_indices, predicate]  # [B]




        
        return output

    def forward(self, full_im, bbox_s, bbox_o, predicate, full_depth=None, 
                subject_label=None, object_label=None, subject_t=None, object_t=None,union_bbox=None, return_attention=False):
        x= self.forward_features(
            full_im, bbox_s, bbox_o, predicate, full_depth, 
            subject_label, object_label, subject_t, object_t,union_bbox,return_attention=return_attention
        )
        if return_attention:
            relation_feat, attention_data_tuple = x
            output = self.forward_head(relation_feat, predicate)
            return output, attention_data_tuple
        else:
            relation_feat = x
            output = self.forward_head(relation_feat, predicate)
            return output