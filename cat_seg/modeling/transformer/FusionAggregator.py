# --------------------------------------------------------
# CAT-Seg: Cost Aggregation for Open-vocabulary Semantic Segmentation
# Licensed under The MIT License [see LICENSE for details]
# Written by Seokju Cho and Heeseong Shin
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from timm.layers import PatchEmbed, Mlp, DropPath, to_2tuple, to_ntuple, trunc_normal_, _assert

# Modified Swin Transformer blocks for guidance implementetion
# https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
def window_partition(x, window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x



class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        head_dim (int): Number of channels per head (dim // num_heads if not set)
        window_size (tuple[int]): The height and width of the window.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, appearance_guidance_dim, num_heads, head_dim=None, window_size=7, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = to_2tuple(window_size)  # Wh, Ww
        win_h, win_w = self.window_size
        self.window_area = win_h * win_w
        self.num_heads = num_heads
        head_dim = head_dim or dim // num_heads
        attn_dim = head_dim * num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim + appearance_guidance_dim, attn_dim, bias=qkv_bias)
        self.k = nn.Linear(dim + appearance_guidance_dim, attn_dim, bias=qkv_bias)
        self.v = nn.Linear(dim, attn_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(attn_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        
        q = self.q(x).reshape(B_, N, self.num_heads, -1).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B_, N, self.num_heads, -1).permute(0, 2, 1, 3)
        v = self.v(x[:, :, :self.dim]).reshape(B_, N, self.num_heads, -1).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if mask is not None:
            num_win = mask.shape[0]
            attn = attn.view(B_ // num_win, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class SwinTransformerBlockVer9e(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        window_size (int): Window size.
        num_heads (int): Number of attention heads.
        head_dim (int): Enforce the number of channels per head
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
            self, dim, appearance_guidance_dim, input_resolution, num_heads=4, head_dim=None, window_size=7, shift_size=0,
            mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
            act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, appearance_guidance_dim=appearance_guidance_dim, num_heads=num_heads, head_dim=head_dim, window_size=to_2tuple(self.window_size),
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            cnt = 0
            for h in (
                    slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None)):
                for w in (
                        slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None)):
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)  # num_win, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, appearance_guidance, dino_guidance):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        if appearance_guidance is not None:
            appearance_guidance = appearance_guidance.view(B, H, W, -1)
            x = torch.cat([x, appearance_guidance], dim=-1)
        if dino_guidance is not None:
            dino_guidance = dino_guidance.view(B, H, W, -1)
            x = torch.cat([x, dino_guidance], dim=-1)
        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # num_win*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, x_windows.shape[-1])  # num_win*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # num_win*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        window_size (int): Window size.
        num_heads (int): Number of attention heads.
        head_dim (int): Enforce the number of channels per head
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
            self, dim, appearance_guidance_dim, input_resolution, num_heads=4, head_dim=None, window_size=7, shift_size=0,
            mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
            act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, appearance_guidance_dim=appearance_guidance_dim, num_heads=num_heads, head_dim=head_dim, window_size=to_2tuple(self.window_size),
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            cnt = 0
            for h in (
                    slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None)):
                for w in (
                        slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None)):
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)  # num_win, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, appearance_guidance):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        if appearance_guidance is not None:
            appearance_guidance = appearance_guidance.view(B, H, W, -1)
            x = torch.cat([x, appearance_guidance], dim=-1)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # num_win*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, x_windows.shape[-1])  # num_win*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # num_win*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class SwinTransformerBlockWrapper(nn.Module):
    def __init__(self, dim, appearance_guidance_dim, input_resolution, nheads=4, window_size=5, pad_len=0):
        super().__init__()

        self.block_1 = SwinTransformerBlock(dim, appearance_guidance_dim, input_resolution, num_heads=nheads, head_dim=None, window_size=window_size, shift_size=0)
        self.block_2 = SwinTransformerBlock(dim, appearance_guidance_dim, input_resolution, num_heads=nheads, head_dim=None, window_size=window_size, shift_size=window_size // 2)
        self.guidance_norm = nn.LayerNorm(appearance_guidance_dim) if appearance_guidance_dim > 0 else None

        self.pad_len = pad_len
        self.padding_tokens = nn.Parameter(torch.zeros(1, 1, dim)) if pad_len > 0 else None
        self.padding_guidance = nn.Parameter(torch.zeros(1, 1, appearance_guidance_dim)) if pad_len > 0 and appearance_guidance_dim > 0 else None
    
    def forward(self, x, appearance_guidance):
        """
        Arguments:
            x: B C T H W
            appearance_guidance: B C H W
        """
        B, C, T, H, W = x.shape
        
        x = rearrange(x, 'B C T H W -> (B T) (H W) C')
        if appearance_guidance is not None:
            appearance_guidance = self.guidance_norm(repeat(appearance_guidance, 'B C H W -> (B T) (H W) C', T=T))
        x = self.block_1(x, appearance_guidance)
        x = self.block_2(x, appearance_guidance)
        x = rearrange(x, '(B T) (H W) C -> B C T H W', B=B, T=T, H=H, W=W)
        return x


class SwinTransformerBlockWrapperVer9e(nn.Module):
    def __init__(self, dim, appearance_guidance_dim, input_resolution, nheads=4, window_size=5, pad_len=0):
        super().__init__()

        self.block_1 = SwinTransformerBlockVer9e(dim, appearance_guidance_dim*2, input_resolution, num_heads=nheads, head_dim=None, window_size=window_size, shift_size=0)
        self.block_2 = SwinTransformerBlockVer9e(dim, appearance_guidance_dim*2, input_resolution, num_heads=nheads, head_dim=None, window_size=window_size, shift_size=window_size // 2)
        self.guidance_norm = nn.LayerNorm(appearance_guidance_dim) if appearance_guidance_dim > 0 else None

        self.pad_len = pad_len
        self.padding_tokens = nn.Parameter(torch.zeros(1, 1, dim)) if pad_len > 0 else None
        self.padding_guidance = nn.Parameter(torch.zeros(1, 1, appearance_guidance_dim)) if pad_len > 0 and appearance_guidance_dim > 0 else None
    
    def forward(self, x, appearance_guidance, dino_guidance):
        """
        Arguments:
            x: B C T H W
            appearance_guidance: B C H W
        """
        B, C, T, H, W = x.shape
        
        x = rearrange(x, 'B C T H W -> (B T) (H W) C')
        if appearance_guidance is not None:
            appearance_guidance = self.guidance_norm(repeat(appearance_guidance, 'B C H W -> (B T) (H W) C', T=T))
        if dino_guidance is not None:
            dino_guidance = self.guidance_norm(repeat(dino_guidance, 'B C H W -> (B T) (H W) C', T=T))
        x = self.block_1(x, appearance_guidance,dino_guidance)
        x = self.block_2(x, appearance_guidance,dino_guidance)
        x = rearrange(x, '(B T) (H W) C -> B C T H W', B=B, T=T, H=H, W=W)
        return x
def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


class LinearAttention(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values):
        """ Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        return queried_values.contiguous()


class FullAttention(nn.Module):
    def __init__(self, use_dropout=False, attention_dropout=0.1):
        super().__init__()
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """ Multi-head scaled dot-product attention, a.k.a full attention.
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """

        # Compute the unnormalized attention and apply the masks
        QK = torch.einsum("nlhd,nshd->nlsh", queries, keys)
        if kv_mask is not None:
            QK.masked_fill_(~(q_mask[:, :, None, None] * kv_mask[:, None, :, None]), float('-inf'))

        # Compute the attention and the weighted average
        softmax_temp = 1. / queries.size(3)**.5  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=2)
        if self.use_dropout:
            A = self.dropout(A)

        queried_values = torch.einsum("nlsh,nshd->nlhd", A, values)

        return queried_values.contiguous()


class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim, guidance_dim, nheads=8, attention_type='linear'):
        super().__init__()
        self.nheads = nheads
        self.q = nn.Linear(hidden_dim + guidance_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim + guidance_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)

        if attention_type == 'linear':
            self.attention = LinearAttention()
        elif attention_type == 'full':
            self.attention = FullAttention()
        else:
            raise NotImplementedError
    
    def forward(self, x, guidance):
        """
        Arguments:
            x: B, L, C
            guidance: B, L, C
        """
        q = self.q(torch.cat([x, guidance], dim=-1)) if guidance is not None else self.q(x)
        k = self.k(torch.cat([x, guidance], dim=-1)) if guidance is not None else self.k(x)
        v = self.v(x)

        q = rearrange(q, 'B L (H D) -> B L H D', H=self.nheads)
        k = rearrange(k, 'B S (H D) -> B S H D', H=self.nheads)
        v = rearrange(v, 'B S (H D) -> B S H D', H=self.nheads)

        out = self.attention(q, k, v)
        out = rearrange(out, 'B L H D -> B L (H D)')
        return out


class ClassTransformerLayer(nn.Module):
    def __init__(self, hidden_dim=64, guidance_dim=64, nheads=8, attention_type='linear', pooling_size=(4, 4), pad_len=256) -> None:
        super().__init__()
        self.pool = nn.AvgPool2d(pooling_size) if pooling_size is not None else nn.Identity()
        self.attention = AttentionLayer(hidden_dim, guidance_dim, nheads=nheads, attention_type=attention_type)
        self.MLP = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.pad_len = pad_len
        self.padding_tokens = nn.Parameter(torch.zeros(1, 1, hidden_dim)) if pad_len > 0 else None
        self.padding_guidance = nn.Parameter(torch.zeros(1, 1, guidance_dim)) if pad_len > 0 and guidance_dim > 0 else None
    
    def pool_features(self, x):
        """
        Intermediate pooling layer for computational efficiency.
        Arguments:
            x: B, C, T, H, W
        """
        B = x.size(0)
        x = rearrange(x, 'B C T H W -> (B T) C H W')
        x = self.pool(x)
        x = rearrange(x, '(B T) C H W -> B C T H W', B=B)
        return x

    def forward(self, x, guidance):
        """
        Arguments:
            x: B, C, T, H, W
            guidance: B, T, C
        """
        B, C, T, H, W = x.size()
        x_pool = self.pool_features(x)
        *_, H_pool, W_pool = x_pool.size()
        
        if self.padding_tokens is not None:
            orig_len = x.size(2)
            if orig_len < self.pad_len:
                # pad to pad_len
                padding_tokens = repeat(self.padding_tokens, '1 1 C -> B C T H W', B=B, T=self.pad_len - orig_len, H=H_pool, W=W_pool)
                x_pool = torch.cat([x_pool, padding_tokens], dim=2)

        x_pool = rearrange(x_pool, 'B C T H W -> (B H W) T C')
        if guidance is not None:
            if self.padding_guidance is not None:
                if orig_len < self.pad_len:
                    padding_guidance = repeat(self.padding_guidance, '1 1 C -> B T C', B=B, T=self.pad_len - orig_len)
                    guidance = torch.cat([guidance, padding_guidance], dim=1)
            guidance = repeat(guidance, 'B T C -> (B H W) T C', H=H_pool, W=W_pool)

        x_pool = x_pool + self.attention(self.norm1(x_pool), guidance) # Attention
        x_pool = x_pool + self.MLP(self.norm2(x_pool)) # MLP

        x_pool = rearrange(x_pool, '(B H W) T C -> (B T) C H W', H=H_pool, W=W_pool)
        x_pool = F.interpolate(x_pool, size=(H, W), mode='bilinear', align_corners=True)
        x_pool = rearrange(x_pool, '(B T) C H W -> B C T H W', B=B)

        if self.padding_tokens is not None:
            if orig_len < self.pad_len:
                x_pool = x_pool[:, :, :orig_len]

        x = x + x_pool # Residual
        return x


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class AggregatorLayerVer9e(nn.Module):
    def __init__(self, hidden_dim=64, text_guidance_dim=512, appearance_guidance=512, nheads=4, input_resolution=(20, 20), pooling_size=(5, 5), window_size=(10, 10), attention_type='linear', pad_len=256) -> None:
        super().__init__()
        self.swin_block = SwinTransformerBlockWrapperVer9e(hidden_dim, appearance_guidance, input_resolution, nheads, window_size)
        self.attention = ClassTransformerLayer(hidden_dim, text_guidance_dim, nheads=nheads, attention_type=attention_type, pooling_size=pooling_size, pad_len=pad_len)

    def forward(self, x, appearance_guidance,dino_guidance, text_guidance):
        """
        Arguments:
            x: B C T H W
            appearance_guidance: B C H W
        """

        x = self.swin_block(x, appearance_guidance,dino_guidance)
        x = self.attention(x, text_guidance)

        return x


class AggregatorLayer(nn.Module):
    def __init__(self, hidden_dim=64, text_guidance_dim=512, appearance_guidance=512, nheads=4, input_resolution=(20, 20), pooling_size=(5, 5), window_size=(10, 10), attention_type='linear', pad_len=256) -> None:
        super().__init__()
        self.swin_block = SwinTransformerBlockWrapper(hidden_dim, appearance_guidance, input_resolution, nheads, window_size)
        self.attention = ClassTransformerLayer(hidden_dim, text_guidance_dim, nheads=nheads, attention_type=attention_type, pooling_size=pooling_size, pad_len=pad_len)

    def forward(self, x, appearance_guidance, text_guidance):
        """
        Arguments:
            x: B C T H W
            appearance_guidance: B C H W
        """

        x = self.swin_block(x, appearance_guidance)
        x = self.attention(x, text_guidance)
        return x
    

class AggregatorResNetLayer(nn.Module):
    def __init__(self, hidden_dim=64, appearance_guidance=512) -> None:
        super().__init__()
        self.conv_linear = nn.Conv2d(hidden_dim + appearance_guidance, hidden_dim, kernel_size=1, stride=1)
        self.conv_layer = Bottleneck(hidden_dim, hidden_dim // 4)


    def forward(self, x, appearance_guidance):
        """
        Arguments:
            x: B C T H W
        """
        B, T = x.size(0), x.size(2)
        x = rearrange(x, 'B C T H W -> (B T) C H W')
        appearance_guidance = repeat(appearance_guidance, 'B C H W -> (B T) C H W', T=T)

        x = self.conv_linear(torch.cat([x, appearance_guidance], dim=1))
        x = self.conv_layer(x)
        x = rearrange(x, '(B T) C H W -> B C T H W', B=B)
        return x

class DoubleConv_GNModified(nn.Module):
    """(convolution => [GN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(mid_channels // 8, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(mid_channels // 8, mid_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class DoubleConv(nn.Module):
    """(convolution => [GN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(mid_channels // 16, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(mid_channels // 16, mid_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, guidance_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels - guidance_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, guidance=None):
        x = self.up(x)
        if guidance is not None:
            T = x.size(0) // guidance.size(0)
            guidance = repeat(guidance, "B C H W -> (B T) C H W", T=T)
            x = torch.cat([x, guidance], dim=1)
        return self.conv(x)



class UPmy(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, guidance_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels - guidance_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv_GNModified(in_channels, out_channels)

    def forward(self, x, guidance=None):
        x = self.up(x)
        if guidance is not None:
            T = x.size(0) // guidance.size(0)
            guidance = repeat(guidance, "B C H W -> (B T) C H W", T=T)
            x = torch.cat([x, guidance], dim=1)
        return self.conv(x)

class FusionUP(nn.Module):
    """"Upscaling using feat from dino and clip"""
    def __init__(self, in_channels, out_channels, guidance_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels - guidance_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels+guidance_channels, out_channels)

    def forward(self, x, clip_guidance,dino_guidance):
        x = self.up(x)
        if clip_guidance is not None:
            T = x.size(0) // clip_guidance.size(0)
            clip_guidance = repeat(clip_guidance, "B C H W -> (B T) C H W", T=T)
            dino_guidance = repeat(dino_guidance, "B C H W -> (B T) C H W", T=T)
            x = torch.cat([x, clip_guidance,dino_guidance], dim=1)
        return self.conv(x)
    
class DecodFuse(nn.Module):
    """"Upscaling using feat from dino and clip"""
    def __init__(self, in_channels): # 128 64 32, 64 32 16, 32 16 8
        super().__init__()
        mid_channels_0 = in_channels//2
        mid_channels_1 = in_channels//4
        self.CV_branch = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=mid_channels_0,kernel_size=1,stride=1,padding=0),
            nn.GroupNorm(mid_channels_0//16, mid_channels_0),
            nn.GELU()
            )
        self.dino_branch = nn.Conv2d(in_channels=mid_channels_1,out_channels=mid_channels_1,kernel_size=1,stride=1,padding=0)
        self.clip_branch = nn.Conv2d(in_channels=mid_channels_1,out_channels=mid_channels_1,kernel_size=1,stride=1,padding=0)
        
        self.start_fuse = nn.Sequential(
            nn.GroupNorm(in_channels//16,in_channels),
            nn.Conv2d(in_channels =in_channels,out_channels=in_channels,stride=1,kernel_size=1,padding=0 ), # 1*1
            nn.GELU()
        )
        self.up = nn.ConvTranspose2d(in_channels, mid_channels_0, kernel_size=2, stride=2)
        self.dw_4_branch = nn.ModuleList(
            [nn.Conv2d(in_channels =mid_channels_0,out_channels=mid_channels_0,stride=1,kernel_size=2*i-1,padding=(2*i-1)//2,groups=mid_channels_0 )]
            for i in range(4))
        self.final_fuse = nn.Sequential(
            nn.Conv2d(in_channels =in_channels*2,out_channels=mid_channels_0,stride=1,kernel_size=1,padding=0),
            nn.GroupNorm(mid_channels_0//16, mid_channels_0),
            nn.GELU()
            )

    def forward(self, x, clip_guidance,dino_guidance):
        cv_branch = self.CV_branch(x)
        if clip_guidance is not None:
            T = x.size(0) // clip_guidance.size(0)
            clip_guidance = repeat(clip_guidance, "B C H W -> (B T) C H W", T=T)
            dino_guidance = repeat(dino_guidance, "B C H W -> (B T) C H W", T=T)
            clip_branch = self.clip_branch(clip_guidance)
            dino_branch = self.dino_branch(dino_guidance)
            intermediate_result = self.start_fuse(torch.cat([cv_branch,dino_branch,clip_branch],dim=1))
            up_result = self.up(intermediate_result)
            dw_conv_4 = []
            for dw_conv in self.dw_4_branch:
                dw_result = dw_conv(up_result)
                dw_conv_4.append(dw_result)
            out = self.final_fuse(torch.stack(dw_conv_4,dim=1))
        return out,intermediate_result
# class GenericClassandSpatialAttentionModule(nn.Module):
#     """ Channel attention module"""
#     def __init__(self, in_dim):
#         super().__init__()
#         self.chanel_in = in_dim


#         self.gamma_class = nn.Parameter(torch.zeros(1))
#         self.gamma_spatial = nn.Parameter(torch.zeros(1))
#         self.class_attn_proj1 = nn.Conv2d(in_dim,in_dim,3,1,0)
#         self.class_attn_proj2 = nn.Sequential(
#             nn.Linear(in_dim,in_dim//2),
#             nn.ReLU(),
#             nn.Linear(in_dim//2,1)
#         )
        
#         self.spatial_attn_proj1 = nn.Conv2d(in_dim,in_dim,3,1,0)
#         self.sptial_attn_proj2 = nn.Sequential(
#             nn.Linear(in_dim,in_dim//2),
#             nn.ReLU(),
#             nn.Linear(in_dim//2,1)
#         )
        
        
        
#         self.sigmoid = nn.Sigmoid()
#         self.softmax  = nn.Softmax(dim=-1)
#     def forward(self,x):
#         """
#             inputs :
#                 x : input feature maps( B T C H W)
#             returns :
#                 out : attention value + input feature
#                 attention: B X C X C
#         """
#         m_batchsize, C, height, width = x.size()
#         proj_query = x.view(m_batchsize, C, -1)
#         proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
#         energy = torch.bmm(proj_query, proj_key)
#         energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
#         attention = self.softmax(energy_new)
#         proj_value = x.view(m_batchsize, C, -1)

#         out = torch.bmm(attention, proj_value)
#         out = out.view(m_batchsize, C, height, width)

#         out = self.gamma*out + x
#         return out
class GenericClassAttentionModule(nn.Module):
    """ Generic class attention module"""
    def __init__(self, in_dim):
        super().__init__()
        self.chanel_in = in_dim


        self.gamma_class = nn.Parameter(torch.zeros(1))

        self.class_attn_proj1 = nn.Conv2d(in_dim,in_dim,3,1,1)
        self.class_attn_proj2 = nn.Sequential(
            nn.Linear(in_dim,in_dim//2),
            nn.ReLU(),
            nn.Linear(in_dim//2,1)
        )

        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        """
            inputs :
                x : input feature maps(B T C H W)
            returns :
                out : attention value + input feature (B T C H W)
               
        """
        T = x.shape[1]
        C = x.shape[2]
        H = x.shape[-2]
        proj_value = self.class_attn_proj1(rearrange(x,'B T C H W -> (B T) C H W'))
        proj_value = rearrange(proj_value,'B C H W -> B (H W) C')
        
        proj_query = self.class_attn_proj2(proj_value) # (BT), (HW), 1
        proj_query = rearrange(proj_query,"(B T) L 1 -> B T L",T=T) # L is (HW)
        proj_key = rearrange(proj_query,"B T L -> B L T")
        attn_map = self.sigmoid(torch.bmm(proj_query, proj_key))
        out = torch.bmm(attn_map, rearrange(proj_value,'(B T) L C -> B T (L C)',T=T))

        out = self.gamma_class*out
        
        out = rearrange(out,"B T (L C) -> B T C L",C = C)
        out = rearrange(out, "B T C (H W) -> B T C H W", H=H)
        out = out + x
        return out
    
class GenericClassAttentionModuleVer2(nn.Module):
    """ Generic class attention module"""
    def __init__(self, in_dim, zero_init=False):
        super().__init__()
        self.chanel_in = in_dim

        if zero_init:
            self.gamma_class = nn.Parameter(torch.zeros(1))
        else:
            self.gamma_class = nn.Parameter(torch.ones(1))

        self.class_attn_MLP = nn.Sequential(
            nn.Linear(in_dim,in_dim//2),
            nn.ReLU(),
            nn.Linear(in_dim//2,1),
        )
        # self.GAP = nn.AvgPool2d(kernel_size=24)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        """
            inputs :
                x : input feature maps(B T C H W)
            returns :
                out : attention value + input feature (B T C H W)
               
        """
        T = x.shape[1]
        C = x.shape[2]
        H = x.shape[-2]
        print(self.gamma_class,'class_attn')
        group_x = rearrange(x,'B T C H W -> (B T) C H W')
        
        GAP_x = F.adaptive_avg_pool2d(group_x,(1,1))
        GAP_x = rearrange(GAP_x,'B C 1 1 -> B 1 C')
        GMP_x =  F.adaptive_max_pool2d(group_x,(1,1))
        GMP_x = rearrange(GMP_x,'B C 1 1 -> B 1 C')
        proj_GAP_cls_attn = self.class_attn_MLP(GAP_x)
        proj_GMP_cls_attn = self.class_attn_MLP(GMP_x)
        cls_attn = proj_GAP_cls_attn + proj_GMP_cls_attn
        cls_attn = self.sigmoid(cls_attn)
        cls_attn = rearrange(cls_attn,'B 1 1 -> B')

        
        out_x = group_x + self.gamma_class * torch.einsum('B, B C H W -> B C H W',cls_attn,group_x)
        out_x = rearrange(out_x,"(B T ) C H W -> B T C H W",T = T)

        return out_x
class GenericClassandSpatialAttentionModuleVer3(nn.Module):
    """ Generic class attention module"""
    def __init__(self, in_dim):
        super().__init__()
        self.class_attn_model = GenericClassAttentionModuleVer2(in_dim)
        self.class_spatial_attn_model = GenericClassandSpatialAttentionModuleVer2(in_dim)

    def forward(self,x):
        """
            inputs :
                x : input feature maps(B T C H W)
            returns :
                out : attention value + input feature (B T C H W)
               
        """
        class_reinforced_x = self.class_attn_model(x)
        class_and_spatial_reinforced_x = self.class_spatial_attn_model(x)
        return class_reinforced_x + class_and_spatial_reinforced_x

class GenericClassandSpatialAttentionModuleVer2(nn.Module):
    """ Generic class attention module"""
    def __init__(self, in_dim,zero_init=False):
        super().__init__()
        self.chanel_in = in_dim

        if zero_init:
            self.gamma_class = nn.Parameter(torch.zeros(1))
        else:
            self.gamma_class = nn.Parameter(torch.ones(1))
        self.class_attn_MLP = nn.Sequential(
            nn.Linear(in_dim,in_dim//2),
            nn.ReLU(),
            nn.Linear(in_dim//2,1),
        )
        # self.GAP = nn.AvgPool2d(kernel_size=24)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        """
            inputs :
                x : input feature maps(B T C H W)
            returns :
                out : attention value + input feature (B T C H W)
               
        """
        print(self.gamma_class,'class_spatial_attn')
        T = x.shape[1]
        C = x.shape[2]
        H = x.shape[-2]
        # group_x = rearrange(x,'B T C H W -> (B T) C H W')
        flatten_x = rearrange(x,'B T C H W -> (B T) (H W) C') # B L C
        S_C_attn_map = self.class_attn_MLP(flatten_x)
        S_C_attn_map = self.sigmoid(S_C_attn_map)
        S_C_attn_map = rearrange(S_C_attn_map,"(B T) (H W) 1 -> B T 1 H W", T=T, H = H)
        S_C_attn_map = repeat(S_C_attn_map, "B T 1 H W -> B T C H W", C=C)
        out_x = x + self.gamma_class*(x*S_C_attn_map)
        return out_x
class ClipDenseProjLayer11(nn.Module):
    def __init__(self, hidden_dim=768, last_dim=512 ,appearance_guidance_proj_dim=128):
        super().__init__()
        self.used_clip_layer = 12
        self.appearance_guidance_proj_dim = appearance_guidance_proj_dim
        self.clip_dense_proj = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(768),
                nn.Linear(in_features=768, out_features=512),
                ) for _ in range(self.used_clip_layer - 1)
        ])
        # self.clip_last_proj = nn.Sequential(
        #         nn.LayerNorm(normalized_shape=last_dim),
        #         nn.Linear(in_features=last_dim, out_features=appearance_guidance_proj_dim),
        #         nn.ReLU()
        #         )
    def forward(self, clip_feat_list):
        # B L C 
        # C=appearance_guidance_proj_dim
        # print(self.sigma)
        # exit()
        B = clip_feat_list[0].shape[0]
        L = clip_feat_list[0].shape[1]
        C = self.appearance_guidance_proj_dim
        clip_aggregated_feat = []
        for l_ind ,clip_feat in enumerate(clip_feat_list):
            if l_ind == len(clip_feat_list)-1:
                proj_feat = clip_feat
                clip_aggregated_feat.append(proj_feat)
            else:
                proj_feat = self.clip_dense_proj[l_ind](clip_feat)
                clip_aggregated_feat.append(proj_feat)
        return clip_aggregated_feat
class ClipIntermediateProjLayer(nn.Module):
    def __init__(self, hidden_dim=768, last_dim=512):
        super().__init__()
        self.LN = nn.LayerNorm(hidden_dim)
        self.linear = nn.Linear(in_features=768, out_features=512)
    def forward(self, clip_intermediate_feat):
        # B L C
        proj_feat = self.LN(clip_intermediate_feat)
        proj_feat = self.linear(proj_feat)
        return proj_feat
class FusionAggregatorVer23(nn.Module):
    def __init__(self, 
        text_guidance_dim=512,
        text_guidance_proj_dim=128,
        appearance_guidance_dim=512,
        appearance_guidance_proj_dim=128,
        decoder_dims = (64, 32),
        decoder_guidance_dims=(256, 128),
        decoder_guidance_proj_dims=(32, 16),
        num_layers=4,
        nheads=4, 
        hidden_dim=128,
        pooling_size=(6, 6),
        feature_resolution=(24, 24),
        window_size=12,
        attention_type='linear',
        prompt_channel=1,
        pad_len=256,
    ) -> None:
        """
        Cost Aggregation Model for CAT-Seg
        Args:
            text_guidance_dim: Dimension of text guidance
            text_guidance_proj_dim: Dimension of projected text guidance
            appearance_guidance_dim: Dimension of appearance guidance
            appearance_guidance_proj_dim: Dimension of projected appearance guidance
            decoder_dims: Upsampling decoder dimensions
            decoder_guidance_dims: Upsampling decoder guidance dimensions
            decoder_guidance_proj_dims: Upsampling decoder guidance projected dimensions
            num_layers: Number of layers for the aggregator
            nheads: Number of attention heads
            hidden_dim: Hidden dimension for transformer blocks
            pooling_size: Pooling size for the class aggregation layer
                          To reduce computation, we apply pooling in class aggregation blocks to reduce the number of tokens during training
            feature_resolution: Feature resolution for spatial aggregation
            window_size: Window size for Swin block in spatial aggregation
            attention_type: Attention type for the class aggregation. 
            prompt_channel: Number of prompts for ensembling text features. Default: 1
            pad_len: Padding length for the class aggregation. Default: 256
                     pad_len enforces the class aggregation block to have a fixed length of tokens for all inputs
                     This means it either pads the sequence with learnable tokens in class aggregation,
                     or truncates the classes with the initial CLIP cosine-similarity scores.
                     Set pad_len to 0 to disable this feature.
            """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.layers = nn.ModuleList([
            AggregatorLayer(
                hidden_dim=hidden_dim, text_guidance_dim=text_guidance_proj_dim, appearance_guidance=appearance_guidance_proj_dim, 
                nheads=nheads, input_resolution=feature_resolution, pooling_size=pooling_size, window_size=window_size, attention_type=attention_type, pad_len=pad_len,
            ) for _ in range(num_layers)
        ])

        # self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.conv3 = nn.Conv2d(32, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.fusion_corr = nn.Conv2d(3*hidden_dim, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.guidance_projection = nn.Sequential(
            nn.Conv2d(appearance_guidance_dim, appearance_guidance_proj_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        ) if appearance_guidance_dim > 0 else None
        
        # self.L4_proj = ClipIntermediateProjLayer()
        # self.L8_proj = ClipIntermediateProjLayer()
        self.text_guidance_projection = nn.Sequential(
            nn.Linear(text_guidance_dim, text_guidance_proj_dim),
            nn.ReLU(),
        ) if text_guidance_dim > 0 else None

        self.CLIP_decoder_guidance_projection = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            ) for d, dp in zip(decoder_guidance_dims, decoder_guidance_proj_dims)
        ]) if decoder_guidance_dims[0] > 0 else None
        
        self.DINO_decoder_guidance_projection = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            ) for d, dp in zip(decoder_guidance_dims, decoder_guidance_proj_dims)
        ]) if decoder_guidance_dims[0] > 0 else None
        
        # self.decoder1 = Up(hidden_dim, decoder_dims[0], decoder_guidance_proj_dims[0])
        # self.decoder2 = Up(decoder_dims[0], decoder_dims[1], decoder_guidance_proj_dims[1])
        
        self.Fusiondecoder1=FusionUP(hidden_dim, decoder_dims[0], decoder_guidance_proj_dims[0])
        self.Fusiondecoder2=FusionUP(decoder_dims[0], decoder_dims[1], decoder_guidance_proj_dims[1])
        self.head = nn.Conv2d(decoder_dims[1], 1, kernel_size=3, stride=1, padding=1)
        # self.sigma = nn.Parameter(torch.ones(2))
        self.pad_len = pad_len

    def feature_map(self, img_feats, text_feats):
        # concatenated feature volume for feature aggregation baselines
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        img_feats = repeat(img_feats, "B C H W -> B C T H W", T=text_feats.shape[1])
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        text_feats = text_feats.mean(dim=-2) # average text features over different prompts
        text_feats = F.normalize(text_feats, dim=-1) # B T C
        text_feats = repeat(text_feats, "B T C -> B C T H W", H=img_feats.shape[-2], W=img_feats.shape[-1])
        return torch.cat((img_feats, text_feats), dim=1) # B 2C T H W

    def correlation(self, img_feats, text_feats):
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        corr = torch.einsum('bchw, btpc -> bpthw', img_feats, text_feats)
        return corr

    def corr_embed(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B P T H W -> (B T) P H W')
        corr_embed = self.conv1(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed
    
    def corr_fusion_embed_minimum(self,clip_corr,dino_corr):
        # this one does not import a 1*1 conv
        # instead we modify the original embedding layer to adapt to the concatenated corr volume.
        B = clip_corr.shape[0]
        clip_corr = rearrange(clip_corr, 'B P T H W -> (B T) P H W')
        dino_corr = rearrange(dino_corr, 'B P T H W -> (B T) P H W')
        fused_corr = torch.cat([clip_corr,dino_corr],dim = 1)
        fused_corr = self.conv1_modified(fused_corr)
        fused_corr = rearrange(fused_corr, '(B T) C H W -> B C T H W', B=B)
        return fused_corr

    def corr_fusion_embed_seperate(self,clip_corr,dino_corr,group_clip_corr):
        # this one does not import a 1*1 conv
        # instead we modify the original embedding layer to adapt to the concatenated corr volume.
        B = clip_corr.shape[0]
        self.sigmoid = nn.Sigmoid()
        clip_corr = rearrange(clip_corr, 'B P T H W -> (B T) P H W')
        dino_corr = rearrange(dino_corr, 'B P T H W -> (B T) P H W')
        clip_embed_corr = self.conv1(clip_corr)
        group_clip_embed_corr = self.conv3(group_clip_corr)
        dino_embed_corr = self.conv2(dino_corr)
        clip_embed_corr = self.sigmoid(clip_embed_corr)
        dino_embed_corr = self.sigmoid(dino_embed_corr)
        group_clip_embed_corr = self.sigmoid(group_clip_embed_corr)
        fused_corr = torch.cat([clip_embed_corr,dino_embed_corr,group_clip_embed_corr],dim = 1)
        fused_corr = self.fusion_corr(fused_corr)
        fused_corr = self.sigmoid(fused_corr)
        fused_corr = rearrange(fused_corr, '(B T) C H W -> B C T H W', B=B)
        clip_embed_corr = rearrange(clip_embed_corr, '(B T) C H W -> B C T H W', B=B)
        dino_embed_corr = rearrange(dino_embed_corr, '(B T) C H W -> B C T H W', B=B)
        return fused_corr, clip_embed_corr, dino_embed_corr
        
        
    def corr_fusion_embed(self,clip_corr,dino_corr):
        B = clip_corr.shape[0]
        clip_corr = rearrange(clip_corr, 'B P T H W -> (B T) P H W')
        dino_corr = rearrange(dino_corr, 'B P T H W -> (B T) P H W')
        fused_corr = torch.cat([clip_corr,dino_corr],dim = 1)
        fused_corr = self.fusion_corr(fused_corr)
        fused_corr = self.conv1(fused_corr)
        fused_corr = rearrange(fused_corr, '(B T) C H W -> B C T H W', B=B)
        return fused_corr
        # exit()
        
    def corr_projection(self, x, proj):
        corr_embed = rearrange(x, 'B C T H W -> B T H W C')
        corr_embed = proj(corr_embed)
        corr_embed = rearrange(corr_embed, 'B T H W C -> B C T H W')
        return corr_embed

    def upsample(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = F.interpolate(corr_embed, scale_factor=2, mode='bilinear', align_corners=True)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed

    def conv_decoder(self, x, guidance):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = self.decoder1(corr_embed, guidance[0])
        corr_embed = self.decoder2(corr_embed, guidance[1])
        corr_embed = self.head(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
        return corr_embed
    
    def Fusion_conv_decoer(self, x, clip_guidance,dino_guidance):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = self.Fusiondecoder1(corr_embed, clip_guidance[0],dino_guidance[0])
        corr_embed = self.Fusiondecoder2(corr_embed, clip_guidance[1],dino_guidance[1])
        corr_embed = self.head(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
        return corr_embed
    def group_wise_correlation(self, img_feats, text_feats, group_nums = 32):
        '''
        Return B P N T H W, wheareas N is the group number, P is prompt number, T is class number
        '''
        class_num = text_feats.shape[1]
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        text_feats = rearrange(text_feats,'B T 1 C -> B T C')
        img_text_mult = torch.einsum('b c h w, b t c -> b t c h w',img_feats,text_feats)
        img_text_grouped =rearrange(img_text_mult, ' B T (N C) H W -> B T N C H W',N=group_nums)
        grouped_corr = img_text_grouped.sum(dim=-3) # B T N H W
        grouped_corr = rearrange(grouped_corr,'B T N H W -> (B T) N H W')
        # group_corr_embed = self.gw_corr_embed(grouped_corr)
        #group_corr_embed = rearrange(group_corr,'(B T) C H W -> B T C H W', T = class_num)
        return grouped_corr
    def forward(self, img_feats,dino_feat, text_feats, appearance_guidance,dino_guidance,clip_L4_L8_feat):
        """
        Arguments:
            img_feats: (B, C, H, W)
            text_feats: (B, T, P, C) T是类别的个数
            apperance_guidance: tuple of (B, C, H, W)
        """

        classes = None
        ################START Here modified by YCY START###############
        H = 24
        corr = self.correlation(img_feats, text_feats)
        group_corr = self.group_wise_correlation(img_feats,text_feats)
        dino_corr = self.correlation(dino_feat,text_feats)

        #corr = self.feature_map(img_feats, text_feats)
        #exit()
        # fused_corr = corr+dino_corr

        # exit()
        
        # fused_corr_embed = self.corr_embed(fused_corr)
        # fused_corr_embed1 = self.corr_fusion_embed_minimum(clip_corr = corr,dino_corr=dino_corr)
        fused_corr_embed,clip_embed_corr, dino_embed_corr  = self.corr_fusion_embed_seperate(clip_corr = corr,dino_corr=dino_corr,group_clip_corr = group_corr)

        # add the res here #

        fused_corr_embed = fused_corr_embed+clip_embed_corr

        # add the res here #
        # print(23333333333)
        # clip_corr_embed = self.corr_embed(corr)
        # dino_corr_embed = self.corr_embed(dino_corr)
        ################END Here modified by YCY END###############
        
        # print(clip_corr_embed)
        projected_guidance, projected_text_guidance, CLIP_projected_decoder_guidance,DINO_projected_decoder_guidance  = None, None, [None, None], [None,None]
        if self.guidance_projection is not None:
            projected_guidance = self.guidance_projection(appearance_guidance[0])
        if self.CLIP_decoder_guidance_projection is not None:
            CLIP_projected_decoder_guidance = [proj(g) for proj, g in zip(self.CLIP_decoder_guidance_projection, appearance_guidance[1:])]
        if self.DINO_decoder_guidance_projection is not None:
            DINO_projected_decoder_guidance = [proj(g) for proj, g in zip(self.DINO_decoder_guidance_projection, dino_guidance)]
        if self.text_guidance_projection is not None:
            text_feats = text_feats.mean(dim=-2)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            projected_text_guidance = self.text_guidance_projection(text_feats)

        for layer in self.layers:
            fused_corr_embed = layer(fused_corr_embed, projected_guidance, projected_text_guidance)
            # clip_corr_embed = layer(clip_corr_embed, projected_guidance, projected_text_guidance)
            # dino_corr_embed = layer(dino_corr_embed, projected_guidance, projected_text_guidance)


        # fusion_corr_embed = clip_corr_embed + dino_corr_embed

        # logit = self.conv_decoder(clip_corr_embed, projected_decoder_guidance)

        logit = self.Fusion_conv_decoer(fused_corr_embed, CLIP_projected_decoder_guidance,DINO_projected_decoder_guidance)

        
        return logit
class FusionAggregatorVer22(nn.Module):
    def __init__(self, 
        text_guidance_dim=512,
        text_guidance_proj_dim=128,
        appearance_guidance_dim=512,
        appearance_guidance_proj_dim=128,
        decoder_dims = (64, 32),
        decoder_guidance_dims=(256, 128),
        decoder_guidance_proj_dims=(32, 16),
        num_layers=4,
        nheads=4, 
        hidden_dim=128,
        pooling_size=(6, 6),
        feature_resolution=(24, 24),
        window_size=12,
        attention_type='linear',
        prompt_channel=1,
        pad_len=256,
    ) -> None:
        """
        Cost Aggregation Model for CAT-Seg
        Args:
            text_guidance_dim: Dimension of text guidance
            text_guidance_proj_dim: Dimension of projected text guidance
            appearance_guidance_dim: Dimension of appearance guidance
            appearance_guidance_proj_dim: Dimension of projected appearance guidance
            decoder_dims: Upsampling decoder dimensions
            decoder_guidance_dims: Upsampling decoder guidance dimensions
            decoder_guidance_proj_dims: Upsampling decoder guidance projected dimensions
            num_layers: Number of layers for the aggregator
            nheads: Number of attention heads
            hidden_dim: Hidden dimension for transformer blocks
            pooling_size: Pooling size for the class aggregation layer
                          To reduce computation, we apply pooling in class aggregation blocks to reduce the number of tokens during training
            feature_resolution: Feature resolution for spatial aggregation
            window_size: Window size for Swin block in spatial aggregation
            attention_type: Attention type for the class aggregation. 
            prompt_channel: Number of prompts for ensembling text features. Default: 1
            pad_len: Padding length for the class aggregation. Default: 256
                     pad_len enforces the class aggregation block to have a fixed length of tokens for all inputs
                     This means it either pads the sequence with learnable tokens in class aggregation,
                     or truncates the classes with the initial CLIP cosine-similarity scores.
                     Set pad_len to 0 to disable this feature.
            """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.layers = nn.ModuleList([
            AggregatorLayer(
                hidden_dim=hidden_dim, text_guidance_dim=text_guidance_proj_dim, appearance_guidance=appearance_guidance_proj_dim, 
                nheads=nheads, input_resolution=feature_resolution, pooling_size=pooling_size, window_size=window_size, attention_type=attention_type, pad_len=pad_len,
            ) for _ in range(num_layers)
        ])

        # self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.fusion_corr = nn.Conv2d(2*hidden_dim, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.guidance_projection = nn.Sequential(
            nn.Conv2d(appearance_guidance_dim, appearance_guidance_proj_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        ) if appearance_guidance_dim > 0 else None
        
        self.L4_proj = ClipIntermediateProjLayer()
        self.L8_proj = ClipIntermediateProjLayer()
        self.text_guidance_projection = nn.Sequential(
            nn.Linear(text_guidance_dim, text_guidance_proj_dim),
            nn.ReLU(),
        ) if text_guidance_dim > 0 else None

        self.CLIP_decoder_guidance_projection = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            ) for d, dp in zip(decoder_guidance_dims, decoder_guidance_proj_dims)
        ]) if decoder_guidance_dims[0] > 0 else None
        
        self.DINO_decoder_guidance_projection = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            ) for d, dp in zip(decoder_guidance_dims, decoder_guidance_proj_dims)
        ]) if decoder_guidance_dims[0] > 0 else None
        
        # self.decoder1 = Up(hidden_dim, decoder_dims[0], decoder_guidance_proj_dims[0])
        # self.decoder2 = Up(decoder_dims[0], decoder_dims[1], decoder_guidance_proj_dims[1])
        
        self.Fusiondecoder1=FusionUP(hidden_dim, decoder_dims[0], decoder_guidance_proj_dims[0])
        self.Fusiondecoder2=FusionUP(decoder_dims[0], decoder_dims[1], decoder_guidance_proj_dims[1])
        self.head = nn.Conv2d(decoder_dims[1], 1, kernel_size=3, stride=1, padding=1)
        self.sigma = nn.Parameter(torch.ones(2))
        self.pad_len = pad_len

    def feature_map(self, img_feats, text_feats):
        # concatenated feature volume for feature aggregation baselines
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        img_feats = repeat(img_feats, "B C H W -> B C T H W", T=text_feats.shape[1])
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        text_feats = text_feats.mean(dim=-2) # average text features over different prompts
        text_feats = F.normalize(text_feats, dim=-1) # B T C
        text_feats = repeat(text_feats, "B T C -> B C T H W", H=img_feats.shape[-2], W=img_feats.shape[-1])
        return torch.cat((img_feats, text_feats), dim=1) # B 2C T H W

    def correlation(self, img_feats, text_feats):
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        corr = torch.einsum('bchw, btpc -> bpthw', img_feats, text_feats)
        return corr

    def corr_embed(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B P T H W -> (B T) P H W')
        corr_embed = self.conv1(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed
    
    def corr_fusion_embed_minimum(self,clip_corr,dino_corr):
        # this one does not import a 1*1 conv
        # instead we modify the original embedding layer to adapt to the concatenated corr volume.
        B = clip_corr.shape[0]
        clip_corr = rearrange(clip_corr, 'B P T H W -> (B T) P H W')
        dino_corr = rearrange(dino_corr, 'B P T H W -> (B T) P H W')
        fused_corr = torch.cat([clip_corr,dino_corr],dim = 1)
        fused_corr = self.conv1_modified(fused_corr)
        fused_corr = rearrange(fused_corr, '(B T) C H W -> B C T H W', B=B)
        return fused_corr

    def corr_fusion_embed_seperate(self,clip_corr,dino_corr):
        # this one does not import a 1*1 conv
        # instead we modify the original embedding layer to adapt to the concatenated corr volume.
        B = clip_corr.shape[0]
        self.sigmoid = nn.Sigmoid()
        clip_corr = rearrange(clip_corr, 'B P T H W -> (B T) P H W')
        dino_corr = rearrange(dino_corr, 'B P T H W -> (B T) P H W')
        clip_embed_corr = self.conv1(clip_corr)
        dino_embed_corr = self.conv2(dino_corr)
        clip_embed_corr = self.sigmoid(clip_embed_corr)
        dino_embed_corr = self.sigmoid(dino_embed_corr)
        fused_corr = torch.cat([clip_embed_corr,dino_embed_corr],dim = 1)
        fused_corr = self.fusion_corr(fused_corr)
        fused_corr = self.sigmoid(fused_corr)
        fused_corr = rearrange(fused_corr, '(B T) C H W -> B C T H W', B=B)
        clip_embed_corr = rearrange(clip_embed_corr, '(B T) C H W -> B C T H W', B=B)
        dino_embed_corr = rearrange(dino_embed_corr, '(B T) C H W -> B C T H W', B=B)
        return fused_corr, clip_embed_corr, dino_embed_corr
        
        
    def corr_fusion_embed(self,clip_corr,dino_corr):
        B = clip_corr.shape[0]
        clip_corr = rearrange(clip_corr, 'B P T H W -> (B T) P H W')
        dino_corr = rearrange(dino_corr, 'B P T H W -> (B T) P H W')
        fused_corr = torch.cat([clip_corr,dino_corr],dim = 1)
        fused_corr = self.fusion_corr(fused_corr)
        fused_corr = self.conv1(fused_corr)
        fused_corr = rearrange(fused_corr, '(B T) C H W -> B C T H W', B=B)
        return fused_corr
        # exit()
        
    def corr_projection(self, x, proj):
        corr_embed = rearrange(x, 'B C T H W -> B T H W C')
        corr_embed = proj(corr_embed)
        corr_embed = rearrange(corr_embed, 'B T H W C -> B C T H W')
        return corr_embed

    def upsample(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = F.interpolate(corr_embed, scale_factor=2, mode='bilinear', align_corners=True)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed

    def conv_decoder(self, x, guidance):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = self.decoder1(corr_embed, guidance[0])
        corr_embed = self.decoder2(corr_embed, guidance[1])
        corr_embed = self.head(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
        return corr_embed
    
    def Fusion_conv_decoer(self, x, clip_guidance,dino_guidance):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = self.Fusiondecoder1(corr_embed, clip_guidance[0],dino_guidance[0])
        corr_embed = self.Fusiondecoder2(corr_embed, clip_guidance[1],dino_guidance[1])
        corr_embed = self.head(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
        return corr_embed
    def forward(self, img_feats,dino_feat, text_feats, appearance_guidance,dino_guidance,clip_L4_L8_feat):
        """
        Arguments:
            img_feats: (B, C, H, W)
            text_feats: (B, T, P, C) T是类别的个数
            apperance_guidance: tuple of (B, C, H, W)
        """

        classes = None
        ################START Here modified by YCY START###############
        H = 24
        corr = self.correlation(img_feats, text_feats)
        clip_L4_proj = self.L4_proj(rearrange(clip_L4_L8_feat[0],'B C H W -> B (H W) C'))
        clip_L4_proj = rearrange(clip_L4_proj,'B (H W) C -> B C H W',H=H)
        clip_L8_proj = self.L8_proj(rearrange(clip_L4_L8_feat[1],'B C H W -> B (H W) C'))
        clip_L8_proj = rearrange(clip_L8_proj,'B (H W) C -> B C H W',H=H)
        L_4_corr = self.correlation(clip_L4_proj,text_feats)
        L_8_corr = self.correlation(clip_L8_proj,text_feats)
        dino_corr = self.correlation(dino_feat,text_feats)

        trip_corr = corr+self.sigma[0]*L_4_corr + self.sigma[1]*L_8_corr
        #corr = self.feature_map(img_feats, text_feats)
        #exit()
        # fused_corr = corr+dino_corr

        # exit()
        
        # fused_corr_embed = self.corr_embed(fused_corr)
        # fused_corr_embed1 = self.corr_fusion_embed_minimum(clip_corr = corr,dino_corr=dino_corr)
        fused_corr_embed,clip_embed_corr, dino_embed_corr  = self.corr_fusion_embed_seperate(clip_corr = trip_corr,dino_corr=dino_corr)
        # add the res here #

        fused_corr_embed = fused_corr_embed+clip_embed_corr

        # add the res here #
        # print(23333333333)
        # clip_corr_embed = self.corr_embed(corr)
        # dino_corr_embed = self.corr_embed(dino_corr)
        ################END Here modified by YCY END###############
        
        # print(clip_corr_embed)
        projected_guidance, projected_text_guidance, CLIP_projected_decoder_guidance,DINO_projected_decoder_guidance  = None, None, [None, None], [None,None]
        if self.guidance_projection is not None:
            projected_guidance = self.guidance_projection(appearance_guidance[0])
        if self.CLIP_decoder_guidance_projection is not None:
            CLIP_projected_decoder_guidance = [proj(g) for proj, g in zip(self.CLIP_decoder_guidance_projection, appearance_guidance[1:])]
        if self.DINO_decoder_guidance_projection is not None:
            DINO_projected_decoder_guidance = [proj(g) for proj, g in zip(self.DINO_decoder_guidance_projection, dino_guidance)]
        if self.text_guidance_projection is not None:
            text_feats = text_feats.mean(dim=-2)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            projected_text_guidance = self.text_guidance_projection(text_feats)

        for layer in self.layers:
            fused_corr_embed = layer(fused_corr_embed, projected_guidance, projected_text_guidance)
            # clip_corr_embed = layer(clip_corr_embed, projected_guidance, projected_text_guidance)
            # dino_corr_embed = layer(dino_corr_embed, projected_guidance, projected_text_guidance)


        # fusion_corr_embed = clip_corr_embed + dino_corr_embed

        # logit = self.conv_decoder(clip_corr_embed, projected_decoder_guidance)

        logit = self.Fusion_conv_decoer(fused_corr_embed, CLIP_projected_decoder_guidance,DINO_projected_decoder_guidance)

        
        return logit
class FusionAggregatorVer21(nn.Module):
    def __init__(self, 
        text_guidance_dim=512,
        text_guidance_proj_dim=128,
        appearance_guidance_dim=512,
        appearance_guidance_proj_dim=128,
        # decoder_dims = (64, 32),
        decoder_dims = (128,96,64,32),
        decoder_guidance_dims=(256, 128),
        decoder_guidance_proj_dims=(32, 16),
        num_layers=4,
        nheads=4, 
        hidden_dim=128,
        pooling_size=(6, 6),
        feature_resolution=(24, 24),
        window_size=12,
        attention_type='linear',
        prompt_channel=1,
        pad_len=256,
    ) -> None:
        """
        Cost Aggregation Model for CAT-Seg
        Args:
            text_guidance_dim: Dimension of text guidance
            text_guidance_proj_dim: Dimension of projected text guidance
            appearance_guidance_dim: Dimension of appearance guidance
            appearance_guidance_proj_dim: Dimension of projected appearance guidance
            decoder_dims: Upsampling decoder dimensions
            decoder_guidance_dims: Upsampling decoder guidance dimensions
            decoder_guidance_proj_dims: Upsampling decoder guidance projected dimensions
            num_layers: Number of layers for the aggregator
            nheads: Number of attention heads
            hidden_dim: Hidden dimension for transformer blocks
            pooling_size: Pooling size for the class aggregation layer
                          To reduce computation, we apply pooling in class aggregation blocks to reduce the number of tokens during training
            feature_resolution: Feature resolution for spatial aggregation
            window_size: Window size for Swin block in spatial aggregation
            attention_type: Attention type for the class aggregation. 
            prompt_channel: Number of prompts for ensembling text features. Default: 1
            pad_len: Padding length for the class aggregation. Default: 256
                     pad_len enforces the class aggregation block to have a fixed length of tokens for all inputs
                     This means it either pads the sequence with learnable tokens in class aggregation,
                     or truncates the classes with the initial CLIP cosine-similarity scores.
                     Set pad_len to 0 to disable this feature.
            """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        # used_clip_layer = 12
        # we dropped the appearance_guidance_dim
        # we make it a must
        self.layers = nn.ModuleList([
            AggregatorLayer(
                hidden_dim=hidden_dim, text_guidance_dim=text_guidance_proj_dim, appearance_guidance=appearance_guidance_proj_dim, 
                nheads=nheads, input_resolution=feature_resolution, pooling_size=pooling_size, window_size=window_size, attention_type=attention_type, pad_len=pad_len,
            ) for _ in range(num_layers)
        ])
        # self.clip_dense_proj_layers = ClipDenseProjLayer(
        #     hidden_dim=768, last_dim=512, appearance_guidance_proj_dim = appearance_guidance_proj_dim
        #     )if appearance_guidance_dim > 0 else None
        
        # self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.conv1 = nn.Conv2d(12, hidden_dim, kernel_size=7, stride=1, padding=3)
        # self.guidance_projection = nn.Sequential(
        #     nn.Conv2d(appearance_guidance_dim, appearance_guidance_proj_dim, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        # ) if appearance_guidance_dim > 0 else None

        self.clip_dense_proj = ClipDenseProjLayer11(
            hidden_dim=768, last_dim=512, appearance_guidance_proj_dim = appearance_guidance_proj_dim
            )
        self.text_guidance_projection = nn.Sequential(
            nn.Linear(text_guidance_dim, text_guidance_proj_dim),
            nn.ReLU(),
        ) if text_guidance_dim > 0 else None
        self.decoder_guidance_projection = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            ) for d, dp in zip(decoder_guidance_dims, decoder_guidance_proj_dims)
        ]) if decoder_guidance_dims[0] > 0 else None

        self.decoder1 = Up(hidden_dim, decoder_dims[0], decoder_guidance_proj_dims[0])
        self.decoder2 = Up(decoder_dims[0], decoder_dims[1], decoder_guidance_proj_dims[1])
        # self.decoder3 = UPmy(decoder_dims[1], decoder_dims[2], 0)
        # self.decoder4 = UPmy(decoder_dims[2], decoder_dims[3], 0)
        # self.head = nn.Conv2d(decoder_dims[1], 1, kernel_size=3, stride=1, padding=1)
        # self.clip_dense_proj_layers = ClipDenseProjLayer(
        #     hidden_dim=768, last_dim=512, appearance_guidance_proj_dim = appearance_guidance_proj_dim
        #     )
        self.head = nn.Conv2d(decoder_dims[1], 1, kernel_size=3, stride=1, padding=1)
        # self.upsample1 = nn.ConvTranspose2d(768, 256, kernel_size=2, stride=2)
        # self.upsample2 = nn.ConvTranspose2d(768, 128, kernel_size=4, stride=4)
        self.pad_len = pad_len
    
    def feature_map(self, img_feats, text_feats):
        # concatenated feature volume for feature aggregation baselines
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        img_feats = repeat(img_feats, "B C H W -> B C T H W", T=text_feats.shape[1])
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        text_feats = text_feats.mean(dim=-2) # average text features over different prompts
        text_feats = F.normalize(text_feats, dim=-1) # B T C
        text_feats = repeat(text_feats, "B T C -> B C T H W", H=img_feats.shape[-2], W=img_feats.shape[-1])
        return torch.cat((img_feats, text_feats), dim=1) # B 2C T H W

    def correlation(self, img_feats, text_feats):
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        corr = torch.einsum('bchw, btpc -> bpthw', img_feats, text_feats)
        return corr

    def corr_embed(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B P T H W -> (B T) P H W')
        corr_embed = self.conv1(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed
    def corr_embed_my(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = self.conv1(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed    
    def corr_projection(self, x, proj):
        corr_embed = rearrange(x, 'B C T H W -> B T H W C')
        corr_embed = proj(corr_embed)
        corr_embed = rearrange(corr_embed, 'B T H W C -> B C T H W')
        return corr_embed

    def upsample(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = F.interpolate(corr_embed, scale_factor=2, mode='bilinear', align_corners=True)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed

    def conv_decoder(self, x, guidance):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = self.decoder1(corr_embed, guidance[0])
        corr_embed = self.decoder2(corr_embed, guidance[1])
        # corr_embed = self.decoder3(corr_embed, None)
        # corr_embed = self.decoder4(corr_embed, None)
        corr_embed = self.head(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
        return corr_embed
    
    def forward(self, img_feats, text_feats, appearance_guidance):
        """
        Arguments:
            img_feats: (B, C, H, W)
            text_feats: (B, T, P, C) T是类别的个数
            apperance_guidance: tuple of (B, C, H, W)
        """

 

        # corr = self.feature_map(img_feats, text_feats)
        
        H = img_feats.shape[-2]
        projected_guidance, projected_text_guidance, projected_decoder_guidance = None, None, [None, None]
        # print('dense CV')
        corr_list = []
        proj_dense_clip_feat = self.clip_dense_proj(appearance_guidance)
        for feat in proj_dense_clip_feat:
            feat = rearrange(feat,'B ( H W ) C -> B C H W', H = H)

            corr = self.correlation(feat, text_feats)
            corr_list.append(corr)

        corr_group = torch.stack(corr_list,dim=1)
        # corr_embed = self.corr_embed(corr_last)
        corr_group = rearrange(corr_group,"B C 1 T H W -> B C T H W")
        corr_embed = self.corr_embed_my(corr_group)


        # clip_L4 = rearrange(appearance_guidance[3], "B (H W) C -> B C H W", H=24)
        # clip_L8 = rearrange(appearance_guidance[7], "B (H W) C -> B C H W", H=24)
        # res4 = self.upsample1(clip_L4)
        # res5 = self.upsample2(clip_L8)
        res4, res5 = None, None
        projected_decoder_guidance =[None,None]
        if self.decoder_guidance_projection is not None:
            #print('decod_guid')
            projected_decoder_guidance = [proj(g) for proj, g in zip(self.decoder_guidance_projection, [res4,res5])]

        if self.text_guidance_projection is not None:
            #print('textGuid')
            text_feats = text_feats.mean(dim=-2)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            projected_text_guidance = self.text_guidance_projection(text_feats)

        for layer in self.layers:
            # print('enc')
            corr_embed = layer(corr_embed, projected_guidance, projected_text_guidance)

        logit = self.conv_decoder(corr_embed, projected_decoder_guidance)

 
        return logit

class FusionAggregatorVer20(nn.Module):
    def __init__(self, 
        text_guidance_dim=512,
        text_guidance_proj_dim=128,
        appearance_guidance_dim=512,
        appearance_guidance_proj_dim=128,
        decoder_dims = (64, 32),
        decoder_guidance_dims=(256, 128),
        decoder_guidance_proj_dims=(32, 16),
        num_layers=4,
        nheads=4, 
        hidden_dim=128,
        pooling_size=(6, 6),
        feature_resolution=(24, 24),
        window_size=12,
        attention_type='linear',
        prompt_channel=1,
        pad_len=256,
    ) -> None:
        """
        Cost Aggregation Model for CAT-Seg
        Args:
            text_guidance_dim: Dimension of text guidance
            text_guidance_proj_dim: Dimension of projected text guidance
            appearance_guidance_dim: Dimension of appearance guidance
            appearance_guidance_proj_dim: Dimension of projected appearance guidance
            decoder_dims: Upsampling decoder dimensions
            decoder_guidance_dims: Upsampling decoder guidance dimensions
            decoder_guidance_proj_dims: Upsampling decoder guidance projected dimensions
            num_layers: Number of layers for the aggregator
            nheads: Number of attention heads
            hidden_dim: Hidden dimension for transformer blocks
            pooling_size: Pooling size for the class aggregation layer
                          To reduce computation, we apply pooling in class aggregation blocks to reduce the number of tokens during training
            feature_resolution: Feature resolution for spatial aggregation
            window_size: Window size for Swin block in spatial aggregation
            attention_type: Attention type for the class aggregation. 
            prompt_channel: Number of prompts for ensembling text features. Default: 1
            pad_len: Padding length for the class aggregation. Default: 256
                     pad_len enforces the class aggregation block to have a fixed length of tokens for all inputs
                     This means it either pads the sequence with learnable tokens in class aggregation,
                     or truncates the classes with the initial CLIP cosine-similarity scores.
                     Set pad_len to 0 to disable this feature.
            """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.layers = nn.ModuleList([
            AggregatorLayer(
                hidden_dim=hidden_dim, text_guidance_dim=text_guidance_proj_dim, appearance_guidance=appearance_guidance_proj_dim, 
                nheads=nheads, input_resolution=feature_resolution, pooling_size=pooling_size, window_size=window_size, attention_type=attention_type, pad_len=pad_len,
            ) for _ in range(num_layers)
        ])

        self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)

        self.guidance_projection = nn.Sequential(
            nn.Conv2d(appearance_guidance_dim, appearance_guidance_proj_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        ) if appearance_guidance_dim > 0 else None
        
        self.text_guidance_projection = nn.Sequential(
            nn.Linear(text_guidance_dim, text_guidance_proj_dim),
            nn.ReLU(),
        ) if text_guidance_dim > 0 else None

        self.decoder_guidance_projection = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            ) for d, dp in zip(decoder_guidance_dims, decoder_guidance_proj_dims)
        ]) if decoder_guidance_dims[0] > 0 else None

        self.decoder1 = Up(hidden_dim, decoder_dims[0], decoder_guidance_proj_dims[0])
        self.decoder2 = Up(decoder_dims[0], decoder_dims[1], decoder_guidance_proj_dims[1])
        self.head = nn.Conv2d(decoder_dims[1], 1, kernel_size=3, stride=1, padding=1)

        self.pad_len = pad_len

    def feature_map(self, img_feats, text_feats):
        # concatenated feature volume for feature aggregation baselines
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        img_feats = repeat(img_feats, "B C H W -> B C T H W", T=text_feats.shape[1])
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        text_feats = text_feats.mean(dim=-2) # average text features over different prompts
        text_feats = F.normalize(text_feats, dim=-1) # B T C
        text_feats = repeat(text_feats, "B T C -> B C T H W", H=img_feats.shape[-2], W=img_feats.shape[-1])
        return torch.cat((img_feats, text_feats), dim=1) # B 2C T H W

    def correlation(self, img_feats, text_feats):
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        corr = torch.einsum('bchw, btpc -> bpthw', img_feats, text_feats)
        return corr

    def corr_embed(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B P T H W -> (B T) P H W')
        corr_embed = self.conv1(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed
    
    def corr_projection(self, x, proj):
        corr_embed = rearrange(x, 'B C T H W -> B T H W C')
        corr_embed = proj(corr_embed)
        corr_embed = rearrange(corr_embed, 'B T H W C -> B C T H W')
        return corr_embed

    def upsample(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = F.interpolate(corr_embed, scale_factor=2, mode='bilinear', align_corners=True)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed

    def conv_decoder(self, x, guidance):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = self.decoder1(corr_embed, guidance[0])
        # print('111 exit')
        corr_embed = self.decoder2(corr_embed, guidance[1])
        corr_embed = self.head(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
        return corr_embed
    
    def forward(self, img_feats, text_feats, appearance_guidance):
        """
        Arguments:
            img_feats: (B, C, H, W)
            text_feats: (B, T, P, C) T是类别的个数
            apperance_guidance: tuple of (B, C, H, W)
        """

        classes = None

        corr = self.correlation(img_feats, text_feats)
        if self.pad_len > 0 and text_feats.size(1) > self.pad_len:
            avg = corr.permute(0, 2, 1, 3, 4).flatten(-3).max(dim=-1)[0] 
            classes = avg.topk(self.pad_len, dim=-1, sorted=False)[1]
            th_text = F.normalize(text_feats, dim=-1)
            th_text = torch.gather(th_text, dim=1, index=classes[..., None, None].expand(-1, -1, th_text.size(-2), th_text.size(-1)))
            orig_clases = text_feats.size(1)
            img_feats = F.normalize(img_feats, dim=1) # B C H W
            text_feats = th_text
            corr = torch.einsum('bchw, btpc -> bpthw', img_feats, th_text)
        #corr = self.feature_map(img_feats, text_feats)
        corr_embed = self.corr_embed(corr)

        projected_guidance, projected_text_guidance, projected_decoder_guidance = None, None, [None, None]
        if self.guidance_projection is not None:
            projected_guidance = self.guidance_projection(appearance_guidance[0])
        if self.decoder_guidance_projection is not None:
            projected_decoder_guidance = [proj(g) for proj, g in zip(self.decoder_guidance_projection, appearance_guidance[1:])]

        if self.text_guidance_projection is not None:
            text_feats = text_feats.mean(dim=-2)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            projected_text_guidance = self.text_guidance_projection(text_feats)

        for layer in self.layers:
            corr_embed = layer(corr_embed, projected_guidance, projected_text_guidance)

        logit = self.conv_decoder(corr_embed, projected_decoder_guidance)
        if classes is not None:
            out = torch.full((logit.size(0), orig_clases, logit.size(2), logit.size(3)), -100., device=logit.device)
            out.scatter_(dim=1, index=classes[..., None, None].expand(-1, -1, logit.size(-2), logit.size(-1)), src=logit)
            logit = out
        return logit

class FusionAggregatorVer14k(nn.Module):
    def __init__(self, 
        text_guidance_dim=512,
        text_guidance_proj_dim=128,
        appearance_guidance_dim=512,
        appearance_guidance_proj_dim=128,
        # decoder_dims = (64, 32),
        decoder_dims = (128,96,64,32),
        decoder_guidance_dims=(256, 128),
        decoder_guidance_proj_dims=(32, 16),
        num_layers=4,
        nheads=4, 
        hidden_dim=128,
        pooling_size=(6, 6),
        feature_resolution=(24, 24),
        window_size=12,
        attention_type='linear',
        prompt_channel=1,
        pad_len=256,
    ) -> None:
        """
        Cost Aggregation Model for CAT-Seg
        Args:
            text_guidance_dim: Dimension of text guidance
            text_guidance_proj_dim: Dimension of projected text guidance
            appearance_guidance_dim: Dimension of appearance guidance
            appearance_guidance_proj_dim: Dimension of projected appearance guidance
            decoder_dims: Upsampling decoder dimensions
            decoder_guidance_dims: Upsampling decoder guidance dimensions
            decoder_guidance_proj_dims: Upsampling decoder guidance projected dimensions
            num_layers: Number of layers for the aggregator
            nheads: Number of attention heads
            hidden_dim: Hidden dimension for transformer blocks
            pooling_size: Pooling size for the class aggregation layer
                          To reduce computation, we apply pooling in class aggregation blocks to reduce the number of tokens during training
            feature_resolution: Feature resolution for spatial aggregation
            window_size: Window size for Swin block in spatial aggregation
            attention_type: Attention type for the class aggregation. 
            prompt_channel: Number of prompts for ensembling text features. Default: 1
            pad_len: Padding length for the class aggregation. Default: 256
                     pad_len enforces the class aggregation block to have a fixed length of tokens for all inputs
                     This means it either pads the sequence with learnable tokens in class aggregation,
                     or truncates the classes with the initial CLIP cosine-similarity scores.
                     Set pad_len to 0 to disable this feature.
            """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.layers = nn.ModuleList([
            AggregatorLayer(
                hidden_dim=hidden_dim, text_guidance_dim=text_guidance_proj_dim, appearance_guidance=appearance_guidance_proj_dim, 
                nheads=nheads, input_resolution=feature_resolution, pooling_size=pooling_size, window_size=window_size, attention_type=attention_type, pad_len=pad_len,
            ) for _ in range(num_layers)
        ])

        self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)

        self.guidance_projection = nn.Sequential(
            nn.Conv2d(appearance_guidance_dim, appearance_guidance_proj_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        ) if appearance_guidance_dim > 0 else None

        self.text_guidance_projection = nn.Sequential(
            nn.Linear(text_guidance_dim, text_guidance_proj_dim),
            nn.ReLU(),
        ) if text_guidance_dim > 0 else None

        self.decoder_guidance_projection = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            ) for d, dp in zip(decoder_guidance_dims, decoder_guidance_proj_dims)
        ]) if decoder_guidance_dims[0] > 0 else None
        self.generic_class_attention1 = GenericClassandSpatialAttentionModuleVer3(hidden_dim)
        self.generic_class_attention2 = GenericClassandSpatialAttentionModuleVer3(decoder_dims[0])
        print('brilliant')
        # exit()
        # print('success4123')
        self.decoder1 = UPmy(hidden_dim, decoder_dims[0], decoder_guidance_proj_dims[0])
        self.decoder2 = UPmy(decoder_dims[0], decoder_dims[1], decoder_guidance_proj_dims[1])
        self.decoder3 = UPmy(decoder_dims[1], decoder_dims[2], 0)
        self.decoder4 = UPmy(decoder_dims[2], decoder_dims[3], 0)
        # self.head = nn.Conv2d(decoder_dims[1], 1, kernel_size=3, stride=1, padding=1)
        self.head = nn.Conv2d(decoder_dims[3], 1, kernel_size=3, stride=1, padding=1)

        self.pad_len = pad_len

    def feature_map(self, img_feats, text_feats):
        # concatenated feature volume for feature aggregation baselines
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        img_feats = repeat(img_feats, "B C H W -> B C T H W", T=text_feats.shape[1])
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        text_feats = text_feats.mean(dim=-2) # average text features over different prompts
        text_feats = F.normalize(text_feats, dim=-1) # B T C
        text_feats = repeat(text_feats, "B T C -> B C T H W", H=img_feats.shape[-2], W=img_feats.shape[-1])
        return torch.cat((img_feats, text_feats), dim=1) # B 2C T H W

    def correlation(self, img_feats, text_feats):
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        corr = torch.einsum('bchw, btpc -> bpthw', img_feats, text_feats)
        return corr

    def corr_embed(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B P T H W -> (B T) P H W')
        corr_embed = self.conv1(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed
    
    def corr_projection(self, x, proj):
        corr_embed = rearrange(x, 'B C T H W -> B T H W C')
        corr_embed = proj(corr_embed)
        corr_embed = rearrange(corr_embed, 'B T H W C -> B C T H W')
        return corr_embed

    def upsample(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = F.interpolate(corr_embed, scale_factor=2, mode='bilinear', align_corners=True)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed
    def conv_decoder_with_generic_class_attn(self, x, guidance):
        # x: B C T H W
        B = x.shape[0]
        corr_embed = self.generic_class_attention1(rearrange(x,'B C T H W -> B T C H W'))
        corr_embed = rearrange(corr_embed, 'B T C H W -> (B T) C H W')
        corr_embed = self.decoder1(corr_embed, guidance[0])
        corr_embed = self.generic_class_attention2(rearrange(corr_embed,'(B T) C H W-> B T C H W', B = B))
        corr_embed = rearrange(corr_embed, 'B T C H W -> (B T) C H W')
        corr_embed = self.decoder2(corr_embed, guidance[1])
        corr_embed = self.decoder3(corr_embed, None)
        corr_embed = self.decoder4(corr_embed, None)
        corr_embed = self.head(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
        return corr_embed
    def conv_decoder_with_generic_class_spatial_attn(self, x, guidance):
        B = x.shape[0]
        corr_embed = self.generic_class_attention1(rearrange(x,'B C T H W -> B T C H W'))
        corr_embed = rearrange(corr_embed, 'B T C H W -> (B T) C H W')
        corr_embed = self.decoder1(corr_embed, guidance[0])
        corr_embed = self.generic_class_attention2(rearrange(corr_embed,'(B T) C H W-> B T C H W', B = B))
        corr_embed = rearrange(corr_embed, 'B T C H W -> (B T) C H W')
        corr_embed = self.decoder2(corr_embed, guidance[1])
        corr_embed = self.decoder3(corr_embed, None)
        corr_embed = self.decoder4(corr_embed, None)
        corr_embed = self.head(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
        return corr_embed
    def conv_decoder(self, x, guidance):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = self.decoder1(corr_embed, guidance[0])
        corr_embed = self.decoder2(corr_embed, guidance[1])
        corr_embed = self.decoder3(corr_embed, None)
        corr_embed = self.decoder4(corr_embed, None)
        corr_embed = self.head(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
        return corr_embed

    def forward(self, img_feats, text_feats, appearance_guidance):
        """
        Arguments:
            img_feats: (B, C, H, W)
            text_feats: (B, T, P, C) T是类别的个数
            apperance_guidance: tuple of (B, C, H, W)
        """

        classes = None

        corr = self.correlation(img_feats, text_feats)
        if self.pad_len > 0 and text_feats.size(1) > self.pad_len:
            avg = corr.permute(0, 2, 1, 3, 4).flatten(-3).max(dim=-1)[0] 
            classes = avg.topk(self.pad_len, dim=-1, sorted=False)[1]
            th_text = F.normalize(text_feats, dim=-1)
            th_text = torch.gather(th_text, dim=1, index=classes[..., None, None].expand(-1, -1, th_text.size(-2), th_text.size(-1)))
            orig_clases = text_feats.size(1)
            img_feats = F.normalize(img_feats, dim=1) # B C H W
            text_feats = th_text
            corr = torch.einsum('bchw, btpc -> bpthw', img_feats, th_text)
        #corr = self.feature_map(img_feats, text_feats)
        batch_size = corr.shape[0]
        corr_embed = self.corr_embed(corr)

        projected_guidance, projected_text_guidance, projected_decoder_guidance = None, None, [None, None]
        if self.guidance_projection is not None:
            projected_guidance = self.guidance_projection(appearance_guidance[0])

        if self.decoder_guidance_projection is not None:
            projected_decoder_guidance = [proj(g) for proj, g in zip(self.decoder_guidance_projection, appearance_guidance[1:])]

        if self.text_guidance_projection is not None:
            text_feats = text_feats.mean(dim=-2)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            projected_text_guidance = self.text_guidance_projection(text_feats)

        for layer in self.layers:
            corr_embed = layer(corr_embed, projected_guidance, projected_text_guidance)

        # logit = self.conv_decoder(corr_embed, projected_decoder_guidance)

        logit = self.conv_decoder_with_generic_class_attn(corr_embed, projected_decoder_guidance)

        if classes is not None:
            out = torch.full((logit.size(0), orig_clases, logit.size(2), logit.size(3)), -100., device=logit.device)
            out.scatter_(dim=1, index=classes[..., None, None].expand(-1, -1, logit.size(-2), logit.size(-1)), src=logit)
            logit = out
 
        return logit

class FusionAggregatorVer14j(nn.Module):
    def __init__(self, 
        text_guidance_dim=512,
        text_guidance_proj_dim=128,
        appearance_guidance_dim=512,
        appearance_guidance_proj_dim=128,
        # decoder_dims = (64, 32),
        decoder_dims = (128,96,64,32),
        decoder_guidance_dims=(256, 128),
        decoder_guidance_proj_dims=(32, 16),
        num_layers=4,
        nheads=4, 
        hidden_dim=128,
        pooling_size=(6, 6),
        feature_resolution=(24, 24),
        window_size=12,
        attention_type='linear',
        prompt_channel=1,
        pad_len=256,
    ) -> None:
        """
        Cost Aggregation Model for CAT-Seg
        Args:
            text_guidance_dim: Dimension of text guidance
            text_guidance_proj_dim: Dimension of projected text guidance
            appearance_guidance_dim: Dimension of appearance guidance
            appearance_guidance_proj_dim: Dimension of projected appearance guidance
            decoder_dims: Upsampling decoder dimensions
            decoder_guidance_dims: Upsampling decoder guidance dimensions
            decoder_guidance_proj_dims: Upsampling decoder guidance projected dimensions
            num_layers: Number of layers for the aggregator
            nheads: Number of attention heads
            hidden_dim: Hidden dimension for transformer blocks
            pooling_size: Pooling size for the class aggregation layer
                          To reduce computation, we apply pooling in class aggregation blocks to reduce the number of tokens during training
            feature_resolution: Feature resolution for spatial aggregation
            window_size: Window size for Swin block in spatial aggregation
            attention_type: Attention type for the class aggregation. 
            prompt_channel: Number of prompts for ensembling text features. Default: 1
            pad_len: Padding length for the class aggregation. Default: 256
                     pad_len enforces the class aggregation block to have a fixed length of tokens for all inputs
                     This means it either pads the sequence with learnable tokens in class aggregation,
                     or truncates the classes with the initial CLIP cosine-similarity scores.
                     Set pad_len to 0 to disable this feature.
            """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.layers = nn.ModuleList([
            AggregatorLayer(
                hidden_dim=hidden_dim, text_guidance_dim=text_guidance_proj_dim, appearance_guidance=appearance_guidance_proj_dim, 
                nheads=nheads, input_resolution=feature_resolution, pooling_size=pooling_size, window_size=window_size, attention_type=attention_type, pad_len=pad_len,
            ) for _ in range(num_layers)
        ])

        self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)

        self.guidance_projection = nn.Sequential(
            nn.Conv2d(appearance_guidance_dim, appearance_guidance_proj_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        ) if appearance_guidance_dim > 0 else None

        self.text_guidance_projection = nn.Sequential(
            nn.Linear(text_guidance_dim, text_guidance_proj_dim),
            nn.ReLU(),
        ) if text_guidance_dim > 0 else None

        self.decoder_guidance_projection = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            ) for d, dp in zip(decoder_guidance_dims, decoder_guidance_proj_dims)
        ]) if decoder_guidance_dims[0] > 0 else None
        self.generic_class_attention1 = GenericClassandSpatialAttentionModuleVer2(hidden_dim)
        self.generic_class_attention2 = GenericClassandSpatialAttentionModuleVer2(decoder_dims[0])
        # print('success4123')
        self.decoder1 = UPmy(hidden_dim, decoder_dims[0], decoder_guidance_proj_dims[0])
        self.decoder2 = UPmy(decoder_dims[0], decoder_dims[1], decoder_guidance_proj_dims[1])
        self.decoder3 = UPmy(decoder_dims[1], decoder_dims[2], 0)
        self.decoder4 = UPmy(decoder_dims[2], decoder_dims[3], 0)
        # self.head = nn.Conv2d(decoder_dims[1], 1, kernel_size=3, stride=1, padding=1)
        self.head = nn.Conv2d(decoder_dims[3], 1, kernel_size=3, stride=1, padding=1)

        self.pad_len = pad_len

    def feature_map(self, img_feats, text_feats):
        # concatenated feature volume for feature aggregation baselines
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        img_feats = repeat(img_feats, "B C H W -> B C T H W", T=text_feats.shape[1])
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        text_feats = text_feats.mean(dim=-2) # average text features over different prompts
        text_feats = F.normalize(text_feats, dim=-1) # B T C
        text_feats = repeat(text_feats, "B T C -> B C T H W", H=img_feats.shape[-2], W=img_feats.shape[-1])
        return torch.cat((img_feats, text_feats), dim=1) # B 2C T H W

    def correlation(self, img_feats, text_feats):
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        corr = torch.einsum('bchw, btpc -> bpthw', img_feats, text_feats)
        return corr

    def corr_embed(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B P T H W -> (B T) P H W')
        corr_embed = self.conv1(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed
    
    def corr_projection(self, x, proj):
        corr_embed = rearrange(x, 'B C T H W -> B T H W C')
        corr_embed = proj(corr_embed)
        corr_embed = rearrange(corr_embed, 'B T H W C -> B C T H W')
        return corr_embed

    def upsample(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = F.interpolate(corr_embed, scale_factor=2, mode='bilinear', align_corners=True)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed
    def conv_decoder_with_generic_class_attn(self, x, guidance):
        # x: B C T H W
        B = x.shape[0]
        corr_embed = self.generic_class_attention1(rearrange(x,'B C T H W -> B T C H W'))
        corr_embed = rearrange(corr_embed, 'B T C H W -> (B T) C H W')
        corr_embed = self.decoder1(corr_embed, guidance[0])
        corr_embed = self.generic_class_attention2(rearrange(corr_embed,'(B T) C H W-> B T C H W', B = B))
        corr_embed = rearrange(corr_embed, 'B T C H W -> (B T) C H W')
        corr_embed = self.decoder2(corr_embed, guidance[1])
        corr_embed = self.decoder3(corr_embed, None)
        corr_embed = self.decoder4(corr_embed, None)
        corr_embed = self.head(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
        return corr_embed
    def conv_decoder_with_generic_class_spatial_attn(self, x, guidance):
        B = x.shape[0]
        corr_embed = self.generic_class_attention1(rearrange(x,'B C T H W -> B T C H W'))
        corr_embed = rearrange(corr_embed, 'B T C H W -> (B T) C H W')
        corr_embed = self.decoder1(corr_embed, guidance[0])
        corr_embed = self.generic_class_attention2(rearrange(corr_embed,'(B T) C H W-> B T C H W', B = B))
        corr_embed = rearrange(corr_embed, 'B T C H W -> (B T) C H W')
        corr_embed = self.decoder2(corr_embed, guidance[1])
        corr_embed = self.decoder3(corr_embed, None)
        corr_embed = self.decoder4(corr_embed, None)
        corr_embed = self.head(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
        return corr_embed
    def conv_decoder(self, x, guidance):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = self.decoder1(corr_embed, guidance[0])
        corr_embed = self.decoder2(corr_embed, guidance[1])
        corr_embed = self.decoder3(corr_embed, None)
        corr_embed = self.decoder4(corr_embed, None)
        corr_embed = self.head(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
        return corr_embed

    def forward(self, img_feats, text_feats, appearance_guidance):
        """
        Arguments:
            img_feats: (B, C, H, W)
            text_feats: (B, T, P, C) T是类别的个数
            apperance_guidance: tuple of (B, C, H, W)
        """

        classes = None

        corr = self.correlation(img_feats, text_feats)
        if self.pad_len > 0 and text_feats.size(1) > self.pad_len:
            avg = corr.permute(0, 2, 1, 3, 4).flatten(-3).max(dim=-1)[0] 
            classes = avg.topk(self.pad_len, dim=-1, sorted=False)[1]
            th_text = F.normalize(text_feats, dim=-1)
            th_text = torch.gather(th_text, dim=1, index=classes[..., None, None].expand(-1, -1, th_text.size(-2), th_text.size(-1)))
            orig_clases = text_feats.size(1)
            img_feats = F.normalize(img_feats, dim=1) # B C H W
            text_feats = th_text
            corr = torch.einsum('bchw, btpc -> bpthw', img_feats, th_text)
        #corr = self.feature_map(img_feats, text_feats)
        batch_size = corr.shape[0]
        corr_embed = self.corr_embed(corr)

        projected_guidance, projected_text_guidance, projected_decoder_guidance = None, None, [None, None]
        if self.guidance_projection is not None:
            projected_guidance = self.guidance_projection(appearance_guidance[0])

        if self.decoder_guidance_projection is not None:
            projected_decoder_guidance = [proj(g) for proj, g in zip(self.decoder_guidance_projection, appearance_guidance[1:])]

        if self.text_guidance_projection is not None:
            text_feats = text_feats.mean(dim=-2)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            projected_text_guidance = self.text_guidance_projection(text_feats)

        for layer in self.layers:
            corr_embed = layer(corr_embed, projected_guidance, projected_text_guidance)

        # logit = self.conv_decoder(corr_embed, projected_decoder_guidance)

        logit = self.conv_decoder_with_generic_class_attn(corr_embed, projected_decoder_guidance)

        if classes is not None:
            out = torch.full((logit.size(0), orig_clases, logit.size(2), logit.size(3)), -100., device=logit.device)
            out.scatter_(dim=1, index=classes[..., None, None].expand(-1, -1, logit.size(-2), logit.size(-1)), src=logit)
            logit = out
 
        return logit
    
class FusionAggregatorVer14i(nn.Module):
    def __init__(self, 
        text_guidance_dim=512,
        text_guidance_proj_dim=128,
        appearance_guidance_dim=512,
        appearance_guidance_proj_dim=128,
        # decoder_dims = (64, 32),
        decoder_dims = (128,96,64,32),
        decoder_guidance_dims=(256, 128),
        decoder_guidance_proj_dims=(32, 16),
        num_layers=4,
        nheads=4, 
        hidden_dim=128,
        pooling_size=(6, 6),
        feature_resolution=(24, 24),
        window_size=12,
        attention_type='linear',
        prompt_channel=1,
        pad_len=256,
    ) -> None:
        """
        Cost Aggregation Model for CAT-Seg
        Args:
            text_guidance_dim: Dimension of text guidance
            text_guidance_proj_dim: Dimension of projected text guidance
            appearance_guidance_dim: Dimension of appearance guidance
            appearance_guidance_proj_dim: Dimension of projected appearance guidance
            decoder_dims: Upsampling decoder dimensions
            decoder_guidance_dims: Upsampling decoder guidance dimensions
            decoder_guidance_proj_dims: Upsampling decoder guidance projected dimensions
            num_layers: Number of layers for the aggregator
            nheads: Number of attention heads
            hidden_dim: Hidden dimension for transformer blocks
            pooling_size: Pooling size for the class aggregation layer
                          To reduce computation, we apply pooling in class aggregation blocks to reduce the number of tokens during training
            feature_resolution: Feature resolution for spatial aggregation
            window_size: Window size for Swin block in spatial aggregation
            attention_type: Attention type for the class aggregation. 
            prompt_channel: Number of prompts for ensembling text features. Default: 1
            pad_len: Padding length for the class aggregation. Default: 256
                     pad_len enforces the class aggregation block to have a fixed length of tokens for all inputs
                     This means it either pads the sequence with learnable tokens in class aggregation,
                     or truncates the classes with the initial CLIP cosine-similarity scores.
                     Set pad_len to 0 to disable this feature.
            """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.layers = nn.ModuleList([
            AggregatorLayer(
                hidden_dim=hidden_dim, text_guidance_dim=text_guidance_proj_dim, appearance_guidance=appearance_guidance_proj_dim, 
                nheads=nheads, input_resolution=feature_resolution, pooling_size=pooling_size, window_size=window_size, attention_type=attention_type, pad_len=pad_len,
            ) for _ in range(num_layers)
        ])

        self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)

        self.guidance_projection = nn.Sequential(
            nn.Conv2d(appearance_guidance_dim, appearance_guidance_proj_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        ) if appearance_guidance_dim > 0 else None

        self.text_guidance_projection = nn.Sequential(
            nn.Linear(text_guidance_dim, text_guidance_proj_dim),
            nn.ReLU(),
        ) if text_guidance_dim > 0 else None

        self.decoder_guidance_projection = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            ) for d, dp in zip(decoder_guidance_dims, decoder_guidance_proj_dims)
        ]) if decoder_guidance_dims[0] > 0 else None
        self.generic_class_attention1 = GenericClassAttentionModuleVer2(hidden_dim)
        self.generic_class_attention2 = GenericClassAttentionModuleVer2(decoder_dims[0])
        # print('success4123')
        self.decoder1 = UPmy(hidden_dim, decoder_dims[0], decoder_guidance_proj_dims[0])
        self.decoder2 = UPmy(decoder_dims[0], decoder_dims[1], decoder_guidance_proj_dims[1])
        self.decoder3 = UPmy(decoder_dims[1], decoder_dims[2], 0)
        self.decoder4 = UPmy(decoder_dims[2], decoder_dims[3], 0)
        # self.head = nn.Conv2d(decoder_dims[1], 1, kernel_size=3, stride=1, padding=1)
        self.head = nn.Conv2d(decoder_dims[3], 1, kernel_size=3, stride=1, padding=1)

        self.pad_len = pad_len

    def feature_map(self, img_feats, text_feats):
        # concatenated feature volume for feature aggregation baselines
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        img_feats = repeat(img_feats, "B C H W -> B C T H W", T=text_feats.shape[1])
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        text_feats = text_feats.mean(dim=-2) # average text features over different prompts
        text_feats = F.normalize(text_feats, dim=-1) # B T C
        text_feats = repeat(text_feats, "B T C -> B C T H W", H=img_feats.shape[-2], W=img_feats.shape[-1])
        return torch.cat((img_feats, text_feats), dim=1) # B 2C T H W

    def correlation(self, img_feats, text_feats):
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        corr = torch.einsum('bchw, btpc -> bpthw', img_feats, text_feats)
        return corr

    def corr_embed(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B P T H W -> (B T) P H W')
        corr_embed = self.conv1(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed
    
    def corr_projection(self, x, proj):
        corr_embed = rearrange(x, 'B C T H W -> B T H W C')
        corr_embed = proj(corr_embed)
        corr_embed = rearrange(corr_embed, 'B T H W C -> B C T H W')
        return corr_embed

    def upsample(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = F.interpolate(corr_embed, scale_factor=2, mode='bilinear', align_corners=True)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed
    def conv_decoder_with_generic_class_attn(self, x, guidance):
        # x: B C T H W
        B = x.shape[0]
        corr_embed = self.generic_class_attention1(rearrange(x,'B C T H W -> B T C H W'))
        corr_embed = rearrange(corr_embed, 'B T C H W -> (B T) C H W')
        corr_embed = self.decoder1(corr_embed, guidance[0])
        corr_embed = self.generic_class_attention2(rearrange(corr_embed,'(B T) C H W-> B T C H W', B = B))
        corr_embed = rearrange(corr_embed, 'B T C H W -> (B T) C H W')
        corr_embed = self.decoder2(corr_embed, guidance[1])
        corr_embed = self.decoder3(corr_embed, None)
        corr_embed = self.decoder4(corr_embed, None)
        corr_embed = self.head(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
        return corr_embed
    def conv_decoder(self, x, guidance):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = self.decoder1(corr_embed, guidance[0])
        corr_embed = self.decoder2(corr_embed, guidance[1])
        corr_embed = self.decoder3(corr_embed, None)
        corr_embed = self.decoder4(corr_embed, None)
        corr_embed = self.head(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
        return corr_embed

    def forward(self, img_feats, text_feats, appearance_guidance):
        """
        Arguments:
            img_feats: (B, C, H, W)
            text_feats: (B, T, P, C) T是类别的个数
            apperance_guidance: tuple of (B, C, H, W)
        """

        classes = None

        corr = self.correlation(img_feats, text_feats)
        if self.pad_len > 0 and text_feats.size(1) > self.pad_len:
            avg = corr.permute(0, 2, 1, 3, 4).flatten(-3).max(dim=-1)[0] 
            classes = avg.topk(self.pad_len, dim=-1, sorted=False)[1]
            th_text = F.normalize(text_feats, dim=-1)
            th_text = torch.gather(th_text, dim=1, index=classes[..., None, None].expand(-1, -1, th_text.size(-2), th_text.size(-1)))
            orig_clases = text_feats.size(1)
            img_feats = F.normalize(img_feats, dim=1) # B C H W
            text_feats = th_text
            corr = torch.einsum('bchw, btpc -> bpthw', img_feats, th_text)
        #corr = self.feature_map(img_feats, text_feats)
        batch_size = corr.shape[0]
        corr_embed = self.corr_embed(corr)

        projected_guidance, projected_text_guidance, projected_decoder_guidance = None, None, [None, None]
        if self.guidance_projection is not None:
            projected_guidance = self.guidance_projection(appearance_guidance[0])

        if self.decoder_guidance_projection is not None:
            projected_decoder_guidance = [proj(g) for proj, g in zip(self.decoder_guidance_projection, appearance_guidance[1:])]

        if self.text_guidance_projection is not None:
            text_feats = text_feats.mean(dim=-2)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            projected_text_guidance = self.text_guidance_projection(text_feats)

        for layer in self.layers:
            corr_embed = layer(corr_embed, projected_guidance, projected_text_guidance)

        # logit = self.conv_decoder(corr_embed, projected_decoder_guidance)

        logit = self.conv_decoder_with_generic_class_attn(corr_embed, projected_decoder_guidance)

        if classes is not None:
            out = torch.full((logit.size(0), orig_clases, logit.size(2), logit.size(3)), -100., device=logit.device)
            out.scatter_(dim=1, index=classes[..., None, None].expand(-1, -1, logit.size(-2), logit.size(-1)), src=logit)
            logit = out
 
        return logit
class FusionAggregatorVer14h(nn.Module):
    def __init__(self, 
        text_guidance_dim=512,
        text_guidance_proj_dim=128,
        appearance_guidance_dim=512,
        appearance_guidance_proj_dim=128,
        # decoder_dims = (64, 32),
        decoder_dims = (128,96,64,32),
        decoder_guidance_dims=(256, 128),
        decoder_guidance_proj_dims=(32, 16),
        num_layers=4,
        nheads=4, 
        hidden_dim=128,
        pooling_size=(6, 6),
        feature_resolution=(24, 24),
        window_size=12,
        attention_type='linear',
        prompt_channel=1,
        pad_len=256,
    ) -> None:
        """
        Cost Aggregation Model for CAT-Seg
        Args:
            text_guidance_dim: Dimension of text guidance
            text_guidance_proj_dim: Dimension of projected text guidance
            appearance_guidance_dim: Dimension of appearance guidance
            appearance_guidance_proj_dim: Dimension of projected appearance guidance
            decoder_dims: Upsampling decoder dimensions
            decoder_guidance_dims: Upsampling decoder guidance dimensions
            decoder_guidance_proj_dims: Upsampling decoder guidance projected dimensions
            num_layers: Number of layers for the aggregator
            nheads: Number of attention heads
            hidden_dim: Hidden dimension for transformer blocks
            pooling_size: Pooling size for the class aggregation layer
                          To reduce computation, we apply pooling in class aggregation blocks to reduce the number of tokens during training
            feature_resolution: Feature resolution for spatial aggregation
            window_size: Window size for Swin block in spatial aggregation
            attention_type: Attention type for the class aggregation. 
            prompt_channel: Number of prompts for ensembling text features. Default: 1
            pad_len: Padding length for the class aggregation. Default: 256
                     pad_len enforces the class aggregation block to have a fixed length of tokens for all inputs
                     This means it either pads the sequence with learnable tokens in class aggregation,
                     or truncates the classes with the initial CLIP cosine-similarity scores.
                     Set pad_len to 0 to disable this feature.
            """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.layers = nn.ModuleList([
            AggregatorLayer(
                hidden_dim=hidden_dim, text_guidance_dim=text_guidance_proj_dim, appearance_guidance=appearance_guidance_proj_dim, 
                nheads=nheads, input_resolution=feature_resolution, pooling_size=pooling_size, window_size=window_size, attention_type=attention_type, pad_len=pad_len,
            ) for _ in range(num_layers)
        ])

        self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)

        self.guidance_projection = nn.Sequential(
            nn.Conv2d(appearance_guidance_dim, appearance_guidance_proj_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        ) if appearance_guidance_dim > 0 else None

        self.text_guidance_projection = nn.Sequential(
            nn.Linear(text_guidance_dim, text_guidance_proj_dim),
            nn.ReLU(),
        ) if text_guidance_dim > 0 else None

        self.decoder_guidance_projection = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            ) for d, dp in zip(decoder_guidance_dims, decoder_guidance_proj_dims)
        ]) if decoder_guidance_dims[0] > 0 else None
        self.generic_class_attention1 = GenericClassAttentionModule(hidden_dim)
        self.generic_class_attention2 = GenericClassAttentionModule(decoder_dims[0])
        self.decoder1 = UPmy(hidden_dim, decoder_dims[0], decoder_guidance_proj_dims[0])
        self.decoder2 = UPmy(decoder_dims[0], decoder_dims[1], decoder_guidance_proj_dims[1])
        self.decoder3 = UPmy(decoder_dims[1], decoder_dims[2], 0)
        self.decoder4 = UPmy(decoder_dims[2], decoder_dims[3], 0)
        # self.head = nn.Conv2d(decoder_dims[1], 1, kernel_size=3, stride=1, padding=1)
        self.head = nn.Conv2d(decoder_dims[3], 1, kernel_size=3, stride=1, padding=1)

        self.pad_len = pad_len

    def feature_map(self, img_feats, text_feats):
        # concatenated feature volume for feature aggregation baselines
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        img_feats = repeat(img_feats, "B C H W -> B C T H W", T=text_feats.shape[1])
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        text_feats = text_feats.mean(dim=-2) # average text features over different prompts
        text_feats = F.normalize(text_feats, dim=-1) # B T C
        text_feats = repeat(text_feats, "B T C -> B C T H W", H=img_feats.shape[-2], W=img_feats.shape[-1])
        return torch.cat((img_feats, text_feats), dim=1) # B 2C T H W

    def correlation(self, img_feats, text_feats):
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        corr = torch.einsum('bchw, btpc -> bpthw', img_feats, text_feats)
        return corr

    def corr_embed(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B P T H W -> (B T) P H W')
        corr_embed = self.conv1(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed
    
    def corr_projection(self, x, proj):
        corr_embed = rearrange(x, 'B C T H W -> B T H W C')
        corr_embed = proj(corr_embed)
        corr_embed = rearrange(corr_embed, 'B T H W C -> B C T H W')
        return corr_embed

    def upsample(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = F.interpolate(corr_embed, scale_factor=2, mode='bilinear', align_corners=True)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed
    def conv_decoder_with_generic_class_attn(self, x, guidance):
        # x: B C T H W
        B = x.shape[0]
        corr_embed = self.generic_class_attention1(rearrange(x,'B C T H W -> B T C H W'))
        corr_embed = rearrange(corr_embed, 'B T C H W -> (B T) C H W')
        corr_embed = self.decoder1(corr_embed, guidance[0])
        corr_embed = self.generic_class_attention2(rearrange(corr_embed,'(B T) C H W-> B T C H W', B = B))
        corr_embed = rearrange(corr_embed, 'B T C H W -> (B T) C H W')
        corr_embed = self.decoder2(corr_embed, guidance[1])
        corr_embed = self.decoder3(corr_embed, None)
        corr_embed = self.decoder4(corr_embed, None)
        corr_embed = self.head(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
        return corr_embed
    def conv_decoder(self, x, guidance):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = self.decoder1(corr_embed, guidance[0])
        corr_embed = self.decoder2(corr_embed, guidance[1])
        corr_embed = self.decoder3(corr_embed, None)
        corr_embed = self.decoder4(corr_embed, None)
        corr_embed = self.head(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
        return corr_embed

    def forward(self, img_feats, text_feats, appearance_guidance):
        """
        Arguments:
            img_feats: (B, C, H, W)
            text_feats: (B, T, P, C) T是类别的个数
            apperance_guidance: tuple of (B, C, H, W)
        """

        classes = None

        corr = self.correlation(img_feats, text_feats)
        if self.pad_len > 0 and text_feats.size(1) > self.pad_len:
            avg = corr.permute(0, 2, 1, 3, 4).flatten(-3).max(dim=-1)[0] 
            classes = avg.topk(self.pad_len, dim=-1, sorted=False)[1]
            th_text = F.normalize(text_feats, dim=-1)
            th_text = torch.gather(th_text, dim=1, index=classes[..., None, None].expand(-1, -1, th_text.size(-2), th_text.size(-1)))
            orig_clases = text_feats.size(1)
            img_feats = F.normalize(img_feats, dim=1) # B C H W
            text_feats = th_text
            corr = torch.einsum('bchw, btpc -> bpthw', img_feats, th_text)
        #corr = self.feature_map(img_feats, text_feats)
        batch_size = corr.shape[0]
        corr_embed = self.corr_embed(corr)

        projected_guidance, projected_text_guidance, projected_decoder_guidance = None, None, [None, None]
        if self.guidance_projection is not None:
            projected_guidance = self.guidance_projection(appearance_guidance[0])

        if self.decoder_guidance_projection is not None:
            projected_decoder_guidance = [proj(g) for proj, g in zip(self.decoder_guidance_projection, appearance_guidance[1:])]

        if self.text_guidance_projection is not None:
            text_feats = text_feats.mean(dim=-2)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            projected_text_guidance = self.text_guidance_projection(text_feats)

        for layer in self.layers:
            corr_embed = layer(corr_embed, projected_guidance, projected_text_guidance)

        # logit = self.conv_decoder(corr_embed, projected_decoder_guidance)

        logit = self.conv_decoder_with_generic_class_attn(corr_embed, projected_decoder_guidance)

        if classes is not None:
            out = torch.full((logit.size(0), orig_clases, logit.size(2), logit.size(3)), -100., device=logit.device)
            out.scatter_(dim=1, index=classes[..., None, None].expand(-1, -1, logit.size(-2), logit.size(-1)), src=logit)
            logit = out
 
        return logit

class FusionAggregatorVer14g(nn.Module):
    def __init__(self, 
        text_guidance_dim=512,
        text_guidance_proj_dim=128,
        appearance_guidance_dim=512,
        appearance_guidance_proj_dim=128,
        # decoder_dims = (64, 32),
        decoder_dims = (128,96,64,32),
        decoder_guidance_dims=(256, 128),
        decoder_guidance_proj_dims=(32, 16),
        num_layers=4,
        nheads=4, 
        hidden_dim=128,
        pooling_size=(6, 6),
        feature_resolution=(24, 24),
        window_size=12,
        attention_type='linear',
        prompt_channel=1,
        pad_len=256,
    ) -> None:
        """
        Cost Aggregation Model for CAT-Seg
        Args:
            text_guidance_dim: Dimension of text guidance
            text_guidance_proj_dim: Dimension of projected text guidance
            appearance_guidance_dim: Dimension of appearance guidance
            appearance_guidance_proj_dim: Dimension of projected appearance guidance
            decoder_dims: Upsampling decoder dimensions
            decoder_guidance_dims: Upsampling decoder guidance dimensions
            decoder_guidance_proj_dims: Upsampling decoder guidance projected dimensions
            num_layers: Number of layers for the aggregator
            nheads: Number of attention heads
            hidden_dim: Hidden dimension for transformer blocks
            pooling_size: Pooling size for the class aggregation layer
                          To reduce computation, we apply pooling in class aggregation blocks to reduce the number of tokens during training
            feature_resolution: Feature resolution for spatial aggregation
            window_size: Window size for Swin block in spatial aggregation
            attention_type: Attention type for the class aggregation. 
            prompt_channel: Number of prompts for ensembling text features. Default: 1
            pad_len: Padding length for the class aggregation. Default: 256
                     pad_len enforces the class aggregation block to have a fixed length of tokens for all inputs
                     This means it either pads the sequence with learnable tokens in class aggregation,
                     or truncates the classes with the initial CLIP cosine-similarity scores.
                     Set pad_len to 0 to disable this feature.
            """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_corr_projector = 3
        self.cat_corr_proj_dim=[1024,512,256,128]
        self.cat_corr_proj_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=self.cat_corr_proj_dim[proj_ind],
                                    out_channels=self.cat_corr_proj_dim[proj_ind+1], 
                                    kernel_size=7,stride=1,padding=3),
                nn.ReLU()
                ) 
            for proj_ind in range(self.num_corr_projector)])

        self.layers = nn.ModuleList([
            AggregatorLayer(
                hidden_dim=hidden_dim, text_guidance_dim=text_guidance_proj_dim, appearance_guidance=appearance_guidance_proj_dim, 
                nheads=nheads, input_resolution=feature_resolution, pooling_size=pooling_size, window_size=window_size, attention_type=attention_type, pad_len=pad_len,
            ) for _ in range(num_layers)
        ])
        
        # self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)

        self.guidance_projection = nn.Sequential(
            nn.Conv2d(appearance_guidance_dim, appearance_guidance_proj_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        ) if appearance_guidance_dim > 0 else None

        self.text_guidance_projection = nn.Sequential(
            nn.Linear(text_guidance_dim, text_guidance_proj_dim),
            nn.ReLU(),
        ) if text_guidance_dim > 0 else None

        self.decoder_guidance_projection = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            ) for d, dp in zip(decoder_guidance_dims, decoder_guidance_proj_dims)
        ]) if decoder_guidance_dims[0] > 0 else None
        # self.cat_corr_embed = nn.Conv2d(1024, hidden_dim, kernel_size=1, stride=1, padding=0)
        # Ver14e
        # self.gw_corr_embed = nn.Conv2d(32, hidden_dim, kernel_size=1, stride=1, padding=0)
        # self.fusion_corr_embed = nn.Conv2d(2*hidden_dim,hidden_dim,kernel_size=7, stride=1, padding=3)
        # Ver14e
        self.decoder1 = UPmy(hidden_dim, decoder_dims[0], decoder_guidance_proj_dims[0])
        self.decoder2 = UPmy(decoder_dims[0], decoder_dims[1], decoder_guidance_proj_dims[1])
        self.decoder3 = UPmy(decoder_dims[1], decoder_dims[2], 0)
        self.decoder4 = UPmy(decoder_dims[2], decoder_dims[3], 0)
        # self.head = nn.Conv2d(decoder_dims[1], 1, kernel_size=3, stride=1, padding=1)
        self.head = nn.Conv2d(decoder_dims[3], 1, kernel_size=3, stride=1, padding=1)

        self.pad_len = pad_len

    def feature_map(self, img_feats, text_feats):
        # concatenated feature volume for feature aggregation baselines
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        img_feats = repeat(img_feats, "B C H W -> B C T H W", T=text_feats.shape[1])
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        text_feats = text_feats.mean(dim=-2) # average text features over different prompts
        text_feats = F.normalize(text_feats, dim=-1) # B T C
        text_feats = repeat(text_feats, "B T C -> B C T H W", H=img_feats.shape[-2], W=img_feats.shape[-1])
        return torch.cat((img_feats, text_feats), dim=1) # B 2C T H W

    def correlation(self, img_feats, text_feats):
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        corr = torch.einsum('bchw, btpc -> bpthw', img_feats, text_feats)
        return corr
    # def group_wise_correlation(self, img_feats, text_feats, group_nums = 32):
    #     '''
    #     Return B P N T H W, wheareas N is the group number, P is prompt number, T is class number
    #     '''
    #     class_num = text_feats.shape[1]
    #     N_g = group_nums
    #     group_img_feats = rearrange(img_feats, 'B (C N) H W ->(B N) C H W', N=N_g)
    #     group_text_feats = rearrange(text_feats, 'B T P (C N) ->(B N) T P C', N=N_g)
    #     group_img_feats = F.normalize(group_img_feats, dim=1)
    #     group_text_feats = F.normalize(group_text_feats, dim = -1)
    #     group_corr = torch.einsum('bchw, btpc -> bpthw', group_img_feats, group_text_feats)
    #     group_corr = group_corr.squeeze(1)
    #     # group_corr = rearrange(group_corr,'(B N) T H W -> B N T H W', N = N_g)
    #     group_corr = rearrange(group_corr,'(B N) T H W -> (B T) N H W', N = N_g)
    #     group_corr_embed = self.gw_corr_embed(group_corr)
    #     #group_corr_embed = rearrange(group_corr,'(B T) C H W -> B T C H W', T = class_num)
    #     return group_corr_embed
    def concatenation_correlation(self, img_feats, text_feats):
        class_num = text_feats.shape[1]
        H, W = img_feats.shape[-2], img_feats.shape[-1]
        img_feats=img_feats.unsqueeze(1).repeat([1,class_num,1,1,1])
        img_feats = rearrange(img_feats, 'B T C H W -> (B T) C H W')
        text_feats=text_feats.unsqueeze(-1).unsqueeze(-1).repeat([1,1,1,1,H,W])
        text_feats = rearrange(text_feats,'B T 1 C H W -> (B T) C H W')
        cat_feats = torch.cat([img_feats,text_feats],dim=1) # C=256
        # cat_corr_embed = self.cat_corr_embed(cat_feats)
        #cat_corr_embed = rearrange(cat_corr_embed,'(B T) C H W -> B T C H W',T = class_num)
        return cat_feats
    # def corr_embed(self, x):
    #     B = x.shape[0]
    #     corr_embed = rearrange(x, 'B P T H W -> (B T) P H W')
    #     corr_embed = self.conv1(corr_embed)
    #     corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
    #     return corr_embed
    
    def corr_projection(self, x, proj):
        corr_embed = rearrange(x, 'B C T H W -> B T H W C')
        corr_embed = proj(corr_embed)
        corr_embed = rearrange(corr_embed, 'B T H W C -> B C T H W')
        return corr_embed

    def upsample(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = F.interpolate(corr_embed, scale_factor=2, mode='bilinear', align_corners=True)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed

    def conv_decoder(self, x, guidance):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = self.decoder1(corr_embed, guidance[0])
        corr_embed = self.decoder2(corr_embed, guidance[1])
        corr_embed = self.decoder3(corr_embed, None)
        corr_embed = self.decoder4(corr_embed, None)
        corr_embed = self.head(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
        return corr_embed
    
    def forward(self, img_feats, text_feats, appearance_guidance):
        """
        Arguments:
            img_feats: (B, C, H, W)
            text_feats: (B, T, P, C) T是类别的个数
            apperance_guidance: tuple of (B, C, H, W)
        """

        classes = None

        # corr = self.correlation(img_feats, text_feats)
        T = text_feats.shape[1]
        # gw_corr = self.group_wise_correlation(img_feats, text_feats)
        corr_embed = self.concatenation_correlation(img_feats, text_feats) # (B T) C H W
        # dual_corr = torch.cat([gw_corr,cat_corr],dim = 1 )
        # corr_embed = self.fusion_corr_embed(dual_corr)
        
        for proj_layer in self.cat_corr_proj_layers:
            corr_embed = proj_layer(corr_embed)
        
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W',T=T)

        #corr = self.feature_map(img_feats, text_feats)
        # corr_embed = self.corr_embed(corr)

        projected_guidance, projected_text_guidance, projected_decoder_guidance = None, None, [None, None]
        if self.guidance_projection is not None:
            projected_guidance = self.guidance_projection(appearance_guidance[0])

        if self.decoder_guidance_projection is not None:
            projected_decoder_guidance = [proj(g) for proj, g in zip(self.decoder_guidance_projection, appearance_guidance[1:])]

        if self.text_guidance_projection is not None:
            text_feats = text_feats.mean(dim=-2)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            projected_text_guidance = self.text_guidance_projection(text_feats)

        for layer in self.layers:
            corr_embed = layer(corr_embed, projected_guidance, projected_text_guidance)

        logit = self.conv_decoder(corr_embed, projected_decoder_guidance)

        return logit    
    
class FusionAggregatorVer14f(nn.Module):
    def __init__(self, 
        text_guidance_dim=512,
        text_guidance_proj_dim=128,
        appearance_guidance_dim=512,
        appearance_guidance_proj_dim=128,
        # decoder_dims = (64, 32),
        decoder_dims = (128,96,64,32),
        decoder_guidance_dims=(256, 128),
        decoder_guidance_proj_dims=(32, 16),
        num_layers=4,
        nheads=4, 
        hidden_dim=128,
        pooling_size=(6, 6),
        feature_resolution=(24, 24),
        window_size=12,
        attention_type='linear',
        prompt_channel=1,
        pad_len=256,
    ) -> None:
        """
        Cost Aggregation Model for CAT-Seg
        Args:
            text_guidance_dim: Dimension of text guidance
            text_guidance_proj_dim: Dimension of projected text guidance
            appearance_guidance_dim: Dimension of appearance guidance
            appearance_guidance_proj_dim: Dimension of projected appearance guidance
            decoder_dims: Upsampling decoder dimensions
            decoder_guidance_dims: Upsampling decoder guidance dimensions
            decoder_guidance_proj_dims: Upsampling decoder guidance projected dimensions
            num_layers: Number of layers for the aggregator
            nheads: Number of attention heads
            hidden_dim: Hidden dimension for transformer blocks
            pooling_size: Pooling size for the class aggregation layer
                          To reduce computation, we apply pooling in class aggregation blocks to reduce the number of tokens during training
            feature_resolution: Feature resolution for spatial aggregation
            window_size: Window size for Swin block in spatial aggregation
            attention_type: Attention type for the class aggregation. 
            prompt_channel: Number of prompts for ensembling text features. Default: 1
            pad_len: Padding length for the class aggregation. Default: 256
                     pad_len enforces the class aggregation block to have a fixed length of tokens for all inputs
                     This means it either pads the sequence with learnable tokens in class aggregation,
                     or truncates the classes with the initial CLIP cosine-similarity scores.
                     Set pad_len to 0 to disable this feature.
            """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.layers = nn.ModuleList([
            AggregatorLayer(
                hidden_dim=hidden_dim, text_guidance_dim=text_guidance_proj_dim, appearance_guidance=appearance_guidance_proj_dim, 
                nheads=nheads, input_resolution=feature_resolution, pooling_size=pooling_size, window_size=window_size, attention_type=attention_type, pad_len=pad_len,
            ) for _ in range(num_layers)
        ])

        # self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)

        self.guidance_projection = nn.Sequential(
            nn.Conv2d(appearance_guidance_dim, appearance_guidance_proj_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        ) if appearance_guidance_dim > 0 else None

        self.text_guidance_projection = nn.Sequential(
            nn.Linear(text_guidance_dim, text_guidance_proj_dim),
            nn.ReLU(),
        ) if text_guidance_dim > 0 else None

        self.decoder_guidance_projection = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            ) for d, dp in zip(decoder_guidance_dims, decoder_guidance_proj_dims)
        ]) if decoder_guidance_dims[0] > 0 else None
        # self.cat_corr_embed = nn.Conv2d(1024, hidden_dim, kernel_size=1, stride=1, padding=0)
        # Ver14e
        self.gw_corr_embed = nn.Conv2d(32, hidden_dim, kernel_size=7, stride=1, padding=3)
        # self.fusion_corr_embed = nn.Conv2d(2*hidden_dim,hidden_dim,kernel_size=7, stride=1, padding=3)
        # Ver14e
        self.decoder1 = UPmy(hidden_dim, decoder_dims[0], decoder_guidance_proj_dims[0])
        self.decoder2 = UPmy(decoder_dims[0], decoder_dims[1], decoder_guidance_proj_dims[1])
        self.decoder3 = UPmy(decoder_dims[1], decoder_dims[2], 0)
        self.decoder4 = UPmy(decoder_dims[2], decoder_dims[3], 0)
        # self.head = nn.Conv2d(decoder_dims[1], 1, kernel_size=3, stride=1, padding=1)
        self.head = nn.Conv2d(decoder_dims[3], 1, kernel_size=3, stride=1, padding=1)

        self.pad_len = pad_len

    def feature_map(self, img_feats, text_feats):
        # concatenated feature volume for feature aggregation baselines
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        img_feats = repeat(img_feats, "B C H W -> B C T H W", T=text_feats.shape[1])
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        text_feats = text_feats.mean(dim=-2) # average text features over different prompts
        text_feats = F.normalize(text_feats, dim=-1) # B T C
        text_feats = repeat(text_feats, "B T C -> B C T H W", H=img_feats.shape[-2], W=img_feats.shape[-1])
        return torch.cat((img_feats, text_feats), dim=1) # B 2C T H W

    def correlation(self, img_feats, text_feats):
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        corr = torch.einsum('bchw, btpc -> bpthw', img_feats, text_feats)
        return corr
    def group_wise_correlation(self, img_feats, text_feats, group_nums = 32):
        '''
        Return B P N T H W, wheareas N is the group number, P is prompt number, T is class number
        '''
        class_num = text_feats.shape[1]
        N_g = group_nums
        group_img_feats = rearrange(img_feats, 'B (C N) H W ->(B N) C H W', N=N_g)
        group_text_feats = rearrange(text_feats, 'B T P (C N) ->(B N) T P C', N=N_g)
        group_img_feats = F.normalize(group_img_feats, dim=1)
        group_text_feats = F.normalize(group_text_feats, dim = -1)
        group_corr = torch.einsum('bchw, btpc -> bpthw', group_img_feats, group_text_feats)
        group_corr = group_corr.squeeze(1)
        # group_corr = rearrange(group_corr,'(B N) T H W -> B N T H W', N = N_g)
        group_corr = rearrange(group_corr,'(B N) T H W -> (B T) N H W', N = N_g)
        
        #group_corr_embed = rearrange(group_corr,'(B T) C H W -> B T C H W', T = class_num)
        return group_corr
    def concatenation_correlation(self, img_feats, text_feats):
        class_num = text_feats.shape[1]
        H, W = img_feats.shape[-2], img_feats.shape[-1]
        img_feats=img_feats.unsqueeze(1).repeat([1,class_num,1,1,1])
        img_feats = rearrange(img_feats, 'B T C H W -> (B T) C H W')
        text_feats=text_feats.unsqueeze(-1).unsqueeze(-1).repeat([1,1,1,1,H,W])
        text_feats = rearrange(text_feats,'B T 1 C H W -> (B T) C H W')
        cat_feats = torch.cat([img_feats,text_feats],dim=1) # C=256
        cat_corr_embed = self.cat_corr_embed(cat_feats)
        #cat_corr_embed = rearrange(cat_corr_embed,'(B T) C H W -> B T C H W',T = class_num)
        return cat_corr_embed
    # def corr_embed(self, x):
    #     B = x.shape[0]
    #     corr_embed = rearrange(x, 'B P T H W -> (B T) P H W')
    #     corr_embed = self.conv1(corr_embed)
    #     corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
    #     return corr_embed
    
    def corr_projection(self, x, proj):
        corr_embed = rearrange(x, 'B C T H W -> B T H W C')
        corr_embed = proj(corr_embed)
        corr_embed = rearrange(corr_embed, 'B T H W C -> B C T H W')
        return corr_embed

    def upsample(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = F.interpolate(corr_embed, scale_factor=2, mode='bilinear', align_corners=True)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed

    def conv_decoder(self, x, guidance):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = self.decoder1(corr_embed, guidance[0])
        corr_embed = self.decoder2(corr_embed, guidance[1])
        corr_embed = self.decoder3(corr_embed, None)
        corr_embed = self.decoder4(corr_embed, None)
        corr_embed = self.head(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
        return corr_embed
    
    def forward(self, img_feats, text_feats, appearance_guidance):
        """
        Arguments:
            img_feats: (B, C, H, W)
            text_feats: (B, T, P, C) T是类别的个数
            apperance_guidance: tuple of (B, C, H, W)
        """

        classes = None

        # corr = self.correlation(img_feats, text_feats)
        T = text_feats.shape[1]
        gw_corr = self.group_wise_correlation(img_feats, text_feats)
        group_corr_embed = self.gw_corr_embed(gw_corr)
        # cat_corr = self.concatenation_correlation(img_feats, text_feats) # (B T) C H W
        #dual_corr = torch.cat([gw_corr,cat_corr],dim = 1 )
        #corr_embed = self.fusion_corr_embed(dual_corr)
        
        corr_embed = group_corr_embed
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W',T=T)
        # print('group only success')
        #corr = self.feature_map(img_feats, text_feats)
        # corr_embed = self.corr_embed(corr)

        projected_guidance, projected_text_guidance, projected_decoder_guidance = None, None, [None, None]
        if self.guidance_projection is not None:
            projected_guidance = self.guidance_projection(appearance_guidance[0])

        if self.decoder_guidance_projection is not None:
            projected_decoder_guidance = [proj(g) for proj, g in zip(self.decoder_guidance_projection, appearance_guidance[1:])]

        if self.text_guidance_projection is not None:
            text_feats = text_feats.mean(dim=-2)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            projected_text_guidance = self.text_guidance_projection(text_feats)

        for layer in self.layers:
            corr_embed = layer(corr_embed, projected_guidance, projected_text_guidance)

        logit = self.conv_decoder(corr_embed, projected_decoder_guidance)

        return logit
    
class FusionAggregatorVer14eb(nn.Module):
    def __init__(self, 
        text_guidance_dim=512,
        text_guidance_proj_dim=128,
        appearance_guidance_dim=512,
        appearance_guidance_proj_dim=128,
        # decoder_dims = (64, 32),
        decoder_dims = (128,96,64,32),
        decoder_guidance_dims=(256, 128),
        decoder_guidance_proj_dims=(32, 16),
        num_layers=4,
        nheads=4, 
        hidden_dim=128,
        pooling_size=(6, 6),
        feature_resolution=(24, 24),
        window_size=12,
        attention_type='linear',
        prompt_channel=1,
        pad_len=256,
    ) -> None:
        """
        Cost Aggregation Model for CAT-Seg
        Args:
            text_guidance_dim: Dimension of text guidance
            text_guidance_proj_dim: Dimension of projected text guidance
            appearance_guidance_dim: Dimension of appearance guidance
            appearance_guidance_proj_dim: Dimension of projected appearance guidance
            decoder_dims: Upsampling decoder dimensions
            decoder_guidance_dims: Upsampling decoder guidance dimensions
            decoder_guidance_proj_dims: Upsampling decoder guidance projected dimensions
            num_layers: Number of layers for the aggregator
            nheads: Number of attention heads
            hidden_dim: Hidden dimension for transformer blocks
            pooling_size: Pooling size for the class aggregation layer
                          To reduce computation, we apply pooling in class aggregation blocks to reduce the number of tokens during training
            feature_resolution: Feature resolution for spatial aggregation
            window_size: Window size for Swin block in spatial aggregation
            attention_type: Attention type for the class aggregation. 
            prompt_channel: Number of prompts for ensembling text features. Default: 1
            pad_len: Padding length for the class aggregation. Default: 256
                     pad_len enforces the class aggregation block to have a fixed length of tokens for all inputs
                     This means it either pads the sequence with learnable tokens in class aggregation,
                     or truncates the classes with the initial CLIP cosine-similarity scores.
                     Set pad_len to 0 to disable this feature.
            """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.layers = nn.ModuleList([
            AggregatorLayer(
                hidden_dim=hidden_dim, text_guidance_dim=text_guidance_proj_dim, appearance_guidance=appearance_guidance_proj_dim, 
                nheads=nheads, input_resolution=feature_resolution, pooling_size=pooling_size, window_size=window_size, attention_type=attention_type, pad_len=pad_len,
            ) for _ in range(num_layers)
        ])

        

        self.guidance_projection = nn.Sequential(
            nn.Conv2d(appearance_guidance_dim, appearance_guidance_proj_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        ) if appearance_guidance_dim > 0 else None

        self.text_guidance_projection = nn.Sequential(
            nn.Linear(text_guidance_dim, text_guidance_proj_dim),
            nn.ReLU(),
        ) if text_guidance_dim > 0 else None

        self.decoder_guidance_projection = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            ) for d, dp in zip(decoder_guidance_dims, decoder_guidance_proj_dims)
        ]) if decoder_guidance_dims[0] > 0 else None
        # self.cat_corr_embed = nn.Conv2d(1024, hidden_dim, kernel_size=7, stride=1, padding=3)
        # Ver14e
        self.vanilla_corr_embed = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.gw_corr_embed = nn.Conv2d(32, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.cat_corr_embed = nn.Conv2d(1024, hidden_dim, kernel_size=7, stride=1, padding=3)
        # self.fusion_corr_embed = nn.Conv2d(2*hidden_dim,hidden_dim,kernel_size=7, stride=1, padding=3)
        # Ver14e
        self.decoder1 = UPmy(hidden_dim, decoder_dims[0], decoder_guidance_proj_dims[0])
        self.decoder2 = UPmy(decoder_dims[0], decoder_dims[1], decoder_guidance_proj_dims[1])
        self.decoder3 = UPmy(decoder_dims[1], decoder_dims[2], 0)
        self.decoder4 = UPmy(decoder_dims[2], decoder_dims[3], 0)
        # self.head = nn.Conv2d(decoder_dims[1], 1, kernel_size=3, stride=1, padding=1)
        self.head = nn.Conv2d(decoder_dims[3], 1, kernel_size=3, stride=1, padding=1)
        self.sigma_vanilla = nn.Parameter(torch.ones(1))
        self.sigma_cat = nn.Parameter(torch.ones(1))
        self.sigma_gw = nn.Parameter(torch.ones(1))
        self.pad_len = pad_len

    def feature_map(self, img_feats, text_feats):
        # concatenated feature volume for feature aggregation baselines
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        img_feats = repeat(img_feats, "B C H W -> B C T H W", T=text_feats.shape[1])
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        text_feats = text_feats.mean(dim=-2) # average text features over different prompts
        text_feats = F.normalize(text_feats, dim=-1) # B T C
        text_feats = repeat(text_feats, "B T C -> B C T H W", H=img_feats.shape[-2], W=img_feats.shape[-1])
        return torch.cat((img_feats, text_feats), dim=1) # B 2C T H W

    def correlation(self, img_feats, text_feats):
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        corr = torch.einsum('bchw, btpc -> bpthw', img_feats, text_feats)
        return corr
    def group_wise_correlation(self, img_feats, text_feats, group_nums = 32):
        '''
        Return B P N T H W, wheareas N is the group number, P is prompt number, T is class number
        '''
        class_num = text_feats.shape[1]
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        text_feats = rearrange(text_feats,'B T 1 C -> B T C')
        img_text_mult = torch.einsum('b c h w, b t c -> b t c h w',img_feats,text_feats)
        img_text_grouped =rearrange(img_text_mult, ' B T (N C) H W -> B T N C H W',N=group_nums)
        grouped_corr = img_text_grouped.sum(dim=-3) # B T N H W
        grouped_corr = rearrange(grouped_corr,'B T N H W -> (B T) N H W')
        # group_corr_embed = self.gw_corr_embed(grouped_corr)
        #group_corr_embed = rearrange(group_corr,'(B T) C H W -> B T C H W', T = class_num)
        return grouped_corr
    def concatenation_correlation(self, img_feats, text_feats):
        class_num = text_feats.shape[1]
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        H, W = img_feats.shape[-2], img_feats.shape[-1]
        img_feats=img_feats.unsqueeze(1).repeat([1,class_num,1,1,1])
        img_feats = rearrange(img_feats, 'B T C H W -> (B T) C H W')
        text_feats=text_feats.unsqueeze(-1).unsqueeze(-1).repeat([1,1,1,1,H,W])
        text_feats = rearrange(text_feats,'B T 1 C H W -> (B T) C H W')
        cat_feats = torch.cat([img_feats,text_feats],dim=1) # C=256
        # cat_corr_embed = self.cat_corr_embed(cat_feats)
        #cat_corr_embed = rearrange(cat_corr_embed,'(B T) C H W -> B T C H W',T = class_num)
        return cat_feats
    def corr_embed(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B P T H W -> (B T) P H W')
        corr_embed = self.vanilla_corr_embed(corr_embed)
        # corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed
    
    def corr_projection(self, x, proj):
        corr_embed = rearrange(x, 'B C T H W -> B T H W C')
        corr_embed = proj(corr_embed)
        corr_embed = rearrange(corr_embed, 'B T H W C -> B C T H W')
        return corr_embed

    def upsample(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = F.interpolate(corr_embed, scale_factor=2, mode='bilinear', align_corners=True)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed

    def conv_decoder(self, x, guidance):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = self.decoder1(corr_embed, guidance[0])
        corr_embed = self.decoder2(corr_embed, guidance[1])
        corr_embed = self.decoder3(corr_embed, None)
        corr_embed = self.decoder4(corr_embed, None)
        corr_embed = self.head(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
        return corr_embed
    
    def forward(self, img_feats, text_feats, appearance_guidance):
        """
        Arguments:
            img_feats: (B, C, H, W)
            text_feats: (B, T, P, C) T是类别的个数
            apperance_guidance: tuple of (B, C, H, W)
        """

        # corr = self.correlation(img_feats, text_feats)
        T = text_feats.shape[1]
        gw_corr = self.group_wise_correlation(img_feats, text_feats) #(BT) C H W
        gw_corr_embed = self.gw_corr_embed(gw_corr)
        cat_corr = self.concatenation_correlation(img_feats,text_feats)
        cat_corr_embed = self.cat_corr_embed(cat_corr)
        vanilla_corr = self.correlation(img_feats, text_feats) #(BT) C H W
        vanilla_corr_embed = self.corr_embed(vanilla_corr)
        
        
        triple_way_corr_embed = self.sigma_gw*gw_corr_embed + self.sigma_cat*cat_corr_embed +self.sigma_vanilla*vanilla_corr_embed
        # print('14eb success')
        
        corr_embed = rearrange(triple_way_corr_embed, '(B T) C H W -> B C T H W',T=T)

        #corr = self.feature_map(img_feats, text_feats)
        # corr_embed = self.corr_embed(corr)

        projected_guidance, projected_text_guidance, projected_decoder_guidance = None, None, [None, None]
        if self.guidance_projection is not None:
            projected_guidance = self.guidance_projection(appearance_guidance[0])

        if self.decoder_guidance_projection is not None:
            projected_decoder_guidance = [proj(g) for proj, g in zip(self.decoder_guidance_projection, appearance_guidance[1:])]

        if self.text_guidance_projection is not None:
            text_feats = text_feats.mean(dim=-2)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            projected_text_guidance = self.text_guidance_projection(text_feats)

        for layer in self.layers:
            corr_embed = layer(corr_embed, projected_guidance, projected_text_guidance)

        logit = self.conv_decoder(corr_embed, projected_decoder_guidance)

        return logit    
class FusionAggregatorVer14ea(nn.Module):
    def __init__(self, 
        text_guidance_dim=512,
        text_guidance_proj_dim=128,
        appearance_guidance_dim=512,
        appearance_guidance_proj_dim=128,
        # decoder_dims = (64, 32),
        decoder_dims = (128,96,64,32),
        decoder_guidance_dims=(256, 128),
        decoder_guidance_proj_dims=(32, 16),
        num_layers=4,
        nheads=4, 
        hidden_dim=128,
        pooling_size=(6, 6),
        feature_resolution=(24, 24),
        window_size=12,
        attention_type='linear',
        prompt_channel=1,
        pad_len=256,
    ) -> None:
        """
        Cost Aggregation Model for CAT-Seg
        Args:
            text_guidance_dim: Dimension of text guidance
            text_guidance_proj_dim: Dimension of projected text guidance
            appearance_guidance_dim: Dimension of appearance guidance
            appearance_guidance_proj_dim: Dimension of projected appearance guidance
            decoder_dims: Upsampling decoder dimensions
            decoder_guidance_dims: Upsampling decoder guidance dimensions
            decoder_guidance_proj_dims: Upsampling decoder guidance projected dimensions
            num_layers: Number of layers for the aggregator
            nheads: Number of attention heads
            hidden_dim: Hidden dimension for transformer blocks
            pooling_size: Pooling size for the class aggregation layer
                          To reduce computation, we apply pooling in class aggregation blocks to reduce the number of tokens during training
            feature_resolution: Feature resolution for spatial aggregation
            window_size: Window size for Swin block in spatial aggregation
            attention_type: Attention type for the class aggregation. 
            prompt_channel: Number of prompts for ensembling text features. Default: 1
            pad_len: Padding length for the class aggregation. Default: 256
                     pad_len enforces the class aggregation block to have a fixed length of tokens for all inputs
                     This means it either pads the sequence with learnable tokens in class aggregation,
                     or truncates the classes with the initial CLIP cosine-similarity scores.
                     Set pad_len to 0 to disable this feature.
            """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.layers = nn.ModuleList([
            AggregatorLayer(
                hidden_dim=hidden_dim, text_guidance_dim=text_guidance_proj_dim, appearance_guidance=appearance_guidance_proj_dim, 
                nheads=nheads, input_resolution=feature_resolution, pooling_size=pooling_size, window_size=window_size, attention_type=attention_type, pad_len=pad_len,
            ) for _ in range(num_layers)
        ])

        

        self.guidance_projection = nn.Sequential(
            nn.Conv2d(appearance_guidance_dim, appearance_guidance_proj_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        ) if appearance_guidance_dim > 0 else None

        self.text_guidance_projection = nn.Sequential(
            nn.Linear(text_guidance_dim, text_guidance_proj_dim),
            nn.ReLU(),
        ) if text_guidance_dim > 0 else None

        self.decoder_guidance_projection = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            ) for d, dp in zip(decoder_guidance_dims, decoder_guidance_proj_dims)
        ]) if decoder_guidance_dims[0] > 0 else None
        # self.cat_corr_embed = nn.Conv2d(1024, hidden_dim, kernel_size=7, stride=1, padding=3)
        # Ver14e
        self.vanilla_corr_embed = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.gw_corr_embed = nn.Conv2d(32, hidden_dim, kernel_size=7, stride=1, padding=3)
        # self.fusion_corr_embed = nn.Conv2d(2*hidden_dim,hidden_dim,kernel_size=7, stride=1, padding=3)
        # Ver14e
        self.decoder1 = UPmy(hidden_dim, decoder_dims[0], decoder_guidance_proj_dims[0])
        self.decoder2 = UPmy(decoder_dims[0], decoder_dims[1], decoder_guidance_proj_dims[1])
        self.decoder3 = UPmy(decoder_dims[1], decoder_dims[2], 0)
        self.decoder4 = UPmy(decoder_dims[2], decoder_dims[3], 0)
        # self.head = nn.Conv2d(decoder_dims[1], 1, kernel_size=3, stride=1, padding=1)
        self.head = nn.Conv2d(decoder_dims[3], 1, kernel_size=3, stride=1, padding=1)

        self.pad_len = pad_len

    def feature_map(self, img_feats, text_feats):
        # concatenated feature volume for feature aggregation baselines
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        img_feats = repeat(img_feats, "B C H W -> B C T H W", T=text_feats.shape[1])
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        text_feats = text_feats.mean(dim=-2) # average text features over different prompts
        text_feats = F.normalize(text_feats, dim=-1) # B T C
        text_feats = repeat(text_feats, "B T C -> B C T H W", H=img_feats.shape[-2], W=img_feats.shape[-1])
        return torch.cat((img_feats, text_feats), dim=1) # B 2C T H W

    def correlation(self, img_feats, text_feats):
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        corr = torch.einsum('bchw, btpc -> bpthw', img_feats, text_feats)
        return corr
    def group_wise_correlation(self, img_feats, text_feats, group_nums = 32):
        '''
        Return B P N T H W, wheareas N is the group number, P is prompt number, T is class number
        '''
        class_num = text_feats.shape[1]
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        text_feats = rearrange(text_feats,'B T 1 C -> B T C')
        img_text_mult = torch.einsum('b c h w, b t c -> b t c h w',img_feats,text_feats)
        img_text_grouped =rearrange(img_text_mult, ' B T (N C) H W -> B T N C H W',N=group_nums)
        grouped_corr = img_text_grouped.sum(dim=-3) # B T N H W
        grouped_corr = rearrange(grouped_corr,'B T N H W -> (B T) N H W')
        # group_corr_embed = self.gw_corr_embed(grouped_corr)
        #group_corr_embed = rearrange(group_corr,'(B T) C H W -> B T C H W', T = class_num)
        return grouped_corr
    def concatenation_correlation(self, img_feats, text_feats):
        class_num = text_feats.shape[1]
        H, W = img_feats.shape[-2], img_feats.shape[-1]
        img_feats=img_feats.unsqueeze(1).repeat([1,class_num,1,1,1])
        img_feats = rearrange(img_feats, 'B T C H W -> (B T) C H W')
        text_feats=text_feats.unsqueeze(-1).unsqueeze(-1).repeat([1,1,1,1,H,W])
        text_feats = rearrange(text_feats,'B T 1 C H W -> (B T) C H W')
        cat_feats = torch.cat([img_feats,text_feats],dim=1) # C=256
        cat_corr_embed = self.cat_corr_embed(cat_feats)
        #cat_corr_embed = rearrange(cat_corr_embed,'(B T) C H W -> B T C H W',T = class_num)
        return cat_corr_embed
    def corr_embed(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B P T H W -> (B T) P H W')
        corr_embed = self.vanilla_corr_embed(corr_embed)
        # corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed
    
    def corr_projection(self, x, proj):
        corr_embed = rearrange(x, 'B C T H W -> B T H W C')
        corr_embed = proj(corr_embed)
        corr_embed = rearrange(corr_embed, 'B T H W C -> B C T H W')
        return corr_embed

    def upsample(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = F.interpolate(corr_embed, scale_factor=2, mode='bilinear', align_corners=True)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed

    def conv_decoder(self, x, guidance):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = self.decoder1(corr_embed, guidance[0])
        corr_embed = self.decoder2(corr_embed, guidance[1])
        corr_embed = self.decoder3(corr_embed, None)
        corr_embed = self.decoder4(corr_embed, None)
        corr_embed = self.head(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
        return corr_embed
    
    def forward(self, img_feats, text_feats, appearance_guidance):
        """
        Arguments:
            img_feats: (B, C, H, W)
            text_feats: (B, T, P, C) T是类别的个数
            apperance_guidance: tuple of (B, C, H, W)
        """

        classes = None

        # corr = self.correlation(img_feats, text_feats)
        T = text_feats.shape[1]
        gw_corr = self.group_wise_correlation(img_feats, text_feats) #(BT) C H W
        # print('success here!!!')
        gw_corr_embed = self.gw_corr_embed(gw_corr)
        vanilla_corr = self.correlation(img_feats, text_feats) #(BT) C H W
        vanilla_corr_embed = self.corr_embed(vanilla_corr)
        dual_corr_embed = gw_corr_embed + vanilla_corr_embed
        corr_embed = rearrange(dual_corr_embed, '(B T) C H W -> B C T H W',T=T)

        #corr = self.feature_map(img_feats, text_feats)
        # corr_embed = self.corr_embed(corr)

        projected_guidance, projected_text_guidance, projected_decoder_guidance = None, None, [None, None]
        if self.guidance_projection is not None:
            projected_guidance = self.guidance_projection(appearance_guidance[0])

        if self.decoder_guidance_projection is not None:
            projected_decoder_guidance = [proj(g) for proj, g in zip(self.decoder_guidance_projection, appearance_guidance[1:])]

        if self.text_guidance_projection is not None:
            text_feats = text_feats.mean(dim=-2)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            projected_text_guidance = self.text_guidance_projection(text_feats)

        for layer in self.layers:
            corr_embed = layer(corr_embed, projected_guidance, projected_text_guidance)

        logit = self.conv_decoder(corr_embed, projected_decoder_guidance)

        return logit

class FusionAggregatorVer14e(nn.Module):
    def __init__(self, 
        text_guidance_dim=512,
        text_guidance_proj_dim=128,
        appearance_guidance_dim=512,
        appearance_guidance_proj_dim=128,
        # decoder_dims = (64, 32),
        decoder_dims = (128,96,64,32),
        decoder_guidance_dims=(256, 128),
        decoder_guidance_proj_dims=(32, 16),
        num_layers=4,
        nheads=4, 
        hidden_dim=128,
        pooling_size=(6, 6),
        feature_resolution=(24, 24),
        window_size=12,
        attention_type='linear',
        prompt_channel=1,
        pad_len=256,
    ) -> None:
        """
        Cost Aggregation Model for CAT-Seg
        Args:
            text_guidance_dim: Dimension of text guidance
            text_guidance_proj_dim: Dimension of projected text guidance
            appearance_guidance_dim: Dimension of appearance guidance
            appearance_guidance_proj_dim: Dimension of projected appearance guidance
            decoder_dims: Upsampling decoder dimensions
            decoder_guidance_dims: Upsampling decoder guidance dimensions
            decoder_guidance_proj_dims: Upsampling decoder guidance projected dimensions
            num_layers: Number of layers for the aggregator
            nheads: Number of attention heads
            hidden_dim: Hidden dimension for transformer blocks
            pooling_size: Pooling size for the class aggregation layer
                          To reduce computation, we apply pooling in class aggregation blocks to reduce the number of tokens during training
            feature_resolution: Feature resolution for spatial aggregation
            window_size: Window size for Swin block in spatial aggregation
            attention_type: Attention type for the class aggregation. 
            prompt_channel: Number of prompts for ensembling text features. Default: 1
            pad_len: Padding length for the class aggregation. Default: 256
                     pad_len enforces the class aggregation block to have a fixed length of tokens for all inputs
                     This means it either pads the sequence with learnable tokens in class aggregation,
                     or truncates the classes with the initial CLIP cosine-similarity scores.
                     Set pad_len to 0 to disable this feature.
            """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.layers = nn.ModuleList([
            AggregatorLayer(
                hidden_dim=hidden_dim, text_guidance_dim=text_guidance_proj_dim, appearance_guidance=appearance_guidance_proj_dim, 
                nheads=nheads, input_resolution=feature_resolution, pooling_size=pooling_size, window_size=window_size, attention_type=attention_type, pad_len=pad_len,
            ) for _ in range(num_layers)
        ])

        # self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)

        self.guidance_projection = nn.Sequential(
            nn.Conv2d(appearance_guidance_dim, appearance_guidance_proj_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        ) if appearance_guidance_dim > 0 else None

        self.text_guidance_projection = nn.Sequential(
            nn.Linear(text_guidance_dim, text_guidance_proj_dim),
            nn.ReLU(),
        ) if text_guidance_dim > 0 else None

        self.decoder_guidance_projection = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            ) for d, dp in zip(decoder_guidance_dims, decoder_guidance_proj_dims)
        ]) if decoder_guidance_dims[0] > 0 else None
        self.cat_corr_embed = nn.Conv2d(1024, hidden_dim, kernel_size=1, stride=1, padding=0)
        # Ver14e
        self.gw_corr_embed = nn.Conv2d(32, hidden_dim, kernel_size=1, stride=1, padding=0)
        self.fusion_corr_embed = nn.Conv2d(2*hidden_dim,hidden_dim,kernel_size=7, stride=1, padding=3)
        # Ver14e
        self.decoder1 = UPmy(hidden_dim, decoder_dims[0], decoder_guidance_proj_dims[0])
        self.decoder2 = UPmy(decoder_dims[0], decoder_dims[1], decoder_guidance_proj_dims[1])
        self.decoder3 = UPmy(decoder_dims[1], decoder_dims[2], 0)
        self.decoder4 = UPmy(decoder_dims[2], decoder_dims[3], 0)
        # self.head = nn.Conv2d(decoder_dims[1], 1, kernel_size=3, stride=1, padding=1)
        self.head = nn.Conv2d(decoder_dims[3], 1, kernel_size=3, stride=1, padding=1)

        self.pad_len = pad_len

    def feature_map(self, img_feats, text_feats):
        # concatenated feature volume for feature aggregation baselines
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        img_feats = repeat(img_feats, "B C H W -> B C T H W", T=text_feats.shape[1])
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        text_feats = text_feats.mean(dim=-2) # average text features over different prompts
        text_feats = F.normalize(text_feats, dim=-1) # B T C
        text_feats = repeat(text_feats, "B T C -> B C T H W", H=img_feats.shape[-2], W=img_feats.shape[-1])
        return torch.cat((img_feats, text_feats), dim=1) # B 2C T H W

    def correlation(self, img_feats, text_feats):
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        corr = torch.einsum('bchw, btpc -> bpthw', img_feats, text_feats)
        return corr
    def group_wise_correlation(self, img_feats, text_feats, group_nums = 32):
        '''
        Return B P N T H W, wheareas N is the group number, P is prompt number, T is class number
        '''
        class_num = text_feats.shape[1]
        N_g = group_nums
        group_img_feats = rearrange(img_feats, 'B (C N) H W ->(B N) C H W', N=N_g)
        group_text_feats = rearrange(text_feats, 'B T P (C N) ->(B N) T P C', N=N_g)
        group_img_feats = F.normalize(group_img_feats, dim=1)
        group_text_feats = F.normalize(group_text_feats, dim = -1)
        group_corr = torch.einsum('bchw, btpc -> bpthw', group_img_feats, group_text_feats)
        group_corr = group_corr.squeeze(1)
        # group_corr = rearrange(group_corr,'(B N) T H W -> B N T H W', N = N_g)
        group_corr = rearrange(group_corr,'(B N) T H W -> (B T) N H W', N = N_g)
        group_corr_embed = self.gw_corr_embed(group_corr)
        #group_corr_embed = rearrange(group_corr,'(B T) C H W -> B T C H W', T = class_num)
        return group_corr_embed
    def concatenation_correlation(self, img_feats, text_feats):
        class_num = text_feats.shape[1]
        H, W = img_feats.shape[-2], img_feats.shape[-1]
        img_feats=img_feats.unsqueeze(1).repeat([1,class_num,1,1,1])
        img_feats = rearrange(img_feats, 'B T C H W -> (B T) C H W')
        text_feats=text_feats.unsqueeze(-1).unsqueeze(-1).repeat([1,1,1,1,H,W])
        text_feats = rearrange(text_feats,'B T 1 C H W -> (B T) C H W')
        cat_feats = torch.cat([img_feats,text_feats],dim=1) # C=256
        cat_corr_embed = self.cat_corr_embed(cat_feats)
        #cat_corr_embed = rearrange(cat_corr_embed,'(B T) C H W -> B T C H W',T = class_num)
        return cat_corr_embed
    # def corr_embed(self, x):
    #     B = x.shape[0]
    #     corr_embed = rearrange(x, 'B P T H W -> (B T) P H W')
    #     corr_embed = self.conv1(corr_embed)
    #     corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
    #     return corr_embed
    
    def corr_projection(self, x, proj):
        corr_embed = rearrange(x, 'B C T H W -> B T H W C')
        corr_embed = proj(corr_embed)
        corr_embed = rearrange(corr_embed, 'B T H W C -> B C T H W')
        return corr_embed

    def upsample(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = F.interpolate(corr_embed, scale_factor=2, mode='bilinear', align_corners=True)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed

    def conv_decoder(self, x, guidance):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = self.decoder1(corr_embed, guidance[0])
        corr_embed = self.decoder2(corr_embed, guidance[1])
        corr_embed = self.decoder3(corr_embed, None)
        corr_embed = self.decoder4(corr_embed, None)
        corr_embed = self.head(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
        return corr_embed
    
    def forward(self, img_feats, text_feats, appearance_guidance):
        """
        Arguments:
            img_feats: (B, C, H, W)
            text_feats: (B, T, P, C) T是类别的个数
            apperance_guidance: tuple of (B, C, H, W)
        """

        classes = None

        # corr = self.correlation(img_feats, text_feats)
        T = text_feats.shape[1]
        gw_corr = self.group_wise_correlation(img_feats, text_feats)
        cat_corr = self.concatenation_correlation(img_feats, text_feats) # (B T) C H W
        dual_corr = torch.cat([gw_corr,cat_corr],dim = 1 )
        corr_embed = self.fusion_corr_embed(dual_corr)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W',T=T)

        #corr = self.feature_map(img_feats, text_feats)
        # corr_embed = self.corr_embed(corr)

        # projected_guidance, projected_text_guidance, projected_decoder_guidance = None, None, [None, None]
        if self.guidance_projection is not None:
            projected_guidance = self.guidance_projection(appearance_guidance[0])

        if self.decoder_guidance_projection is not None:
            projected_decoder_guidance = [proj(g) for proj, g in zip(self.decoder_guidance_projection, appearance_guidance[1:])]

        if self.text_guidance_projection is not None:
            text_feats = text_feats.mean(dim=-2)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            projected_text_guidance = self.text_guidance_projection(text_feats)

        for layer in self.layers:
            corr_embed = layer(corr_embed, projected_guidance, projected_text_guidance)

        logit = self.conv_decoder(corr_embed, projected_decoder_guidance)

        return logit
class ClipDenseProjLayer(nn.Module):
    def __init__(self, hidden_dim=768, last_dim=512 ,appearance_guidance_proj_dim=128):
        super().__init__()
        self.used_clip_layer = 12
        self.sigma =  nn.Parameter(torch.zeros(12))
        self.appearance_guidance_proj_dim = appearance_guidance_proj_dim
        self.clip_dense_proj = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(normalized_shape=hidden_dim),
                nn.Linear(in_features=hidden_dim, out_features=appearance_guidance_proj_dim),
                nn.ReLU()
                ) for _ in range(self.used_clip_layer - 1)
        ])
        self.clip_last_proj = nn.Sequential(
                nn.LayerNorm(normalized_shape=last_dim),
                nn.Linear(in_features=last_dim, out_features=appearance_guidance_proj_dim),
                nn.ReLU()
                )
    def forward(self, clip_feat_list):
        # B L C 
        # C=appearance_guidance_proj_dim
        # print(self.sigma)
        # exit()
        B = clip_feat_list[0].shape[0]
        L = clip_feat_list[0].shape[1]
        C = self.appearance_guidance_proj_dim
        clip_aggregated_feat = []
        for l_ind ,clip_feat in enumerate(clip_feat_list):
            if l_ind == len(clip_feat_list)-1:
                proj_feat = self.clip_last_proj(clip_feat)
                clip_aggregated_feat.append(self.sigma[l_ind]*proj_feat)
            else:
                proj_feat = self.clip_dense_proj[l_ind](clip_feat)
                clip_aggregated_feat.append(self.sigma[l_ind]*proj_feat)
        return clip_aggregated_feat
                
                
            

class FusionAggregatorVer14db(nn.Module):
    def __init__(self, 
        text_guidance_dim=512,
        text_guidance_proj_dim=128,
        appearance_guidance_dim=512,
        appearance_guidance_proj_dim=128,
        
        # decoder_dims = (64, 32),
        decoder_dims = (128,96,64,32),
        decoder_guidance_dims=(256, 128),
        decoder_guidance_proj_dims=(32, 16),
        num_layers=4,
        nheads=4, 
        hidden_dim=128,
        pooling_size=(6, 6),
        feature_resolution=(24, 24),
        window_size=12,
        attention_type='linear',
        prompt_channel=1,
        pad_len=256,
    ) -> None:
        """
        Cost Aggregation Model for CAT-Seg
        Args:
            text_guidance_dim: Dimension of text guidance
            text_guidance_proj_dim: Dimension of projected text guidance
            appearance_guidance_dim: Dimension of appearance guidance
            appearance_guidance_proj_dim: Dimension of projected appearance guidance
            decoder_dims: Upsampling decoder dimensions
            decoder_guidance_dims: Upsampling decoder guidance dimensions
            decoder_guidance_proj_dims: Upsampling decoder guidance projected dimensions
            num_layers: Number of layers for the aggregator
            nheads: Number of attention heads
            hidden_dim: Hidden dimension for transformer blocks
            pooling_size: Pooling size for the class aggregation layer
                          To reduce computation, we apply pooling in class aggregation blocks to reduce the number of tokens during training
            feature_resolution: Feature resolution for spatial aggregation
            window_size: Window size for Swin block in spatial aggregation
            attention_type: Attention type for the class aggregation. 
            prompt_channel: Number of prompts for ensembling text features. Default: 1
            pad_len: Padding length for the class aggregation. Default: 256
                     pad_len enforces the class aggregation block to have a fixed length of tokens for all inputs
                     This means it either pads the sequence with learnable tokens in class aggregation,
                     or truncates the classes with the initial CLIP cosine-similarity scores.
                     Set pad_len to 0 to disable this feature.
            """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        used_clip_layer = 12
        # we dropped the appearance_guidance_dim
        # we make it a must
        self.layers = nn.ModuleList([
            AggregatorLayer(
                hidden_dim=hidden_dim, text_guidance_dim=text_guidance_proj_dim, appearance_guidance=appearance_guidance_proj_dim, 
                nheads=nheads, input_resolution=feature_resolution, pooling_size=pooling_size, window_size=window_size, attention_type=attention_type, pad_len=pad_len,
            ) for _ in range(num_layers)
        ])
        self.clip_dense_proj_layers = ClipDenseProjLayer(
            hidden_dim=768, last_dim=512, appearance_guidance_proj_dim = appearance_guidance_proj_dim
            )if appearance_guidance_dim > 0 else None
        
        self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)

        # self.guidance_projection = nn.Sequential(
        #     nn.Conv2d(appearance_guidance_dim, appearance_guidance_proj_dim, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        # ) if appearance_guidance_dim > 0 else None


        self.text_guidance_projection = nn.Sequential(
            nn.LayerNorm(text_guidance_dim),
            nn.Linear(text_guidance_dim, text_guidance_proj_dim),
            nn.ReLU(),
            nn.LayerNorm(text_guidance_proj_dim),
            nn.Linear(text_guidance_proj_dim, text_guidance_proj_dim//2),
            nn.ReLU(),
            nn.LayerNorm(text_guidance_proj_dim//2),
            nn.Linear(text_guidance_proj_dim//2, text_guidance_proj_dim),
            nn.ReLU(),
        ) 
        self.decoder_guidance_projection = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            ) for d, dp in zip(decoder_guidance_dims, decoder_guidance_proj_dims)
        ]) if decoder_guidance_dims[0] > 0 else None

        self.decoder1 = UPmy(hidden_dim, decoder_dims[0], decoder_guidance_proj_dims[0])
        self.decoder2 = UPmy(decoder_dims[0], decoder_dims[1], decoder_guidance_proj_dims[1])
        self.decoder3 = UPmy(decoder_dims[1], decoder_dims[2], 0)
        self.decoder4 = UPmy(decoder_dims[2], decoder_dims[3], 0)
        # self.head = nn.Conv2d(decoder_dims[1], 1, kernel_size=3, stride=1, padding=1)
        self.head = nn.Conv2d(decoder_dims[3], 1, kernel_size=3, stride=1, padding=1)

        self.pad_len = pad_len
    
    def feature_map(self, img_feats, text_feats):
        # concatenated feature volume for feature aggregation baselines
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        img_feats = repeat(img_feats, "B C H W -> B C T H W", T=text_feats.shape[1])
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        text_feats = text_feats.mean(dim=-2) # average text features over different prompts
        text_feats = F.normalize(text_feats, dim=-1) # B T C
        text_feats = repeat(text_feats, "B T C -> B C T H W", H=img_feats.shape[-2], W=img_feats.shape[-1])
        return torch.cat((img_feats, text_feats), dim=1) # B 2C T H W

    def correlation(self, img_feats, text_feats):
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        corr = torch.einsum('bchw, btpc -> bpthw', img_feats, text_feats)
        return corr

    def corr_embed(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B P T H W -> (B T) P H W')
        corr_embed = self.conv1(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed
    
    def corr_projection(self, x, proj):
        corr_embed = rearrange(x, 'B C T H W -> B T H W C')
        corr_embed = proj(corr_embed)
        corr_embed = rearrange(corr_embed, 'B T H W C -> B C T H W')
        return corr_embed

    def upsample(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = F.interpolate(corr_embed, scale_factor=2, mode='bilinear', align_corners=True)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed

    def conv_decoder(self, x, guidance):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = self.decoder1(corr_embed, guidance[0])
        corr_embed = self.decoder2(corr_embed, guidance[1])
        corr_embed = self.head(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
        return corr_embed
    
    def forward(self, img_feats, text_feats, appearance_guidance):
        """
        Arguments:
            img_feats: (B, C, H, W)
            text_feats: (B, T, P, C) T是类别的个数
            apperance_guidance: tuple of (B, C, H, W)
        """


        corr = self.correlation(img_feats, text_feats)

        #corr = self.feature_map(img_feats, text_feats)
        corr_embed = self.corr_embed(corr)

        projected_guidance, projected_text_guidance, projected_decoder_guidance = None, None, [None, None]
        if self.clip_dense_proj_layers is not None:
            projected_guidance = self.clip_dense_proj_layers(appearance_guidance)
            projected_guidance = sum(projected_guidance)
            projected_guidance = rearrange(projected_guidance,'B (H W) C -> B C H W',H=img_feats.shape[-2])


            

        if self.decoder_guidance_projection is not None:
            projected_decoder_guidance = [proj(g) for proj, g in zip(self.decoder_guidance_projection, appearance_guidance[1:])]

        if self.text_guidance_projection is not None:
            # print('textGuid')
            text_feats = text_feats.mean(dim=-2)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            projected_text_guidance = self.text_guidance_projection(text_feats)

        for layer in self.layers:
            # print('enc')
            corr_embed = layer(corr_embed, projected_guidance, projected_text_guidance)

        logit = self.conv_decoder(corr_embed, projected_decoder_guidance)

 
        return logit
class FusionAggregatorVer14dc(nn.Module):
    def __init__(self, 
        text_guidance_dim=512,
        text_guidance_proj_dim=128,
        appearance_guidance_dim=512,
        appearance_guidance_proj_dim=128,
        # decoder_dims = (64, 32),
        decoder_dims = (128,96,64,32),
        decoder_guidance_dims=(256, 128),
        decoder_guidance_proj_dims=(32, 16),
        num_layers=4,
        nheads=4, 
        hidden_dim=128,
        pooling_size=(6, 6),
        feature_resolution=(24, 24),
        window_size=12,
        attention_type='linear',
        prompt_channel=1,
        pad_len=256,
    ) -> None:
        """
        Cost Aggregation Model for CAT-Seg
        Args:
            text_guidance_dim: Dimension of text guidance
            text_guidance_proj_dim: Dimension of projected text guidance
            appearance_guidance_dim: Dimension of appearance guidance
            appearance_guidance_proj_dim: Dimension of projected appearance guidance
            decoder_dims: Upsampling decoder dimensions
            decoder_guidance_dims: Upsampling decoder guidance dimensions
            decoder_guidance_proj_dims: Upsampling decoder guidance projected dimensions
            num_layers: Number of layers for the aggregator
            nheads: Number of attention heads
            hidden_dim: Hidden dimension for transformer blocks
            pooling_size: Pooling size for the class aggregation layer
                          To reduce computation, we apply pooling in class aggregation blocks to reduce the number of tokens during training
            feature_resolution: Feature resolution for spatial aggregation
            window_size: Window size for Swin block in spatial aggregation
            attention_type: Attention type for the class aggregation. 
            prompt_channel: Number of prompts for ensembling text features. Default: 1
            pad_len: Padding length for the class aggregation. Default: 256
                     pad_len enforces the class aggregation block to have a fixed length of tokens for all inputs
                     This means it either pads the sequence with learnable tokens in class aggregation,
                     or truncates the classes with the initial CLIP cosine-similarity scores.
                     Set pad_len to 0 to disable this feature.
            """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        used_clip_layer = 12
        # we dropped the appearance_guidance_dim
        # we make it a must
        self.layers = nn.ModuleList([
            AggregatorLayer(
                hidden_dim=hidden_dim, text_guidance_dim=text_guidance_proj_dim, appearance_guidance=appearance_guidance_proj_dim, 
                nheads=nheads, input_resolution=feature_resolution, pooling_size=pooling_size, window_size=window_size, attention_type=attention_type, pad_len=pad_len,
            ) for _ in range(num_layers)
        ])
        self.clip_dense_proj_layers = ClipDenseProjLayer(
            hidden_dim=768, last_dim=512, appearance_guidance_proj_dim = appearance_guidance_proj_dim
            )if appearance_guidance_dim > 0 else None
        
        self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)

        # self.guidance_projection = nn.Sequential(
        #     nn.Conv2d(appearance_guidance_dim, appearance_guidance_proj_dim, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        # ) if appearance_guidance_dim > 0 else None


        self.text_guidance_projection = nn.Sequential(
            nn.LayerNorm(text_guidance_dim),
            nn.Linear(text_guidance_dim, text_guidance_proj_dim),
            nn.ReLU(),
            nn.LayerNorm(text_guidance_proj_dim),
            nn.Linear(text_guidance_proj_dim, text_guidance_proj_dim//2),
            nn.ReLU(),
            nn.LayerNorm(text_guidance_proj_dim//2),
            nn.Linear(text_guidance_proj_dim//2, text_guidance_proj_dim),
            nn.ReLU(),
        ) 
        self.decoder_guidance_projection = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            ) for d, dp in zip(decoder_guidance_dims, decoder_guidance_proj_dims)
        ]) if decoder_guidance_dims[0] > 0 else None

        self.decoder1 = Up(hidden_dim, decoder_dims[0], decoder_guidance_proj_dims[0])
        self.decoder2 = Up(decoder_dims[0], decoder_dims[1], decoder_guidance_proj_dims[1])
        # self.decoder3 = UPmy(decoder_dims[1], decoder_dims[2], 0)
        # self.decoder4 = UPmy(decoder_dims[2], decoder_dims[3], 0)
        self.head = nn.Conv2d(decoder_dims[1], 1, kernel_size=3, stride=1, padding=1)
        # self.head = nn.Conv2d(decoder_dims[3], 1, kernel_size=3, stride=1, padding=1)
        self.upsample1 = nn.ConvTranspose2d(768, 256, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(768, 128, kernel_size=4, stride=4)
        self.pad_len = pad_len
    
    def feature_map(self, img_feats, text_feats):
        # concatenated feature volume for feature aggregation baselines
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        img_feats = repeat(img_feats, "B C H W -> B C T H W", T=text_feats.shape[1])
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        text_feats = text_feats.mean(dim=-2) # average text features over different prompts
        text_feats = F.normalize(text_feats, dim=-1) # B T C
        text_feats = repeat(text_feats, "B T C -> B C T H W", H=img_feats.shape[-2], W=img_feats.shape[-1])
        return torch.cat((img_feats, text_feats), dim=1) # B 2C T H W

    def correlation(self, img_feats, text_feats):
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        corr = torch.einsum('bchw, btpc -> bpthw', img_feats, text_feats)
        return corr

    def corr_embed(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B P T H W -> (B T) P H W')
        corr_embed = self.conv1(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed
    
    def corr_projection(self, x, proj):
        corr_embed = rearrange(x, 'B C T H W -> B T H W C')
        corr_embed = proj(corr_embed)
        corr_embed = rearrange(corr_embed, 'B T H W C -> B C T H W')
        return corr_embed

    def upsample(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = F.interpolate(corr_embed, scale_factor=2, mode='bilinear', align_corners=True)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed

    def conv_decoder(self, x, guidance):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = self.decoder1(corr_embed, guidance[0])
        corr_embed = self.decoder2(corr_embed, guidance[1])
        corr_embed = self.head(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
        return corr_embed
    
    def forward(self, img_feats, text_feats, appearance_guidance):
        """
        Arguments:
            img_feats: (B, C, H, W)
            text_feats: (B, T, P, C) T是类别的个数
            apperance_guidance: tuple of (B, C, H, W)
        """


        corr = self.correlation(img_feats, text_feats)

        #corr = self.feature_map(img_feats, text_feats)
        corr_embed = self.corr_embed(corr)

        projected_guidance, projected_text_guidance, projected_decoder_guidance = None, None, [None, None]
        if self.clip_dense_proj_layers is not None:
            #print('encod_guid')
            projected_guidance = self.clip_dense_proj_layers(appearance_guidance)
            projected_guidance = sum(projected_guidance)
            projected_guidance = rearrange(projected_guidance,'B (H W) C -> B C H W',H=img_feats.shape[-2])

        clip_L4 = rearrange(appearance_guidance[3], "B (H W) C -> B C H W", H=24)
        clip_L8 = rearrange(appearance_guidance[7], "B (H W) C -> B C H W", H=24)
        res4 = self.upsample1(clip_L4)
        res5 = self.upsample2(clip_L8)
        
        if self.decoder_guidance_projection is not None:
            #print('decod_guid')
            projected_decoder_guidance = [proj(g) for proj, g in zip(self.decoder_guidance_projection, [res4,res5])]

        if self.text_guidance_projection is not None:
            #print('textGuid')
            text_feats = text_feats.mean(dim=-2)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            projected_text_guidance = self.text_guidance_projection(text_feats)

        for layer in self.layers:
            # print('enc')
            corr_embed = layer(corr_embed, projected_guidance, projected_text_guidance)

        logit = self.conv_decoder(corr_embed, projected_decoder_guidance)
        # print('success here')
 
        return logit
class FusionAggregatorVer14da(nn.Module):
    def __init__(self, 
        text_guidance_dim=512,
        text_guidance_proj_dim=128,
        appearance_guidance_dim=512,
        appearance_guidance_proj_dim=128,
        # decoder_dims = (64, 32),
        decoder_dims = (128,96,64,32),
        decoder_guidance_dims=(256, 128),
        decoder_guidance_proj_dims=(32, 16),
        num_layers=4,
        nheads=4, 
        hidden_dim=128,
        pooling_size=(6, 6),
        feature_resolution=(24, 24),
        window_size=12,
        attention_type='linear',
        prompt_channel=1,
        pad_len=256,
    ) -> None:
        """
        Cost Aggregation Model for CAT-Seg
        Args:
            text_guidance_dim: Dimension of text guidance
            text_guidance_proj_dim: Dimension of projected text guidance
            appearance_guidance_dim: Dimension of appearance guidance
            appearance_guidance_proj_dim: Dimension of projected appearance guidance
            decoder_dims: Upsampling decoder dimensions
            decoder_guidance_dims: Upsampling decoder guidance dimensions
            decoder_guidance_proj_dims: Upsampling decoder guidance projected dimensions
            num_layers: Number of layers for the aggregator
            nheads: Number of attention heads
            hidden_dim: Hidden dimension for transformer blocks
            pooling_size: Pooling size for the class aggregation layer
                          To reduce computation, we apply pooling in class aggregation blocks to reduce the number of tokens during training
            feature_resolution: Feature resolution for spatial aggregation
            window_size: Window size for Swin block in spatial aggregation
            attention_type: Attention type for the class aggregation. 
            prompt_channel: Number of prompts for ensembling text features. Default: 1
            pad_len: Padding length for the class aggregation. Default: 256
                     pad_len enforces the class aggregation block to have a fixed length of tokens for all inputs
                     This means it either pads the sequence with learnable tokens in class aggregation,
                     or truncates the classes with the initial CLIP cosine-similarity scores.
                     Set pad_len to 0 to disable this feature.
            """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        used_clip_layer = 12
        # we dropped the appearance_guidance_dim
        # we make it a must
        self.layers = nn.ModuleList([
            AggregatorLayer(
                hidden_dim=hidden_dim, text_guidance_dim=text_guidance_proj_dim, appearance_guidance=appearance_guidance_proj_dim, 
                nheads=nheads, input_resolution=feature_resolution, pooling_size=pooling_size, window_size=window_size, attention_type=attention_type, pad_len=pad_len,
            ) for _ in range(num_layers)
        ])
        self.clip_dense_proj_layers = ClipDenseProjLayer(
            hidden_dim=768, last_dim=512, appearance_guidance_proj_dim = appearance_guidance_proj_dim
            )if appearance_guidance_dim > 0 else None
        
        self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)

        # self.guidance_projection = nn.Sequential(
        #     nn.Conv2d(appearance_guidance_dim, appearance_guidance_proj_dim, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        # ) if appearance_guidance_dim > 0 else None


        self.text_guidance_projection = nn.Sequential(
            nn.LayerNorm(text_guidance_dim),
            nn.Linear(text_guidance_dim, text_guidance_proj_dim),
            nn.ReLU(),
            nn.LayerNorm(text_guidance_proj_dim),
            nn.Linear(text_guidance_proj_dim, text_guidance_proj_dim//2),
            nn.ReLU(),
            nn.LayerNorm(text_guidance_proj_dim//2),
            nn.Linear(text_guidance_proj_dim//2, text_guidance_proj_dim),
            nn.ReLU(),
        ) 
        self.decoder_guidance_projection = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            ) for d, dp in zip(decoder_guidance_dims, decoder_guidance_proj_dims)
        ]) if decoder_guidance_dims[0] > 0 else None

        self.decoder1 = UPmy(hidden_dim, decoder_dims[0], decoder_guidance_proj_dims[0])
        self.decoder2 = UPmy(decoder_dims[0], decoder_dims[1], decoder_guidance_proj_dims[1])
        self.decoder3 = UPmy(decoder_dims[1], decoder_dims[2], 0)
        self.decoder4 = UPmy(decoder_dims[2], decoder_dims[3], 0)
        # self.head = nn.Conv2d(decoder_dims[1], 1, kernel_size=3, stride=1, padding=1)
        self.head = nn.Conv2d(decoder_dims[3], 1, kernel_size=3, stride=1, padding=1)
        self.upsample1 = nn.ConvTranspose2d(768, 256, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(768, 128, kernel_size=4, stride=4)
        self.pad_len = pad_len
    
    def feature_map(self, img_feats, text_feats):
        # concatenated feature volume for feature aggregation baselines
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        img_feats = repeat(img_feats, "B C H W -> B C T H W", T=text_feats.shape[1])
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        text_feats = text_feats.mean(dim=-2) # average text features over different prompts
        text_feats = F.normalize(text_feats, dim=-1) # B T C
        text_feats = repeat(text_feats, "B T C -> B C T H W", H=img_feats.shape[-2], W=img_feats.shape[-1])
        return torch.cat((img_feats, text_feats), dim=1) # B 2C T H W

    def correlation(self, img_feats, text_feats):
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        corr = torch.einsum('bchw, btpc -> bpthw', img_feats, text_feats)
        return corr

    def corr_embed(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B P T H W -> (B T) P H W')
        corr_embed = self.conv1(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed
    
    def corr_projection(self, x, proj):
        corr_embed = rearrange(x, 'B C T H W -> B T H W C')
        corr_embed = proj(corr_embed)
        corr_embed = rearrange(corr_embed, 'B T H W C -> B C T H W')
        return corr_embed

    def upsample(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = F.interpolate(corr_embed, scale_factor=2, mode='bilinear', align_corners=True)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed

    def conv_decoder(self, x, guidance):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = self.decoder1(corr_embed, guidance[0])
        corr_embed = self.decoder2(corr_embed, guidance[1])
        corr_embed = self.decoder3(corr_embed, None)
        corr_embed = self.decoder4(corr_embed, None)
        corr_embed = self.head(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
        return corr_embed
    
    def forward(self, img_feats, text_feats, appearance_guidance):
        """
        Arguments:
            img_feats: (B, C, H, W)
            text_feats: (B, T, P, C) T是类别的个数
            apperance_guidance: tuple of (B, C, H, W)
        """


        corr = self.correlation(img_feats, text_feats)

        #corr = self.feature_map(img_feats, text_feats)
        corr_embed = self.corr_embed(corr)

        projected_guidance, projected_text_guidance, projected_decoder_guidance = None, None, [None, None]
        if self.clip_dense_proj_layers is not None:
            #print('encod_guid')
            projected_guidance = self.clip_dense_proj_layers(appearance_guidance)
            projected_guidance = sum(projected_guidance)
            projected_guidance = rearrange(projected_guidance,'B (H W) C -> B C H W',H=img_feats.shape[-2])

        clip_L4 = rearrange(appearance_guidance[3], "B (H W) C -> B C H W", H=24)
        clip_L8 = rearrange(appearance_guidance[7], "B (H W) C -> B C H W", H=24)
        res4 = self.upsample1(clip_L4)
        res5 = self.upsample2(clip_L8)
        
        if self.decoder_guidance_projection is not None:
            #print('decod_guid')
            projected_decoder_guidance = [proj(g) for proj, g in zip(self.decoder_guidance_projection, [res4,res5])]

        if self.text_guidance_projection is not None:
            #print('textGuid')
            text_feats = text_feats.mean(dim=-2)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            projected_text_guidance = self.text_guidance_projection(text_feats)

        for layer in self.layers:
            # print('enc')
            corr_embed = layer(corr_embed, projected_guidance, projected_text_guidance)

        logit = self.conv_decoder(corr_embed, projected_decoder_guidance)

 
        return logit

class FusionAggregatorVer14bd(nn.Module):
    def __init__(self, 
        text_guidance_dim=512,
        text_guidance_proj_dim=128,
        appearance_guidance_dim=512,
        appearance_guidance_proj_dim=128,
        decoder_dims = (64, 32),
        decoder_guidance_dims=(256, 128),
        decoder_guidance_proj_dims=(32, 16),
        num_layers=4,
        nheads=4, 
        hidden_dim=128,
        pooling_size=(6, 6),
        feature_resolution=(24, 24),
        window_size=12,
        attention_type='linear',
        prompt_channel=1,
        pad_len=256,
    ) -> None:
        """
        Cost Aggregation Model for CAT-Seg
        Args:
            text_guidance_dim: Dimension of text guidance
            text_guidance_proj_dim: Dimension of projected text guidance
            appearance_guidance_dim: Dimension of appearance guidance
            appearance_guidance_proj_dim: Dimension of projected appearance guidance
            decoder_dims: Upsampling decoder dimensions
            decoder_guidance_dims: Upsampling decoder guidance dimensions
            decoder_guidance_proj_dims: Upsampling decoder guidance projected dimensions
            num_layers: Number of layers for the aggregator
            nheads: Number of attention heads
            hidden_dim: Hidden dimension for transformer blocks
            pooling_size: Pooling size for the class aggregation layer
                          To reduce computation, we apply pooling in class aggregation blocks to reduce the number of tokens during training
            feature_resolution: Feature resolution for spatial aggregation
            window_size: Window size for Swin block in spatial aggregation
            attention_type: Attention type for the class aggregation. 
            prompt_channel: Number of prompts for ensembling text features. Default: 1
            pad_len: Padding length for the class aggregation. Default: 256
                     pad_len enforces the class aggregation block to have a fixed length of tokens for all inputs
                     This means it either pads the sequence with learnable tokens in class aggregation,
                     or truncates the classes with the initial CLIP cosine-similarity scores.
                     Set pad_len to 0 to disable this feature.
            """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.layers = nn.ModuleList([
            AggregatorLayer(
                hidden_dim=hidden_dim, text_guidance_dim=text_guidance_proj_dim, appearance_guidance=appearance_guidance_proj_dim, 
                nheads=nheads, input_resolution=feature_resolution, pooling_size=pooling_size, window_size=window_size, attention_type=attention_type, pad_len=pad_len,
            ) for _ in range(num_layers)
        ])

        # self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.conv1 = nn.Conv2d(512, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.guidance_projection = nn.Sequential(
            nn.Conv2d(appearance_guidance_dim, appearance_guidance_proj_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        ) if appearance_guidance_dim > 0 else None
        
        self.text_guidance_projection = nn.Sequential(
            nn.Linear(text_guidance_dim, text_guidance_proj_dim),
            nn.ReLU(),
        ) if text_guidance_dim > 0 else None

        self.decoder_guidance_projection = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            ) for d, dp in zip(decoder_guidance_dims, decoder_guidance_proj_dims)
        ]) if decoder_guidance_dims[0] > 0 else None

        self.decoder1 = Up(hidden_dim, decoder_dims[0], decoder_guidance_proj_dims[0])
        self.decoder2 = Up(decoder_dims[0], decoder_dims[1], decoder_guidance_proj_dims[1])
        self.head = nn.Conv2d(decoder_dims[1], 1, kernel_size=3, stride=1, padding=1)

        self.pad_len = pad_len

    def feature_map(self, img_feats, text_feats):
        # concatenated feature volume for feature aggregation baselines
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        img_feats = repeat(img_feats, "B C H W -> B C T H W", T=text_feats.shape[1])
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        text_feats = text_feats.mean(dim=-2) # average text features over different prompts
        text_feats = F.normalize(text_feats, dim=-1) # B T C
        text_feats = repeat(text_feats, "B T C -> B C T H W", H=img_feats.shape[-2], W=img_feats.shape[-1])
        return torch.cat((img_feats, text_feats), dim=1) # B 2C T H W

    def correlation(self, img_feats, text_feats):
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        corr = torch.einsum('bchw, btpc -> bpthw', img_feats, text_feats)
        return corr

    def corr_embed(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B P T H W -> (B T) P H W')
        corr_embed = self.conv1(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed
    
    def corr_projection(self, x, proj):
        corr_embed = rearrange(x, 'B C T H W -> B T H W C')
        corr_embed = proj(corr_embed)
        corr_embed = rearrange(corr_embed, 'B T H W C -> B C T H W')
        return corr_embed

    def upsample(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = F.interpolate(corr_embed, scale_factor=2, mode='bilinear', align_corners=True)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed

    def conv_decoder(self, x, guidance):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = self.decoder1(corr_embed, guidance[0])
        corr_embed = self.decoder2(corr_embed, guidance[1])
        corr_embed = self.head(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
        return corr_embed
    
    def forward(self, img_feats, text_feats, appearance_guidance):
        """
        Arguments:
            img_feats: (B, C, H, W)
            text_feats: (B, T, P, C) T是类别的个数
            apperance_guidance: tuple of (B, C, H, W)
        """

        classes = None

        # corr = self.correlation(img_feats, text_feats)
        # if self.pad_len > 0 and text_feats.size(1) > self.pad_len:
        #     avg = corr.permute(0, 2, 1, 3, 4).flatten(-3).max(dim=-1)[0] 
        #     classes = avg.topk(self.pad_len, dim=-1, sorted=False)[1]
        #     th_text = F.normalize(text_feats, dim=-1)
        #     th_text = torch.gather(th_text, dim=1, index=classes[..., None, None].expand(-1, -1, th_text.size(-2), th_text.size(-1)))
        #     orig_clases = text_feats.size(1)
        #     img_feats = F.normalize(img_feats, dim=1) # B C H W
        #     text_feats = th_text
        #     corr = torch.einsum('bchw, btpc -> bpthw', img_feats, th_text)
        #corr = self.feature_map(img_feats, text_feats)
        class_num = text_feats.shape[1]
        img_feats = img_feats.unsqueeze(2).repeat(1,1,class_num,1,1) # B C H W -> B C T H W

        corr_embed = self.corr_embed(img_feats)

        projected_guidance, projected_text_guidance, projected_decoder_guidance = None, None, [None, None]
        if self.guidance_projection is not None:
            projected_guidance = self.guidance_projection(appearance_guidance[0])
        if self.decoder_guidance_projection is not None:
            projected_decoder_guidance = [proj(g) for proj, g in zip(self.decoder_guidance_projection, appearance_guidance[1:])]

        if self.text_guidance_projection is not None:
            text_feats = text_feats.mean(dim=-2)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            projected_text_guidance = self.text_guidance_projection(text_feats)

        for layer in self.layers:
            corr_embed = layer(corr_embed, projected_guidance, projected_text_guidance)

        logit = self.conv_decoder(corr_embed, projected_decoder_guidance)
        if classes is not None:
            out = torch.full((logit.size(0), orig_clases, logit.size(2), logit.size(3)), -100., device=logit.device)
            out.scatter_(dim=1, index=classes[..., None, None].expand(-1, -1, logit.size(-2), logit.size(-1)), src=logit)
            logit = out
        return logit
class FusionAggregatorVer14b(nn.Module):
    def __init__(self, 
        text_guidance_dim=512,
        text_guidance_proj_dim=128,
        appearance_guidance_dim=512,
        appearance_guidance_proj_dim=128,
        # decoder_dims = (64, 32),
        decoder_dims = (128,96,64,32),
        decoder_guidance_dims=(256, 128),
        decoder_guidance_proj_dims=(32, 16),
        num_layers=4,
        nheads=4, 
        hidden_dim=128,
        pooling_size=(6, 6),
        feature_resolution=(24, 24),
        window_size=12,
        attention_type='linear',
        prompt_channel=1,
        pad_len=256,
    ) -> None:
        """
        Cost Aggregation Model for CAT-Seg
        Args:
            text_guidance_dim: Dimension of text guidance
            text_guidance_proj_dim: Dimension of projected text guidance
            appearance_guidance_dim: Dimension of appearance guidance
            appearance_guidance_proj_dim: Dimension of projected appearance guidance
            decoder_dims: Upsampling decoder dimensions
            decoder_guidance_dims: Upsampling decoder guidance dimensions
            decoder_guidance_proj_dims: Upsampling decoder guidance projected dimensions
            num_layers: Number of layers for the aggregator
            nheads: Number of attention heads
            hidden_dim: Hidden dimension for transformer blocks
            pooling_size: Pooling size for the class aggregation layer
                          To reduce computation, we apply pooling in class aggregation blocks to reduce the number of tokens during training
            feature_resolution: Feature resolution for spatial aggregation
            window_size: Window size for Swin block in spatial aggregation
            attention_type: Attention type for the class aggregation. 
            prompt_channel: Number of prompts for ensembling text features. Default: 1
            pad_len: Padding length for the class aggregation. Default: 256
                     pad_len enforces the class aggregation block to have a fixed length of tokens for all inputs
                     This means it either pads the sequence with learnable tokens in class aggregation,
                     or truncates the classes with the initial CLIP cosine-similarity scores.
                     Set pad_len to 0 to disable this feature.
            """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.layers = nn.ModuleList([
            AggregatorLayer(
                hidden_dim=hidden_dim, text_guidance_dim=text_guidance_proj_dim, appearance_guidance=appearance_guidance_proj_dim, 
                nheads=nheads, input_resolution=feature_resolution, pooling_size=pooling_size, window_size=window_size, attention_type=attention_type, pad_len=pad_len,
            ) for _ in range(num_layers)
        ])

        self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)

        self.guidance_projection = nn.Sequential(
            nn.Conv2d(appearance_guidance_dim, appearance_guidance_proj_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        ) if appearance_guidance_dim > 0 else None

        self.text_guidance_projection = nn.Sequential(
            nn.Linear(text_guidance_dim, text_guidance_proj_dim),
            nn.ReLU(),
        ) if text_guidance_dim > 0 else None

        self.decoder_guidance_projection = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            ) for d, dp in zip(decoder_guidance_dims, decoder_guidance_proj_dims)
        ]) if decoder_guidance_dims[0] > 0 else None

        self.decoder1 = UPmy(hidden_dim, decoder_dims[0], decoder_guidance_proj_dims[0])
        self.decoder2 = UPmy(decoder_dims[0], decoder_dims[1], decoder_guidance_proj_dims[1])
        self.decoder3 = UPmy(decoder_dims[1], decoder_dims[2], 0)
        self.decoder4 = UPmy(decoder_dims[2], decoder_dims[3], 0)
        # self.head = nn.Conv2d(decoder_dims[1], 1, kernel_size=3, stride=1, padding=1)
        self.head = nn.Conv2d(decoder_dims[3], 1, kernel_size=3, stride=1, padding=1)

        self.pad_len = pad_len

    def feature_map(self, img_feats, text_feats):
        # concatenated feature volume for feature aggregation baselines
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        img_feats = repeat(img_feats, "B C H W -> B C T H W", T=text_feats.shape[1])
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        text_feats = text_feats.mean(dim=-2) # average text features over different prompts
        text_feats = F.normalize(text_feats, dim=-1) # B T C
        text_feats = repeat(text_feats, "B T C -> B C T H W", H=img_feats.shape[-2], W=img_feats.shape[-1])
        return torch.cat((img_feats, text_feats), dim=1) # B 2C T H W

    def correlation(self, img_feats, text_feats):
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        corr = torch.einsum('bchw, btpc -> bpthw', img_feats, text_feats)
        return corr

    def corr_embed(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B P T H W -> (B T) P H W')
        corr_embed = self.conv1(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed
    
    def corr_projection(self, x, proj):
        corr_embed = rearrange(x, 'B C T H W -> B T H W C')
        corr_embed = proj(corr_embed)
        corr_embed = rearrange(corr_embed, 'B T H W C -> B C T H W')
        return corr_embed

    def upsample(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = F.interpolate(corr_embed, scale_factor=2, mode='bilinear', align_corners=True)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed

    def conv_decoder(self, x, guidance):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = self.decoder1(corr_embed, guidance[0])
        corr_embed = self.decoder2(corr_embed, guidance[1])
        corr_embed = self.decoder3(corr_embed, None)
        corr_embed = self.decoder4(corr_embed, None)
        corr_embed = self.head(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
        return corr_embed
    
    def forward(self, img_feats, text_feats, appearance_guidance):
        """
        Arguments:
            img_feats: (B, C, H, W)
            text_feats: (B, T, P, C) T是类别的个数
            apperance_guidance: tuple of (B, C, H, W)
        """

        classes = None

        corr = self.correlation(img_feats, text_feats)
        if self.pad_len > 0 and text_feats.size(1) > self.pad_len:
            avg = corr.permute(0, 2, 1, 3, 4).flatten(-3).max(dim=-1)[0] 
            classes = avg.topk(self.pad_len, dim=-1, sorted=False)[1]
            th_text = F.normalize(text_feats, dim=-1)
            th_text = torch.gather(th_text, dim=1, index=classes[..., None, None].expand(-1, -1, th_text.size(-2), th_text.size(-1)))
            orig_clases = text_feats.size(1)
            img_feats = F.normalize(img_feats, dim=1) # B C H W
            text_feats = th_text
            corr = torch.einsum('bchw, btpc -> bpthw', img_feats, th_text)
        #corr = self.feature_map(img_feats, text_feats)
        corr_embed = self.corr_embed(corr)

        projected_guidance, projected_text_guidance, projected_decoder_guidance = None, None, [None, None]
        if self.guidance_projection is not None:
            projected_guidance = self.guidance_projection(appearance_guidance[0])

        if self.decoder_guidance_projection is not None:
            projected_decoder_guidance = [proj(g) for proj, g in zip(self.decoder_guidance_projection, appearance_guidance[1:])]

        if self.text_guidance_projection is not None:
            text_feats = text_feats.mean(dim=-2)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            projected_text_guidance = self.text_guidance_projection(text_feats)

        for layer in self.layers:
            corr_embed = layer(corr_embed, projected_guidance, projected_text_guidance)

        logit = self.conv_decoder(corr_embed, projected_decoder_guidance)
        if classes is not None:
            out = torch.full((logit.size(0), orig_clases, logit.size(2), logit.size(3)), -100., device=logit.device)
            out.scatter_(dim=1, index=classes[..., None, None].expand(-1, -1, logit.size(-2), logit.size(-1)), src=logit)
            logit = out
 
        return logit

    
class FusionAggregatorVer14(nn.Module):
    def __init__(self, 
        text_guidance_dim=512,
        text_guidance_proj_dim=128,
        appearance_guidance_dim=512,
        appearance_guidance_proj_dim=128,
        decoder_dims = (64, 32),
        decoder_guidance_dims=(256, 128),
        decoder_guidance_proj_dims=(32, 16),
        num_layers=4,
        nheads=4, 
        hidden_dim=128,
        pooling_size=(6, 6),
        feature_resolution=(24, 24),
        window_size=12,
        attention_type='linear',
        prompt_channel=1,
        pad_len=256,
    ) -> None:
        """
        Cost Aggregation Model for CAT-Seg
        Args:
            text_guidance_dim: Dimension of text guidance
            text_guidance_proj_dim: Dimension of projected text guidance
            appearance_guidance_dim: Dimension of appearance guidance
            appearance_guidance_proj_dim: Dimension of projected appearance guidance
            decoder_dims: Upsampling decoder dimensions
            decoder_guidance_dims: Upsampling decoder guidance dimensions
            decoder_guidance_proj_dims: Upsampling decoder guidance projected dimensions
            num_layers: Number of layers for the aggregator
            nheads: Number of attention heads
            hidden_dim: Hidden dimension for transformer blocks
            pooling_size: Pooling size for the class aggregation layer
                          To reduce computation, we apply pooling in class aggregation blocks to reduce the number of tokens during training
            feature_resolution: Feature resolution for spatial aggregation
            window_size: Window size for Swin block in spatial aggregation
            attention_type: Attention type for the class aggregation. 
            prompt_channel: Number of prompts for ensembling text features. Default: 1
            pad_len: Padding length for the class aggregation. Default: 256
                     pad_len enforces the class aggregation block to have a fixed length of tokens for all inputs
                     This means it either pads the sequence with learnable tokens in class aggregation,
                     or truncates the classes with the initial CLIP cosine-similarity scores.
                     Set pad_len to 0 to disable this feature.
            """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # self.layers = nn.ModuleList([
        #     AggregatorLayer(
        #         hidden_dim=hidden_dim, text_guidance_dim=text_guidance_proj_dim, appearance_guidance=appearance_guidance_proj_dim, 
        #         nheads=nheads, input_resolution=feature_resolution, pooling_size=pooling_size, window_size=window_size, attention_type=attention_type, pad_len=pad_len,
        #     ) for _ in range(num_layers)
        # ])

        # self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)
        #self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)
        #self.conv2 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)
        #self.fusion_corr = nn.Conv2d(2*hidden_dim, hidden_dim, kernel_size=7, stride=1, padding=3)
        # self.guidance_projection = nn.Sequential(
        #     nn.Conv2d(appearance_guidance_dim, appearance_guidance_proj_dim, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        # ) if appearance_guidance_dim > 0 else None
        
        # self.text_guidance_projection = nn.Sequential(
        #     nn.Linear(text_guidance_dim, text_guidance_proj_dim),
        #     nn.ReLU(),
        # ) if text_guidance_dim > 0 else None

        # self.CLIP_decoder_guidance_projection = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
        #         nn.ReLU(),
        #     ) for d, dp in zip(decoder_guidance_dims, decoder_guidance_proj_dims)
        # ]) if decoder_guidance_dims[0] > 0 else None
        
        # self.DINO_decoder_guidance_projection = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
        #         nn.ReLU(),
        #     ) for d, dp in zip(decoder_guidance_dims, decoder_guidance_proj_dims)
        # ]) if decoder_guidance_dims[0] > 0 else None
        
        # self.decoder1 = Up(hidden_dim, decoder_dims[0], decoder_guidance_proj_dims[0])
        # self.decoder2 = Up(decoder_dims[0], decoder_dims[1], decoder_guidance_proj_dims[1])
        
        # self.Fusiondecoder1=FusionUP(hidden_dim, decoder_dims[0], decoder_guidance_proj_dims[0])
        # self.Fusiondecoder2=FusionUP(decoder_dims[0], decoder_dims[1], decoder_guidance_proj_dims[1])
        # self.head = nn.Conv2d(decoder_dims[1], 1, kernel_size=3, stride=1, padding=1)
        # print(2333333333)

        self.pad_len = pad_len

    def feature_map(self, img_feats, text_feats):
        # concatenated feature volume for feature aggregation baselines
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        img_feats = repeat(img_feats, "B C H W -> B C T H W", T=text_feats.shape[1])
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        text_feats = text_feats.mean(dim=-2) # average text features over different prompts
        text_feats = F.normalize(text_feats, dim=-1) # B T C
        text_feats = repeat(text_feats, "B T C -> B C T H W", H=img_feats.shape[-2], W=img_feats.shape[-1])
        return torch.cat((img_feats, text_feats), dim=1) # B 2C T H W

    def correlation(self, img_feats, text_feats):
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        corr = torch.einsum('bchw, btpc -> bpthw', img_feats, text_feats)
        return corr

    def corr_embed(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B P T H W -> (B T) P H W')
        corr_embed = self.conv1(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed
    
    def corr_fusion_embed_minimum(self,clip_corr,dino_corr):
        # this one does not import a 1*1 conv
        # instead we modify the original embedding layer to adapt to the concatenated corr volume.
        B = clip_corr.shape[0]
        clip_corr = rearrange(clip_corr, 'B P T H W -> (B T) P H W')
        dino_corr = rearrange(dino_corr, 'B P T H W -> (B T) P H W')
        fused_corr = torch.cat([clip_corr,dino_corr],dim = 1)
        fused_corr = self.conv1_modified(fused_corr)
        fused_corr = rearrange(fused_corr, '(B T) C H W -> B C T H W', B=B)
        return fused_corr

    def corr_fusion_embed_seperate(self,clip_corr,dino_corr):
        # this one does not import a 1*1 conv
        # instead we modify the original embedding layer to adapt to the concatenated corr volume.
        B = clip_corr.shape[0]
        self.sigmoid = nn.Sigmoid()
        clip_corr = rearrange(clip_corr, 'B P T H W -> (B T) P H W')
        dino_corr = rearrange(dino_corr, 'B P T H W -> (B T) P H W')
 
        clip_embed_corr = self.conv1(clip_corr)
        dino_embed_corr = self.conv2(dino_corr)
        clip_embed_corr = self.sigmoid(clip_embed_corr)
        dino_embed_corr = self.sigmoid(dino_embed_corr)
        fused_corr = torch.cat([clip_embed_corr,dino_embed_corr],dim = 1)
        fused_corr = self.fusion_corr(fused_corr)
        fused_corr = self.sigmoid(fused_corr)
        fused_corr = rearrange(fused_corr, '(B T) C H W -> B C T H W', B=B)
        clip_embed_corr = rearrange(clip_embed_corr, '(B T) C H W -> B C T H W', B=B)
        dino_embed_corr = rearrange(dino_embed_corr, '(B T) C H W -> B C T H W', B=B)
        return fused_corr, clip_embed_corr
        
        
    def corr_fusion_embed(self,clip_corr,dino_corr):
        B = clip_corr.shape[0]
        clip_corr = rearrange(clip_corr, 'B P T H W -> (B T) P H W')
        dino_corr = rearrange(dino_corr, 'B P T H W -> (B T) P H W')
        fused_corr = torch.cat([clip_corr,dino_corr],dim = 1)
        fused_corr = self.fusion_corr(fused_corr)
        fused_corr = self.conv1(fused_corr)
        fused_corr = rearrange(fused_corr, '(B T) C H W -> B C T H W', B=B)
        return fused_corr
        # exit()
        
    def corr_projection(self, x, proj):
        corr_embed = rearrange(x, 'B C T H W -> B T H W C')
        corr_embed = proj(corr_embed)
        corr_embed = rearrange(corr_embed, 'B T H W C -> B C T H W')
        return corr_embed

    def upsample(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = F.interpolate(corr_embed, scale_factor=2, mode='bilinear', align_corners=True)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed

    # def conv_decoder(self, x, guidance):
    #     B = x.shape[0]
    #     corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
    #     corr_embed = self.decoder1(corr_embed, guidance[0])
    #     corr_embed = self.decoder2(corr_embed, guidance[1])
    #     corr_embed = self.head(corr_embed)
    #     corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
    #     return corr_embed
    
    def Fusion_conv_decoer(self, x, clip_guidance,dino_guidance):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')

        corr_embed = self.Fusiondecoder1(corr_embed, clip_guidance[0],dino_guidance[0])

        corr_embed = self.Fusiondecoder2(corr_embed, clip_guidance[1],dino_guidance[1])

        corr_embed = self.head(corr_embed)

        corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
        return corr_embed
    #def forward(self, img_feats,dino_feat, text_feats, appearance_guidance,dino_guidance):
    def forward(self, img_feats,dino_feat, text_feats, appearance_guidance,dino_guidance):
        """
        Arguments:
            img_feats: (B, C, H, W)
            text_feats: (B, T, P, C) T是类别的个数
            apperance_guidance: tuple of (B, C, H, W)
        """

        classes = None
        ################START Here modified by YCY START###############
        clip_corr = self.correlation(img_feats, text_feats)

        # dino_corr = self.correlation(dino_feat,text_feats)
        if self.pad_len > 0 and text_feats.size(1) > self.pad_len:
            avg = corr.permute(0, 2, 1, 3, 4).flatten(-3).max(dim=-1)[0] 
            # avg_dino = dino_corr.permute(0, 2, 1, 3, 4).flatten(-3).max(dim=-1)[0] 
            classes = avg.topk(self.pad_len, dim=-1, sorted=False)[1]
            # classes_dino = avg_dino.topk(self.pad_len, dim=-1, sorted=False)[1]
            th_text = F.normalize(text_feats, dim=-1)
            clip_th_text = torch.gather(th_text, dim=1, index=classes[..., None, None].expand(-1, -1, th_text.size(-2), th_text.size(-1)))
            # dino_th_text = torch.gather(th_text, dim=1, index=classes_dino[..., None, None].expand(-1, -1, th_text.size(-2), th_text.size(-1)))
            orig_clases = text_feats.size(1)
            img_feats = F.normalize(img_feats, dim=1) # B C H W
            # dino_feats = F.normalize(dino_feat, dim=1) # B C H W
            corr = torch.einsum('bchw, btpc -> bpthw', img_feats, clip_th_text)
            # dino_corr = torch.einsum('bchw, btpc -> bpthw', dino_feats, dino_th_text)
        clip_corr = clip_corr.squeeze(1)

        #corr = self.feature_map(img_feats, text_feats)
        #exit()
        # fused_corr = corr+dino_corr

        # exit()
        
        # fused_corr_embed = self.corr_embed(fused_corr)
        # fused_corr_embed1 = self.corr_fusion_embed_minimum(clip_corr = corr,dino_corr=dino_corr)
        # fused_corr_embed,clip_embed_corr  = self.corr_fusion_embed_seperate(clip_corr = corr,dino_corr=dino_corr)
        # add the res here #
        
        # fused_corr_embed = fused_corr_embed+clip_embed_corr

        # add the res here #
        # print(23333333333)
        # clip_corr_embed = self.corr_embed(corr)
        # dino_corr_embed = self.corr_embed(dino_corr)
        ################END Here modified by YCY END###############
        
        # print(clip_corr_embed)
        # projected_guidance, projected_text_guidance, CLIP_projected_decoder_guidance,DINO_projected_decoder_guidance  = None, None, [None, None], [None,None]
        # if self.guidance_projection is not None:
        #     projected_guidance = self.guidance_projection(appearance_guidance[0])
        # if self.CLIP_decoder_guidance_projection is not None:
        #     CLIP_projected_decoder_guidance = [proj(g) for proj, g in zip(self.CLIP_decoder_guidance_projection, appearance_guidance[1:])]
        # if self.DINO_decoder_guidance_projection is not None:
        #     DINO_projected_decoder_guidance = [proj(g) for proj, g in zip(self.DINO_decoder_guidance_projection, dino_guidance)]
        # if self.text_guidance_projection is not None:
        #     text_feats = text_feats.mean(dim=-2)
        #     text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        #     projected_text_guidance = self.text_guidance_projection(text_feats)

        # for layer in self.layers:
        #     fused_corr_embed = layer(fused_corr_embed, projected_guidance, projected_text_guidance)
        #     # clip_corr_embed = layer(clip_corr_embed, projected_guidance, projected_text_guidance)
        #     # dino_corr_embed = layer(dino_corr_embed, projected_guidance, projected_text_guidance)


        # # fusion_corr_embed = clip_corr_embed + dino_corr_embed

        # # logit = self.conv_decoder(clip_corr_embed, projected_decoder_guidance)

        # logit = self.Fusion_conv_decoer(fused_corr_embed, CLIP_projected_decoder_guidance,DINO_projected_decoder_guidance)

        logit = clip_corr
        if classes is not None:
            out = torch.full((logit.size(0), orig_clases, logit.size(2), logit.size(3)), -100., device=logit.device)
            out.scatter_(dim=1, index=classes[..., None, None].expand(-1, -1, logit.size(-2), logit.size(-1)), src=logit)
            logit = out
        return logit
class FusionAggregatorVer13(nn.Module):
    def __init__(self, 
        text_guidance_dim=512,
        text_guidance_proj_dim=128,
        appearance_guidance_dim=512,
        appearance_guidance_proj_dim=128,
        decoder_dims = (64, 32),
        decoder_guidance_dims=(256, 128),
        decoder_guidance_proj_dims=(32, 16),
        num_layers=4,
        nheads=4, 
        hidden_dim=128,
        pooling_size=(6, 6),
        feature_resolution=(24, 24),
        window_size=12,
        attention_type='linear',
        prompt_channel=1,
        pad_len=256,
    ) -> None:
        """
        Cost Aggregation Model for CAT-Seg
        Args:
            text_guidance_dim: Dimension of text guidance
            text_guidance_proj_dim: Dimension of projected text guidance
            appearance_guidance_dim: Dimension of appearance guidance
            appearance_guidance_proj_dim: Dimension of projected appearance guidance
            decoder_dims: Upsampling decoder dimensions
            decoder_guidance_dims: Upsampling decoder guidance dimensions
            decoder_guidance_proj_dims: Upsampling decoder guidance projected dimensions
            num_layers: Number of layers for the aggregator
            nheads: Number of attention heads
            hidden_dim: Hidden dimension for transformer blocks
            pooling_size: Pooling size for the class aggregation layer
                          To reduce computation, we apply pooling in class aggregation blocks to reduce the number of tokens during training
            feature_resolution: Feature resolution for spatial aggregation
            window_size: Window size for Swin block in spatial aggregation
            attention_type: Attention type for the class aggregation. 
            prompt_channel: Number of prompts for ensembling text features. Default: 1
            pad_len: Padding length for the class aggregation. Default: 256
                     pad_len enforces the class aggregation block to have a fixed length of tokens for all inputs
                     This means it either pads the sequence with learnable tokens in class aggregation,
                     or truncates the classes with the initial CLIP cosine-similarity scores.
                     Set pad_len to 0 to disable this feature.
            """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.layers = nn.ModuleList([
            AggregatorLayer(
                hidden_dim=hidden_dim, text_guidance_dim=text_guidance_proj_dim, appearance_guidance=appearance_guidance_proj_dim, 
                nheads=nheads, input_resolution=feature_resolution, pooling_size=pooling_size, window_size=window_size, attention_type=attention_type, pad_len=pad_len,
            ) for _ in range(num_layers)
        ])

        # self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.fusion_corr = nn.Conv2d(2*hidden_dim, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.guidance_projection = nn.Sequential(
            nn.Conv2d(appearance_guidance_dim, appearance_guidance_proj_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        ) if appearance_guidance_dim > 0 else None
        
        self.text_guidance_projection = nn.Sequential(
            nn.Linear(text_guidance_dim, text_guidance_proj_dim),
            nn.ReLU(),
        ) if text_guidance_dim > 0 else None

        self.CLIP_decoder_guidance_projection = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            ) for d, dp in zip(decoder_guidance_dims, decoder_guidance_proj_dims)
        ]) if decoder_guidance_dims[0] > 0 else None
        
        self.DINO_decoder_guidance_projection = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            ) for d, dp in zip(decoder_guidance_dims, decoder_guidance_proj_dims)
        ]) if decoder_guidance_dims[0] > 0 else None
        
        # self.decoder1 = Up(hidden_dim, decoder_dims[0], decoder_guidance_proj_dims[0])
        # self.decoder2 = Up(decoder_dims[0], decoder_dims[1], decoder_guidance_proj_dims[1])
        
        self.Fusiondecoder1=FusionUP(hidden_dim, decoder_dims[0], decoder_guidance_proj_dims[0])
        self.Fusiondecoder2=FusionUP(decoder_dims[0], decoder_dims[1], decoder_guidance_proj_dims[1])
        self.head = nn.Conv2d(decoder_dims[1], 1, kernel_size=3, stride=1, padding=1)
        # print(2333333333)

        self.pad_len = pad_len

    def feature_map(self, img_feats, text_feats):
        # concatenated feature volume for feature aggregation baselines
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        img_feats = repeat(img_feats, "B C H W -> B C T H W", T=text_feats.shape[1])
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        text_feats = text_feats.mean(dim=-2) # average text features over different prompts
        text_feats = F.normalize(text_feats, dim=-1) # B T C
        text_feats = repeat(text_feats, "B T C -> B C T H W", H=img_feats.shape[-2], W=img_feats.shape[-1])
        return torch.cat((img_feats, text_feats), dim=1) # B 2C T H W

    def correlation(self, img_feats, text_feats):
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        corr = torch.einsum('bchw, btpc -> bpthw', img_feats, text_feats)
        return corr

    def corr_embed(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B P T H W -> (B T) P H W')
        corr_embed = self.conv1(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed
    
    def corr_fusion_embed_minimum(self,clip_corr,dino_corr):
        # this one does not import a 1*1 conv
        # instead we modify the original embedding layer to adapt to the concatenated corr volume.
        B = clip_corr.shape[0]
        clip_corr = rearrange(clip_corr, 'B P T H W -> (B T) P H W')
        dino_corr = rearrange(dino_corr, 'B P T H W -> (B T) P H W')
        fused_corr = torch.cat([clip_corr,dino_corr],dim = 1)
        fused_corr = self.conv1_modified(fused_corr)
        fused_corr = rearrange(fused_corr, '(B T) C H W -> B C T H W', B=B)
        return fused_corr

    def corr_fusion_embed_seperate(self,clip_corr,dino_corr):
        # this one does not import a 1*1 conv
        # instead we modify the original embedding layer to adapt to the concatenated corr volume.
        B = clip_corr.shape[0]
        self.sigmoid = nn.Sigmoid()
        clip_corr = rearrange(clip_corr, 'B P T H W -> (B T) P H W')
        dino_corr = rearrange(dino_corr, 'B P T H W -> (B T) P H W')
 
        clip_embed_corr = self.conv1(clip_corr)
        dino_embed_corr = self.conv2(dino_corr)
        clip_embed_corr = self.sigmoid(clip_embed_corr)
        dino_embed_corr = self.sigmoid(dino_embed_corr)
        fused_corr = torch.cat([clip_embed_corr,dino_embed_corr],dim = 1)
        fused_corr = self.fusion_corr(fused_corr)
        fused_corr = self.sigmoid(fused_corr)
        fused_corr = rearrange(fused_corr, '(B T) C H W -> B C T H W', B=B)
        clip_embed_corr = rearrange(clip_embed_corr, '(B T) C H W -> B C T H W', B=B)
        dino_embed_corr = rearrange(dino_embed_corr, '(B T) C H W -> B C T H W', B=B)
        return fused_corr, clip_embed_corr
        
        
    def corr_fusion_embed(self,clip_corr,dino_corr):
        B = clip_corr.shape[0]
        clip_corr = rearrange(clip_corr, 'B P T H W -> (B T) P H W')
        dino_corr = rearrange(dino_corr, 'B P T H W -> (B T) P H W')
        fused_corr = torch.cat([clip_corr,dino_corr],dim = 1)
        fused_corr = self.fusion_corr(fused_corr)
        fused_corr = self.conv1(fused_corr)
        fused_corr = rearrange(fused_corr, '(B T) C H W -> B C T H W', B=B)
        return fused_corr
        # exit()
        
    def corr_projection(self, x, proj):
        corr_embed = rearrange(x, 'B C T H W -> B T H W C')
        corr_embed = proj(corr_embed)
        corr_embed = rearrange(corr_embed, 'B T H W C -> B C T H W')
        return corr_embed

    def upsample(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = F.interpolate(corr_embed, scale_factor=2, mode='bilinear', align_corners=True)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed

    # def conv_decoder(self, x, guidance):
    #     B = x.shape[0]
    #     corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
    #     corr_embed = self.decoder1(corr_embed, guidance[0])
    #     corr_embed = self.decoder2(corr_embed, guidance[1])
    #     corr_embed = self.head(corr_embed)
    #     corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
    #     return corr_embed
    
    def Fusion_conv_decoer(self, x, clip_guidance,dino_guidance):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')

        corr_embed = self.Fusiondecoder1(corr_embed, clip_guidance[0],dino_guidance[0])

        corr_embed = self.Fusiondecoder2(corr_embed, clip_guidance[1],dino_guidance[1])

        corr_embed = self.head(corr_embed)

        corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
        return corr_embed
    def forward(self, img_feats,dino_feat, text_feats, appearance_guidance,dino_guidance):
        """
        Arguments:
            img_feats: (B, C, H, W)
            text_feats: (B, T, P, C) T是类别的个数
            apperance_guidance: tuple of (B, C, H, W)
        """

        classes = None
        ################START Here modified by YCY START###############
        corr = self.correlation(img_feats, text_feats)

        dino_corr = self.correlation(dino_feat,text_feats)
        if self.pad_len > 0 and text_feats.size(1) > self.pad_len:
            avg = corr.permute(0, 2, 1, 3, 4).flatten(-3).max(dim=-1)[0] 
            avg_dino = dino_corr.permute(0, 2, 1, 3, 4).flatten(-3).max(dim=-1)[0] 
            classes = avg.topk(self.pad_len, dim=-1, sorted=False)[1]
            classes_dino = avg_dino.topk(self.pad_len, dim=-1, sorted=False)[1]
            th_text = F.normalize(text_feats, dim=-1)
            clip_th_text = torch.gather(th_text, dim=1, index=classes[..., None, None].expand(-1, -1, th_text.size(-2), th_text.size(-1)))
            dino_th_text = torch.gather(th_text, dim=1, index=classes_dino[..., None, None].expand(-1, -1, th_text.size(-2), th_text.size(-1)))
            orig_clases = text_feats.size(1)
            img_feats = F.normalize(img_feats, dim=1) # B C H W
            dino_feats = F.normalize(dino_feat, dim=1) # B C H W
            corr = torch.einsum('bchw, btpc -> bpthw', img_feats, clip_th_text)
            dino_corr = torch.einsum('bchw, btpc -> bpthw', dino_feats, dino_th_text)
        #corr = self.feature_map(img_feats, text_feats)
        #exit()
        # fused_corr = corr+dino_corr

        # exit()
        
        # fused_corr_embed = self.corr_embed(fused_corr)
        # fused_corr_embed1 = self.corr_fusion_embed_minimum(clip_corr = corr,dino_corr=dino_corr)
        fused_corr_embed,clip_embed_corr  = self.corr_fusion_embed_seperate(clip_corr = corr,dino_corr=dino_corr)
        # add the res here #

        fused_corr_embed = fused_corr_embed+clip_embed_corr

        # add the res here #
        # print(23333333333)
        # clip_corr_embed = self.corr_embed(corr)
        # dino_corr_embed = self.corr_embed(dino_corr)
        ################END Here modified by YCY END###############
        
        # print(clip_corr_embed)
        projected_guidance, projected_text_guidance, CLIP_projected_decoder_guidance,DINO_projected_decoder_guidance  = None, None, [None, None], [None,None]
        if self.guidance_projection is not None:
            projected_guidance = self.guidance_projection(appearance_guidance[0])
        if self.CLIP_decoder_guidance_projection is not None:
            CLIP_projected_decoder_guidance = [proj(g) for proj, g in zip(self.CLIP_decoder_guidance_projection, appearance_guidance[1:])]
        if self.DINO_decoder_guidance_projection is not None:
            DINO_projected_decoder_guidance = [proj(g) for proj, g in zip(self.DINO_decoder_guidance_projection, dino_guidance)]
        if self.text_guidance_projection is not None:
            text_feats = text_feats.mean(dim=-2)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            projected_text_guidance = self.text_guidance_projection(text_feats)

        for layer in self.layers:
            fused_corr_embed = layer(fused_corr_embed, projected_guidance, projected_text_guidance)
            # clip_corr_embed = layer(clip_corr_embed, projected_guidance, projected_text_guidance)
            # dino_corr_embed = layer(dino_corr_embed, projected_guidance, projected_text_guidance)


        # fusion_corr_embed = clip_corr_embed + dino_corr_embed

        # logit = self.conv_decoder(clip_corr_embed, projected_decoder_guidance)

        logit = self.Fusion_conv_decoer(fused_corr_embed, CLIP_projected_decoder_guidance,DINO_projected_decoder_guidance)

        
        if classes is not None:
            out = torch.full((logit.size(0), orig_clases, logit.size(2), logit.size(3)), -100., device=logit.device)
            out.scatter_(dim=1, index=classes[..., None, None].expand(-1, -1, logit.size(-2), logit.size(-1)), src=logit)
            logit = out
        return logit

class FusionAggregatorVer12a(nn.Module):
    def __init__(self, 
        text_guidance_dim=512,
        text_guidance_proj_dim=128,
        appearance_guidance_dim=512,
        appearance_guidance_proj_dim=128,
        decoder_dims = (64, 32),
        decoder_guidance_dims=(256, 128),
        decoder_guidance_proj_dims=(32, 16),
        num_layers=4,
        nheads=4, 
        hidden_dim=128,
        pooling_size=(6, 6),
        feature_resolution=(24, 24),
        window_size=12,
        attention_type='linear',
        prompt_channel=1,
        pad_len=256,
    ) -> None:
        """
        Cost Aggregation Model for CAT-Seg
        Args:
            text_guidance_dim: Dimension of text guidance
            text_guidance_proj_dim: Dimension of projected text guidance
            appearance_guidance_dim: Dimension of appearance guidance
            appearance_guidance_proj_dim: Dimension of projected appearance guidance
            decoder_dims: Upsampling decoder dimensions
            decoder_guidance_dims: Upsampling decoder guidance dimensions
            decoder_guidance_proj_dims: Upsampling decoder guidance projected dimensions
            num_layers: Number of layers for the aggregator
            nheads: Number of attention heads
            hidden_dim: Hidden dimension for transformer blocks
            pooling_size: Pooling size for the class aggregation layer
                          To reduce computation, we apply pooling in class aggregation blocks to reduce the number of tokens during training
            feature_resolution: Feature resolution for spatial aggregation
            window_size: Window size for Swin block in spatial aggregation
            attention_type: Attention type for the class aggregation. 
            prompt_channel: Number of prompts for ensembling text features. Default: 1
            pad_len: Padding length for the class aggregation. Default: 256
                     pad_len enforces the class aggregation block to have a fixed length of tokens for all inputs
                     This means it either pads the sequence with learnable tokens in class aggregation,
                     or truncates the classes with the initial CLIP cosine-similarity scores.
                     Set pad_len to 0 to disable this feature.
            """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.layers = nn.ModuleList([
            AggregatorLayerVer9e(
                hidden_dim=hidden_dim, text_guidance_dim=text_guidance_proj_dim, appearance_guidance=appearance_guidance_proj_dim, 
                nheads=nheads, input_resolution=feature_resolution, pooling_size=pooling_size, window_size=window_size, attention_type=attention_type, pad_len=pad_len,
            ) for _ in range(num_layers)
        ])

        # self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.fusion_corr = nn.Conv2d(2*hidden_dim, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.clip_guidance_projection = nn.Sequential(
            nn.Conv2d(appearance_guidance_dim, appearance_guidance_proj_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        ) if appearance_guidance_dim > 0 else None
        
        self.dino_guidance_projection = nn.Sequential(
            nn.Conv2d(appearance_guidance_dim, appearance_guidance_proj_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        ) if appearance_guidance_dim > 0 else None
        
        self.text_guidance_projection = nn.Sequential(
            nn.Linear(text_guidance_dim, text_guidance_proj_dim),
            nn.ReLU(),
        ) if text_guidance_dim > 0 else None

        self.CLIP_decoder_guidance_projection = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            ) for d, dp in zip(decoder_guidance_dims, decoder_guidance_proj_dims)
        ]) if decoder_guidance_dims[0] > 0 else None
        
        self.DINO_decoder_guidance_projection = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            ) for d, dp in zip(decoder_guidance_dims, decoder_guidance_proj_dims)
        ]) if decoder_guidance_dims[0] > 0 else None
        
        # self.decoder1 = Up(hidden_dim, decoder_dims[0], decoder_guidance_proj_dims[0])
        # self.decoder2 = Up(decoder_dims[0], decoder_dims[1], decoder_guidance_proj_dims[1])
        
        self.Fusiondecoder1=FusionUP(hidden_dim, decoder_dims[0], decoder_guidance_proj_dims[0])
        self.Fusiondecoder2=FusionUP(decoder_dims[0], decoder_dims[1], decoder_guidance_proj_dims[1])
        self.head = nn.Conv2d(decoder_dims[1], 1, kernel_size=3, stride=1, padding=1)

        self.pad_len = pad_len

    def feature_map(self, img_feats, text_feats):
        # concatenated feature volume for feature aggregation baselines
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        img_feats = repeat(img_feats, "B C H W -> B C T H W", T=text_feats.shape[1])
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        text_feats = text_feats.mean(dim=-2) # average text features over different prompts
        text_feats = F.normalize(text_feats, dim=-1) # B T C
        text_feats = repeat(text_feats, "B T C -> B C T H W", H=img_feats.shape[-2], W=img_feats.shape[-1])
        return torch.cat((img_feats, text_feats), dim=1) # B 2C T H W

    def correlation(self, img_feats, text_feats):
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        corr = torch.einsum('bchw, btpc -> bpthw', img_feats, text_feats)
        return corr

    def corr_embed(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B P T H W -> (B T) P H W')
        corr_embed = self.conv1(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed
    
    def corr_fusion_embed_minimum(self,clip_corr,dino_corr):
        # this one does not import a 1*1 conv
        # instead we modify the original embedding layer to adapt to the concatenated corr volume.
        B = clip_corr.shape[0]
        clip_corr = rearrange(clip_corr, 'B P T H W -> (B T) P H W')
        dino_corr = rearrange(dino_corr, 'B P T H W -> (B T) P H W')
        fused_corr = torch.cat([clip_corr,dino_corr],dim = 1)
        fused_corr = self.conv1_modified(fused_corr)
        fused_corr = rearrange(fused_corr, '(B T) C H W -> B C T H W', B=B)
        return fused_corr

    def corr_fusion_embed_seperate(self,clip_corr,dino_corr):
        # this one does not import a 1*1 conv
        # instead we modify the original embedding layer to adapt to the concatenated corr volume.
        B = clip_corr.shape[0]
        self.sigmoid = nn.Sigmoid()
        clip_corr = rearrange(clip_corr, 'B P T H W -> (B T) P H W')
        dino_corr = rearrange(dino_corr, 'B P T H W -> (B T) P H W')
 
        clip_embed_corr = self.conv1(clip_corr)
        dino_embed_corr = self.conv2(dino_corr)
        clip_embed_corr = self.sigmoid(clip_embed_corr)
        dino_embed_corr = self.sigmoid(dino_embed_corr)
        fused_corr = torch.cat([clip_embed_corr,dino_embed_corr],dim = 1)
        fused_corr = self.fusion_corr(fused_corr)
        fused_corr = self.sigmoid(fused_corr)
        fused_corr = rearrange(fused_corr, '(B T) C H W -> B C T H W', B=B)
        clip_embed_corr = rearrange(clip_embed_corr, '(B T) C H W -> B C T H W', B=B)
        dino_embed_corr = rearrange(dino_embed_corr, '(B T) C H W -> B C T H W', B=B)
        return fused_corr, clip_embed_corr, dino_embed_corr
        
        
    def corr_fusion_embed(self,clip_corr,dino_corr):
        B = clip_corr.shape[0]
        clip_corr = rearrange(clip_corr, 'B P T H W -> (B T) P H W')
        dino_corr = rearrange(dino_corr, 'B P T H W -> (B T) P H W')
        fused_corr = torch.cat([clip_corr,dino_corr],dim = 1)
        fused_corr = self.fusion_corr(fused_corr)
        fused_corr = self.conv1(fused_corr)
        fused_corr = rearrange(fused_corr, '(B T) C H W -> B C T H W', B=B)
        return fused_corr
        # exit()
        
    def corr_projection(self, x, proj):
        corr_embed = rearrange(x, 'B C T H W -> B T H W C')
        corr_embed = proj(corr_embed)
        corr_embed = rearrange(corr_embed, 'B T H W C -> B C T H W')
        return corr_embed

    def upsample(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = F.interpolate(corr_embed, scale_factor=2, mode='bilinear', align_corners=True)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed

    def conv_decoder(self, x, guidance):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = self.decoder1(corr_embed, guidance[0])
        corr_embed = self.decoder2(corr_embed, guidance[1])
        corr_embed = self.head(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
        return corr_embed
    
    def Fusion_conv_decoer(self, x, clip_guidance,dino_guidance):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = self.Fusiondecoder1(corr_embed, clip_guidance[0],dino_guidance[0])
        corr_embed = self.Fusiondecoder2(corr_embed, clip_guidance[1],dino_guidance[1])
        corr_embed = self.head(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
        return corr_embed
    def forward(self, img_feats,dino_feat, text_feats, appearance_guidance,dino_guidance):
        """
        Arguments:
            img_feats: (B, C, H, W)
            text_feats: (B, T, P, C) T是类别的个数
            apperance_guidance: tuple of (B, C, H, W)
        """

        classes = None
        ################START Here modified by YCY START###############
        corr = self.correlation(img_feats, text_feats)
        dino_corr = self.correlation(dino_feat,text_feats)
        if self.pad_len > 0 and text_feats.size(1) > self.pad_len:
            avg = corr.permute(0, 2, 1, 3, 4).flatten(-3).max(dim=-1)[0] 
            avg_dino = dino_corr.permute(0, 2, 1, 3, 4).flatten(-3).max(dim=-1)[0] 
            classes = avg.topk(self.pad_len, dim=-1, sorted=False)[1]
            classes_dino = avg_dino.topk(self.pad_len, dim=-1, sorted=False)[1]
            th_text = F.normalize(text_feats, dim=-1)
            clip_th_text = torch.gather(th_text, dim=1, index=classes[..., None, None].expand(-1, -1, th_text.size(-2), th_text.size(-1)))
            dino_th_text = torch.gather(th_text, dim=1, index=classes_dino[..., None, None].expand(-1, -1, th_text.size(-2), th_text.size(-1)))
            orig_clases = text_feats.size(1)
            img_feats = F.normalize(img_feats, dim=1) # B C H W
            dino_feats = F.normalize(dino_feat, dim=1) # B C H W
            corr = torch.einsum('bchw, btpc -> bpthw', img_feats, clip_th_text)
            dino_corr = torch.einsum('bchw, btpc -> bpthw', dino_feats, dino_th_text)
        #corr = self.feature_map(img_feats, text_feats)
        #exit()
        # fused_corr = corr+dino_corr

        # exit()
        
        # fused_corr_embed = self.corr_embed(fused_corr)
        # fused_corr_embed1 = self.corr_fusion_embed_minimum(clip_corr = corr,dino_corr=dino_corr)
        fused_corr_embed,clip_embed_corr, dino_embed_corr  = self.corr_fusion_embed_seperate(clip_corr = corr,dino_corr=dino_corr)
        # add the res here #

        fused_corr_embed = fused_corr_embed+clip_embed_corr

        # add the res here #
        # print(23333333333)
        # clip_corr_embed = self.corr_embed(corr)
        # dino_corr_embed = self.corr_embed(dino_corr)
        ################END Here modified by YCY END###############
        
        # print(clip_corr_embed)
        clip_projected_guidance, projected_text_guidance, CLIP_projected_decoder_guidance,DINO_projected_decoder_guidance  = None, None, [None, None], [None,None]
        dino_projected_guidance = None
        if self.clip_guidance_projection is not None:
            clip_projected_guidance = self.clip_guidance_projection(appearance_guidance[0])
        if self.dino_guidance_projection is not None:
            dino_projected_guidance = self.dino_guidance_projection(dino_feat)
        if self.CLIP_decoder_guidance_projection is not None:
            CLIP_projected_decoder_guidance = [proj(g) for proj, g in zip(self.CLIP_decoder_guidance_projection, appearance_guidance[1:])]
        if self.DINO_decoder_guidance_projection is not None:
            DINO_projected_decoder_guidance = [proj(g) for proj, g in zip(self.DINO_decoder_guidance_projection, dino_guidance)]
        if self.text_guidance_projection is not None:
            text_feats = text_feats.mean(dim=-2)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            projected_text_guidance = self.text_guidance_projection(text_feats)

        for layer in self.layers:
            fused_corr_embed = layer(fused_corr_embed, clip_projected_guidance, dino_projected_guidance, projected_text_guidance)
            # clip_corr_embed = layer(clip_corr_embed, projected_guidance, projected_text_guidance)
            # dino_corr_embed = layer(dino_corr_embed, projected_guidance, projected_text_guidance)
        

        # fusion_corr_embed = clip_corr_embed + dino_corr_embed

        # logit = self.conv_decoder(clip_corr_embed, projected_decoder_guidance)

        logit = self.Fusion_conv_decoer(fused_corr_embed, CLIP_projected_decoder_guidance,DINO_projected_decoder_guidance)

        
        if classes is not None:
            out = torch.full((logit.size(0), orig_clases, logit.size(2), logit.size(3)), -100., device=logit.device)
            out.scatter_(dim=1, index=classes[..., None, None].expand(-1, -1, logit.size(-2), logit.size(-1)), src=logit)
            logit = out
        return logit

class FusionAggregatorVer12(nn.Module):
    def __init__(self, 
        text_guidance_dim=512,
        text_guidance_proj_dim=128,
        appearance_guidance_dim=512,
        appearance_guidance_proj_dim=128,
        decoder_dims = (64, 32),
        decoder_guidance_dims=(256, 128),
        decoder_guidance_proj_dims=(32, 16),
        num_layers=4,
        nheads=4, 
        hidden_dim=128,
        pooling_size=(6, 6),
        feature_resolution=(24, 24),
        window_size=12,
        attention_type='linear',
        prompt_channel=1,
        pad_len=256,
    ) -> None:
        """
        Cost Aggregation Model for CAT-Seg
        Args:
            text_guidance_dim: Dimension of text guidance
            text_guidance_proj_dim: Dimension of projected text guidance
            appearance_guidance_dim: Dimension of appearance guidance
            appearance_guidance_proj_dim: Dimension of projected appearance guidance
            decoder_dims: Upsampling decoder dimensions
            decoder_guidance_dims: Upsampling decoder guidance dimensions
            decoder_guidance_proj_dims: Upsampling decoder guidance projected dimensions
            num_layers: Number of layers for the aggregator
            nheads: Number of attention heads
            hidden_dim: Hidden dimension for transformer blocks
            pooling_size: Pooling size for the class aggregation layer
                          To reduce computation, we apply pooling in class aggregation blocks to reduce the number of tokens during training
            feature_resolution: Feature resolution for spatial aggregation
            window_size: Window size for Swin block in spatial aggregation
            attention_type: Attention type for the class aggregation. 
            prompt_channel: Number of prompts for ensembling text features. Default: 1
            pad_len: Padding length for the class aggregation. Default: 256
                     pad_len enforces the class aggregation block to have a fixed length of tokens for all inputs
                     This means it either pads the sequence with learnable tokens in class aggregation,
                     or truncates the classes with the initial CLIP cosine-similarity scores.
                     Set pad_len to 0 to disable this feature.
            """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.layers = nn.ModuleList([
            AggregatorLayer(
                hidden_dim=hidden_dim, text_guidance_dim=text_guidance_proj_dim, appearance_guidance=appearance_guidance_proj_dim, 
                nheads=nheads, input_resolution=feature_resolution, pooling_size=pooling_size, window_size=window_size, attention_type=attention_type, pad_len=pad_len,
            ) for _ in range(num_layers)
        ])

        # self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.fusion_corr = nn.Conv2d(2*hidden_dim, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.guidance_projection = nn.Sequential(
            nn.Conv2d(appearance_guidance_dim, appearance_guidance_proj_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        ) if appearance_guidance_dim > 0 else None
        
        self.text_guidance_projection = nn.Sequential(
            nn.Linear(text_guidance_dim, text_guidance_proj_dim),
            nn.ReLU(),
        ) if text_guidance_dim > 0 else None

        self.CLIP_decoder_guidance_projection = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            ) for d, dp in zip(decoder_guidance_dims, decoder_guidance_proj_dims)
        ]) if decoder_guidance_dims[0] > 0 else None
        
        self.DINO_decoder_guidance_projection = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            ) for d, dp in zip(decoder_guidance_dims, decoder_guidance_proj_dims)
        ]) if decoder_guidance_dims[0] > 0 else None
        
        # self.decoder1 = Up(hidden_dim, decoder_dims[0], decoder_guidance_proj_dims[0])
        # self.decoder2 = Up(decoder_dims[0], decoder_dims[1], decoder_guidance_proj_dims[1])
        
        self.Fusiondecoder1=FusionUP(hidden_dim, decoder_dims[0], decoder_guidance_proj_dims[0])
        self.Fusiondecoder2=FusionUP(decoder_dims[0], decoder_dims[1], decoder_guidance_proj_dims[1])
        self.head = nn.Conv2d(decoder_dims[1], 1, kernel_size=3, stride=1, padding=1)
        print(2333333333)
        self.pad_len = pad_len

    def feature_map(self, img_feats, text_feats):
        # concatenated feature volume for feature aggregation baselines
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        img_feats = repeat(img_feats, "B C H W -> B C T H W", T=text_feats.shape[1])
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        text_feats = text_feats.mean(dim=-2) # average text features over different prompts
        text_feats = F.normalize(text_feats, dim=-1) # B T C
        text_feats = repeat(text_feats, "B T C -> B C T H W", H=img_feats.shape[-2], W=img_feats.shape[-1])
        return torch.cat((img_feats, text_feats), dim=1) # B 2C T H W

    def correlation(self, img_feats, text_feats):
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        corr = torch.einsum('bchw, btpc -> bpthw', img_feats, text_feats)
        return corr

    def corr_embed(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B P T H W -> (B T) P H W')
        corr_embed = self.conv1(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed
    
    def corr_fusion_embed_minimum(self,clip_corr,dino_corr):
        # this one does not import a 1*1 conv
        # instead we modify the original embedding layer to adapt to the concatenated corr volume.
        B = clip_corr.shape[0]
        clip_corr = rearrange(clip_corr, 'B P T H W -> (B T) P H W')
        dino_corr = rearrange(dino_corr, 'B P T H W -> (B T) P H W')
        fused_corr = torch.cat([clip_corr,dino_corr],dim = 1)
        fused_corr = self.conv1_modified(fused_corr)
        fused_corr = rearrange(fused_corr, '(B T) C H W -> B C T H W', B=B)
        return fused_corr

    def corr_fusion_embed_seperate(self,clip_corr,dino_corr):
        # this one does not import a 1*1 conv
        # instead we modify the original embedding layer to adapt to the concatenated corr volume.
        B = clip_corr.shape[0]
        self.sigmoid = nn.Sigmoid()
        clip_corr = rearrange(clip_corr, 'B P T H W -> (B T) P H W')
        dino_corr = rearrange(dino_corr, 'B P T H W -> (B T) P H W')
 
        clip_embed_corr = self.conv1(clip_corr)
        dino_embed_corr = self.conv2(dino_corr)
        clip_embed_corr = self.sigmoid(clip_embed_corr)
        dino_embed_corr = self.sigmoid(dino_embed_corr)
        fused_corr = torch.cat([clip_embed_corr,dino_embed_corr],dim = 1)
        fused_corr = self.fusion_corr(fused_corr)
        fused_corr = self.sigmoid(fused_corr)
        fused_corr = rearrange(fused_corr, '(B T) C H W -> B C T H W', B=B)
        clip_embed_corr = rearrange(clip_embed_corr, '(B T) C H W -> B C T H W', B=B)
        dino_embed_corr = rearrange(dino_embed_corr, '(B T) C H W -> B C T H W', B=B)
        return fused_corr, clip_embed_corr
        
        
    def corr_fusion_embed(self,clip_corr,dino_corr):
        B = clip_corr.shape[0]
        clip_corr = rearrange(clip_corr, 'B P T H W -> (B T) P H W')
        dino_corr = rearrange(dino_corr, 'B P T H W -> (B T) P H W')
        fused_corr = torch.cat([clip_corr,dino_corr],dim = 1)
        fused_corr = self.fusion_corr(fused_corr)
        fused_corr = self.conv1(fused_corr)
        fused_corr = rearrange(fused_corr, '(B T) C H W -> B C T H W', B=B)
        return fused_corr
        # exit()
        
    def corr_projection(self, x, proj):
        corr_embed = rearrange(x, 'B C T H W -> B T H W C')
        corr_embed = proj(corr_embed)
        corr_embed = rearrange(corr_embed, 'B T H W C -> B C T H W')
        return corr_embed

    def upsample(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = F.interpolate(corr_embed, scale_factor=2, mode='bilinear', align_corners=True)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed

    # def conv_decoder(self, x, guidance):
    #     B = x.shape[0]
    #     corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
    #     corr_embed = self.decoder1(corr_embed, guidance[0])
    #     corr_embed = self.decoder2(corr_embed, guidance[1])
    #     corr_embed = self.head(corr_embed)
    #     corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
    #     return corr_embed
    
    def Fusion_conv_decoer(self, x, clip_guidance,dino_guidance):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        print(corr_embed.shape)
        corr_embed = self.Fusiondecoder1(corr_embed, clip_guidance[0],dino_guidance[0])
        print(corr_embed.shape)
        corr_embed = self.Fusiondecoder2(corr_embed, clip_guidance[1],dino_guidance[1])
        print(corr_embed.shape)
        corr_embed = self.head(corr_embed)
        print(corr_embed.shape)
        print(233)
        exit()
        corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
        return corr_embed
    def forward(self, img_feats,dino_feat, text_feats, appearance_guidance,dino_guidance):
        """
        Arguments:
            img_feats: (B, C, H, W)
            text_feats: (B, T, P, C) T是类别的个数
            apperance_guidance: tuple of (B, C, H, W)
        """

        classes = None
        ################START Here modified by YCY START###############
        corr = self.correlation(img_feats, text_feats)
        dino_corr = self.correlation(dino_feat,text_feats)
        if self.pad_len > 0 and text_feats.size(1) > self.pad_len:
            avg = corr.permute(0, 2, 1, 3, 4).flatten(-3).max(dim=-1)[0] 
            avg_dino = dino_corr.permute(0, 2, 1, 3, 4).flatten(-3).max(dim=-1)[0] 
            classes = avg.topk(self.pad_len, dim=-1, sorted=False)[1]
            classes_dino = avg_dino.topk(self.pad_len, dim=-1, sorted=False)[1]
            th_text = F.normalize(text_feats, dim=-1)
            clip_th_text = torch.gather(th_text, dim=1, index=classes[..., None, None].expand(-1, -1, th_text.size(-2), th_text.size(-1)))
            dino_th_text = torch.gather(th_text, dim=1, index=classes_dino[..., None, None].expand(-1, -1, th_text.size(-2), th_text.size(-1)))
            orig_clases = text_feats.size(1)
            img_feats = F.normalize(img_feats, dim=1) # B C H W
            dino_feats = F.normalize(dino_feat, dim=1) # B C H W
            corr = torch.einsum('bchw, btpc -> bpthw', img_feats, clip_th_text)
            dino_corr = torch.einsum('bchw, btpc -> bpthw', dino_feats, dino_th_text)
        #corr = self.feature_map(img_feats, text_feats)
        #exit()
        # fused_corr = corr+dino_corr

        # exit()
        
        # fused_corr_embed = self.corr_embed(fused_corr)
        # fused_corr_embed1 = self.corr_fusion_embed_minimum(clip_corr = corr,dino_corr=dino_corr)
        fused_corr_embed,clip_embed_corr  = self.corr_fusion_embed_seperate(clip_corr = corr,dino_corr=dino_corr)
        # add the res here #

        fused_corr_embed = fused_corr_embed+clip_embed_corr

        # add the res here #
        # print(23333333333)
        # clip_corr_embed = self.corr_embed(corr)
        # dino_corr_embed = self.corr_embed(dino_corr)
        ################END Here modified by YCY END###############
        
        # print(clip_corr_embed)
        projected_guidance, projected_text_guidance, CLIP_projected_decoder_guidance,DINO_projected_decoder_guidance  = None, None, [None, None], [None,None]
        if self.guidance_projection is not None:
            projected_guidance = self.guidance_projection(appearance_guidance[0])
        if self.CLIP_decoder_guidance_projection is not None:
            CLIP_projected_decoder_guidance = [proj(g) for proj, g in zip(self.CLIP_decoder_guidance_projection, appearance_guidance[1:])]
        if self.DINO_decoder_guidance_projection is not None:
            DINO_projected_decoder_guidance = [proj(g) for proj, g in zip(self.DINO_decoder_guidance_projection, dino_guidance)]
        if self.text_guidance_projection is not None:
            text_feats = text_feats.mean(dim=-2)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            projected_text_guidance = self.text_guidance_projection(text_feats)

        for layer in self.layers:
            fused_corr_embed = layer(fused_corr_embed, projected_guidance, projected_text_guidance)
            # clip_corr_embed = layer(clip_corr_embed, projected_guidance, projected_text_guidance)
            # dino_corr_embed = layer(dino_corr_embed, projected_guidance, projected_text_guidance)


        # fusion_corr_embed = clip_corr_embed + dino_corr_embed

        # logit = self.conv_decoder(clip_corr_embed, projected_decoder_guidance)

        logit = self.Fusion_conv_decoer(fused_corr_embed, CLIP_projected_decoder_guidance,DINO_projected_decoder_guidance)

        
        if classes is not None:
            out = torch.full((logit.size(0), orig_clases, logit.size(2), logit.size(3)), -100., device=logit.device)
            out.scatter_(dim=1, index=classes[..., None, None].expand(-1, -1, logit.size(-2), logit.size(-1)), src=logit)
            logit = out
        return logit

class FusionAggregatorVer11(nn.Module):
    def __init__(self, 
        text_guidance_dim=512,
        text_guidance_proj_dim=128,
        appearance_guidance_dim=512,
        appearance_guidance_proj_dim=128,
        decoder_dims = (64, 32),
        decoder_guidance_dims=(256, 128),
        decoder_guidance_proj_dims=(32, 16),
        num_layers=4,
        nheads=4, 
        hidden_dim=128,
        pooling_size=(6, 6),
        feature_resolution=(24, 24),
        window_size=12,
        attention_type='linear',
        prompt_channel=1,
        pad_len=256,
    ) -> None:
        """
        Cost Aggregation Model for CAT-Seg
        Args:
            text_guidance_dim: Dimension of text guidance
            text_guidance_proj_dim: Dimension of projected text guidance
            appearance_guidance_dim: Dimension of appearance guidance
            appearance_guidance_proj_dim: Dimension of projected appearance guidance
            decoder_dims: Upsampling decoder dimensions
            decoder_guidance_dims: Upsampling decoder guidance dimensions
            decoder_guidance_proj_dims: Upsampling decoder guidance projected dimensions
            num_layers: Number of layers for the aggregator
            nheads: Number of attention heads
            hidden_dim: Hidden dimension for transformer blocks
            pooling_size: Pooling size for the class aggregation layer
                          To reduce computation, we apply pooling in class aggregation blocks to reduce the number of tokens during training
            feature_resolution: Feature resolution for spatial aggregation
            window_size: Window size for Swin block in spatial aggregation
            attention_type: Attention type for the class aggregation. 
            prompt_channel: Number of prompts for ensembling text features. Default: 1
            pad_len: Padding length for the class aggregation. Default: 256
                     pad_len enforces the class aggregation block to have a fixed length of tokens for all inputs
                     This means it either pads the sequence with learnable tokens in class aggregation,
                     or truncates the classes with the initial CLIP cosine-similarity scores.
                     Set pad_len to 0 to disable this feature.
            """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.layers = nn.ModuleList([
            AggregatorLayer(
                hidden_dim=hidden_dim, text_guidance_dim=text_guidance_proj_dim, appearance_guidance=appearance_guidance_proj_dim, 
                nheads=nheads, input_resolution=feature_resolution, pooling_size=pooling_size, window_size=window_size, attention_type=attention_type, pad_len=pad_len,
            ) for _ in range(num_layers)
        ])

        # self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.fusion_corr = nn.Conv2d(2*hidden_dim, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.guidance_projection = nn.Sequential(
            nn.Conv2d(appearance_guidance_dim, appearance_guidance_proj_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        ) if appearance_guidance_dim > 0 else None
        
        self.text_guidance_projection = nn.Sequential(
            nn.Linear(text_guidance_dim, text_guidance_proj_dim),
            nn.ReLU(),
        ) if text_guidance_dim > 0 else None

        self.decoder_guidance_projection = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            ) for d, dp in zip(decoder_guidance_dims, decoder_guidance_proj_dims)
        ]) if decoder_guidance_dims[0] > 0 else None

        self.decoder1 = Up(hidden_dim, decoder_dims[0], decoder_guidance_proj_dims[0])
        self.decoder2 = Up(decoder_dims[0], decoder_dims[1], decoder_guidance_proj_dims[1])
        self.head = nn.Conv2d(decoder_dims[1], 1, kernel_size=3, stride=1, padding=1)

        self.pad_len = pad_len

    def feature_map(self, img_feats, text_feats):
        # concatenated feature volume for feature aggregation baselines
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        img_feats = repeat(img_feats, "B C H W -> B C T H W", T=text_feats.shape[1])
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        text_feats = text_feats.mean(dim=-2) # average text features over different prompts
        text_feats = F.normalize(text_feats, dim=-1) # B T C
        text_feats = repeat(text_feats, "B T C -> B C T H W", H=img_feats.shape[-2], W=img_feats.shape[-1])
        return torch.cat((img_feats, text_feats), dim=1) # B 2C T H W

    def correlation(self, img_feats, text_feats):
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        corr = torch.einsum('bchw, btpc -> bpthw', img_feats, text_feats)
        return corr

    def corr_embed(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B P T H W -> (B T) P H W')
        corr_embed = self.conv1(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed
    
    def corr_fusion_embed_minimum(self,clip_corr,dino_corr):
        # this one does not import a 1*1 conv
        # instead we modify the original embedding layer to adapt to the concatenated corr volume.
        B = clip_corr.shape[0]
        clip_corr = rearrange(clip_corr, 'B P T H W -> (B T) P H W')
        dino_corr = rearrange(dino_corr, 'B P T H W -> (B T) P H W')
        fused_corr = torch.cat([clip_corr,dino_corr],dim = 1)
        fused_corr = self.conv1_modified(fused_corr)
        fused_corr = rearrange(fused_corr, '(B T) C H W -> B C T H W', B=B)
        return fused_corr

    def corr_fusion_embed_seperate(self,clip_corr,dino_corr):
        # this one does not import a 1*1 conv
        # instead we modify the original embedding layer to adapt to the concatenated corr volume.
        B = clip_corr.shape[0]
        clip_corr = rearrange(clip_corr, 'B P T H W -> (B T) P H W')
        dino_corr = rearrange(dino_corr, 'B P T H W -> (B T) P H W')
 
        clip_embed_corr = self.conv1(clip_corr)
        dino_embed_corr = self.conv2(dino_corr)

        fused_corr = torch.cat([clip_embed_corr,dino_embed_corr],dim = 1)
        fused_corr = self.fusion_corr(fused_corr)
        fused_corr = rearrange(fused_corr, '(B T) C H W -> B C T H W', B=B)

        return fused_corr
        
        
    def corr_fusion_embed(self,clip_corr,dino_corr):
        B = clip_corr.shape[0]
        clip_corr = rearrange(clip_corr, 'B P T H W -> (B T) P H W')
        dino_corr = rearrange(dino_corr, 'B P T H W -> (B T) P H W')
        fused_corr = torch.cat([clip_corr,dino_corr],dim = 1)
        fused_corr = self.fusion_corr(fused_corr)
        fused_corr = self.conv1(fused_corr)
        fused_corr = rearrange(fused_corr, '(B T) C H W -> B C T H W', B=B)
        return fused_corr
        # exit()
        
    def corr_projection(self, x, proj):
        corr_embed = rearrange(x, 'B C T H W -> B T H W C')
        corr_embed = proj(corr_embed)
        corr_embed = rearrange(corr_embed, 'B T H W C -> B C T H W')
        return corr_embed

    def upsample(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = F.interpolate(corr_embed, scale_factor=2, mode='bilinear', align_corners=True)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed

    def conv_decoder(self, x, guidance):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = self.decoder1(corr_embed, guidance[0])

        corr_embed = self.decoder2(corr_embed, guidance[1])
        corr_embed = self.head(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
        return corr_embed
    def forward(self, img_feats,dino_feat, text_feats, appearance_guidance):
        """
        Arguments:
            img_feats: (B, C, H, W)
            text_feats: (B, T, P, C) T是类别的个数
            apperance_guidance: tuple of (B, C, H, W)
        """

        classes = None
        ################START Here modified by YCY START###############
        corr = self.correlation(img_feats, text_feats)
        dino_corr = self.correlation(dino_feat,text_feats)
        if self.pad_len > 0 and text_feats.size(1) > self.pad_len:
            avg = corr.permute(0, 2, 1, 3, 4).flatten(-3).max(dim=-1)[0] 
            avg_dino = dino_corr.permute(0, 2, 1, 3, 4).flatten(-3).max(dim=-1)[0] 
            classes = avg.topk(self.pad_len, dim=-1, sorted=False)[1]
            classes_dino = avg_dino.topk(self.pad_len, dim=-1, sorted=False)[1]
            th_text = F.normalize(text_feats, dim=-1)
            clip_th_text = torch.gather(th_text, dim=1, index=classes[..., None, None].expand(-1, -1, th_text.size(-2), th_text.size(-1)))
            dino_th_text = torch.gather(th_text, dim=1, index=classes_dino[..., None, None].expand(-1, -1, th_text.size(-2), th_text.size(-1)))
            orig_clases = text_feats.size(1)
            img_feats = F.normalize(img_feats, dim=1) # B C H W
            dino_feats = F.normalize(dino_feat, dim=1) # B C H W
            corr = torch.einsum('bchw, btpc -> bpthw', img_feats, clip_th_text)
            dino_corr = torch.einsum('bchw, btpc -> bpthw', dino_feats, dino_th_text)
        #corr = self.feature_map(img_feats, text_feats)
        #exit()
        # fused_corr = corr+dino_corr

        # exit()
        
        # fused_corr_embed = self.corr_embed(fused_corr)
        # fused_corr_embed1 = self.corr_fusion_embed_minimum(clip_corr = corr,dino_corr=dino_corr)
        fused_corr_embed = self.corr_fusion_embed_seperate(clip_corr = corr,dino_corr=dino_corr)
        # clip_corr_embed = self.corr_embed(corr)
        # dino_corr_embed = self.corr_embed(dino_corr)
        ################END Here modified by YCY END###############
        
        # print(clip_corr_embed)
        projected_guidance, projected_text_guidance, projected_decoder_guidance = None, None, [None, None]
        if self.guidance_projection is not None:
            projected_guidance = self.guidance_projection(appearance_guidance[0])
        if self.decoder_guidance_projection is not None:
            projected_decoder_guidance = [proj(g) for proj, g in zip(self.decoder_guidance_projection, appearance_guidance[1:])]

        if self.text_guidance_projection is not None:
            text_feats = text_feats.mean(dim=-2)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            projected_text_guidance = self.text_guidance_projection(text_feats)

        for layer in self.layers:
            fused_corr_embed = layer(fused_corr_embed, projected_guidance, projected_text_guidance)
            # clip_corr_embed = layer(clip_corr_embed, projected_guidance, projected_text_guidance)
            # dino_corr_embed = layer(dino_corr_embed, projected_guidance, projected_text_guidance)


        # fusion_corr_embed = clip_corr_embed + dino_corr_embed

        # logit = self.conv_decoder(clip_corr_embed, projected_decoder_guidance)
        logit = self.conv_decoder(fused_corr_embed, projected_decoder_guidance)

        
        if classes is not None:
            out = torch.full((logit.size(0), orig_clases, logit.size(2), logit.size(3)), -100., device=logit.device)
            out.scatter_(dim=1, index=classes[..., None, None].expand(-1, -1, logit.size(-2), logit.size(-1)), src=logit)
            logit = out
        return logit
class FusionAggregatorVer10(nn.Module):
    def __init__(self, 
        text_guidance_dim=512,
        text_guidance_proj_dim=128,
        appearance_guidance_dim=512,
        appearance_guidance_proj_dim=128,
        decoder_dims = (64, 32),
        decoder_guidance_dims=(256, 128),
        decoder_guidance_proj_dims=(32, 16),
        num_layers=4,
        nheads=4, 
        hidden_dim=128,
        pooling_size=(6, 6),
        feature_resolution=(24, 24),
        window_size=12,
        attention_type='linear',
        prompt_channel=1,
        pad_len=256,
    ) -> None:
        """
        Cost Aggregation Model for CAT-Seg
        Args:
            text_guidance_dim: Dimension of text guidance
            text_guidance_proj_dim: Dimension of projected text guidance
            appearance_guidance_dim: Dimension of appearance guidance
            appearance_guidance_proj_dim: Dimension of projected appearance guidance
            decoder_dims: Upsampling decoder dimensions
            decoder_guidance_dims: Upsampling decoder guidance dimensions
            decoder_guidance_proj_dims: Upsampling decoder guidance projected dimensions
            num_layers: Number of layers for the aggregator
            nheads: Number of attention heads
            hidden_dim: Hidden dimension for transformer blocks
            pooling_size: Pooling size for the class aggregation layer
                          To reduce computation, we apply pooling in class aggregation blocks to reduce the number of tokens during training
            feature_resolution: Feature resolution for spatial aggregation
            window_size: Window size for Swin block in spatial aggregation
            attention_type: Attention type for the class aggregation. 
            prompt_channel: Number of prompts for ensembling text features. Default: 1
            pad_len: Padding length for the class aggregation. Default: 256
                     pad_len enforces the class aggregation block to have a fixed length of tokens for all inputs
                     This means it either pads the sequence with learnable tokens in class aggregation,
                     or truncates the classes with the initial CLIP cosine-similarity scores.
                     Set pad_len to 0 to disable this feature.
            """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.layers = nn.ModuleList([
            AggregatorLayer(
                hidden_dim=hidden_dim, text_guidance_dim=text_guidance_proj_dim, appearance_guidance=appearance_guidance_proj_dim, 
                nheads=nheads, input_resolution=feature_resolution, pooling_size=pooling_size, window_size=window_size, attention_type=attention_type, pad_len=pad_len,
            ) for _ in range(num_layers)
        ])

        # self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.fusion_corr = nn.Conv2d(2*hidden_dim, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.guidance_projection = nn.Sequential(
            nn.Conv2d(appearance_guidance_dim, appearance_guidance_proj_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        ) if appearance_guidance_dim > 0 else None
        
        self.text_guidance_projection = nn.Sequential(
            nn.Linear(text_guidance_dim, text_guidance_proj_dim),
            nn.ReLU(),
        ) if text_guidance_dim > 0 else None

        self.decoder_guidance_projection = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            ) for d, dp in zip(decoder_guidance_dims, decoder_guidance_proj_dims)
        ]) if decoder_guidance_dims[0] > 0 else None

        self.decoder1 = Up(hidden_dim, decoder_dims[0], decoder_guidance_proj_dims[0])
        self.decoder2 = Up(decoder_dims[0], decoder_dims[1], decoder_guidance_proj_dims[1])
        self.head = nn.Conv2d(decoder_dims[1], 1, kernel_size=3, stride=1, padding=1)

        self.pad_len = pad_len

    def feature_map(self, img_feats, text_feats):
        # concatenated feature volume for feature aggregation baselines
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        img_feats = repeat(img_feats, "B C H W -> B C T H W", T=text_feats.shape[1])
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        text_feats = text_feats.mean(dim=-2) # average text features over different prompts
        text_feats = F.normalize(text_feats, dim=-1) # B T C
        text_feats = repeat(text_feats, "B T C -> B C T H W", H=img_feats.shape[-2], W=img_feats.shape[-1])
        return torch.cat((img_feats, text_feats), dim=1) # B 2C T H W

    def correlation(self, img_feats, text_feats):
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        corr = torch.einsum('bchw, btpc -> bpthw', img_feats, text_feats)
        return corr

    def corr_embed(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B P T H W -> (B T) P H W')
        corr_embed = self.conv1(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed
    
    def corr_fusion_embed_minimum(self,clip_corr,dino_corr):
        # this one does not import a 1*1 conv
        # instead we modify the original embedding layer to adapt to the concatenated corr volume.
        B = clip_corr.shape[0]
        clip_corr = rearrange(clip_corr, 'B P T H W -> (B T) P H W')
        dino_corr = rearrange(dino_corr, 'B P T H W -> (B T) P H W')
        fused_corr = torch.cat([clip_corr,dino_corr],dim = 1)
        fused_corr = self.conv1_modified(fused_corr)
        fused_corr = rearrange(fused_corr, '(B T) C H W -> B C T H W', B=B)
        return fused_corr

    def corr_fusion_embed_seperate(self,clip_corr,dino_corr):
        # this one does not import a 1*1 conv
        # instead we modify the original embedding layer to adapt to the concatenated corr volume.
        B = clip_corr.shape[0]
        clip_corr = rearrange(clip_corr, 'B P T H W -> (B T) P H W')
        dino_corr = rearrange(dino_corr, 'B P T H W -> (B T) P H W')
 
        clip_embed_corr = self.conv1(clip_corr)
        dino_embed_corr = self.conv2(dino_corr)

        fused_corr = torch.cat([clip_embed_corr,dino_embed_corr],dim = 1)
        fused_corr = self.fusion_corr(fused_corr)
        fused_corr = rearrange(fused_corr, '(B T) C H W -> B C T H W', B=B)

        return fused_corr
        
        
    def corr_fusion_embed(self,clip_corr,dino_corr):
        B = clip_corr.shape[0]
        clip_corr = rearrange(clip_corr, 'B P T H W -> (B T) P H W')
        dino_corr = rearrange(dino_corr, 'B P T H W -> (B T) P H W')
        fused_corr = torch.cat([clip_corr,dino_corr],dim = 1)
        fused_corr = self.fusion_corr(fused_corr)
        fused_corr = self.conv1(fused_corr)
        fused_corr = rearrange(fused_corr, '(B T) C H W -> B C T H W', B=B)
        return fused_corr
        # exit()
        
    def corr_projection(self, x, proj):
        corr_embed = rearrange(x, 'B C T H W -> B T H W C')
        corr_embed = proj(corr_embed)
        corr_embed = rearrange(corr_embed, 'B T H W C -> B C T H W')
        return corr_embed

    def upsample(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = F.interpolate(corr_embed, scale_factor=2, mode='bilinear', align_corners=True)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed

    def conv_decoder(self, x, guidance):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = self.decoder1(corr_embed, guidance[0])

        corr_embed = self.decoder2(corr_embed, guidance[1])
        corr_embed = self.head(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
        return corr_embed
    def forward(self, img_feats,dino_feat, text_feats, appearance_guidance):
        """
        Arguments:
            img_feats: (B, C, H, W)
            text_feats: (B, T, P, C) T是类别的个数
            apperance_guidance: tuple of (B, C, H, W)
        """

        classes = None
        ################START Here modified by YCY START###############
        corr = self.correlation(img_feats, text_feats)
        dino_corr = self.correlation(dino_feat,text_feats)
        if self.pad_len > 0 and text_feats.size(1) > self.pad_len:
            avg = corr.permute(0, 2, 1, 3, 4).flatten(-3).max(dim=-1)[0] 
            avg_dino = dino_corr.permute(0, 2, 1, 3, 4).flatten(-3).max(dim=-1)[0] 
            classes = avg.topk(self.pad_len, dim=-1, sorted=False)[1]
            classes_dino = avg_dino.topk(self.pad_len, dim=-1, sorted=False)[1]
            th_text = F.normalize(text_feats, dim=-1)
            clip_th_text = torch.gather(th_text, dim=1, index=classes[..., None, None].expand(-1, -1, th_text.size(-2), th_text.size(-1)))
            dino_th_text = torch.gather(th_text, dim=1, index=classes_dino[..., None, None].expand(-1, -1, th_text.size(-2), th_text.size(-1)))
            orig_clases = text_feats.size(1)
            img_feats = F.normalize(img_feats, dim=1) # B C H W
            dino_feats = F.normalize(dino_feat, dim=1) # B C H W
            corr = torch.einsum('bchw, btpc -> bpthw', img_feats, clip_th_text)
            dino_corr = torch.einsum('bchw, btpc -> bpthw', dino_feats, dino_th_text)
        #corr = self.feature_map(img_feats, text_feats)
        #exit()
        # fused_corr = corr+dino_corr

        # exit()
        
        # fused_corr_embed = self.corr_embed(fused_corr)
        # fused_corr_embed1 = self.corr_fusion_embed_minimum(clip_corr = corr,dino_corr=dino_corr)
        fused_corr_embed = self.corr_fusion_embed_seperate(clip_corr = corr,dino_corr=dino_corr)
        # clip_corr_embed = self.corr_embed(corr)
        # dino_corr_embed = self.corr_embed(dino_corr)
        ################END Here modified by YCY END###############
        
        # print(clip_corr_embed)
        projected_guidance, projected_text_guidance, projected_decoder_guidance = None, None, [None, None]
        if self.guidance_projection is not None:
            projected_guidance = self.guidance_projection(appearance_guidance[0])
        if self.decoder_guidance_projection is not None:
            projected_decoder_guidance = [proj(g) for proj, g in zip(self.decoder_guidance_projection, appearance_guidance[1:])]

        if self.text_guidance_projection is not None:
            text_feats = text_feats.mean(dim=-2)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            projected_text_guidance = self.text_guidance_projection(text_feats)

        for layer in self.layers:
            fused_corr_embed = layer(fused_corr_embed, projected_guidance, projected_text_guidance)
            # clip_corr_embed = layer(clip_corr_embed, projected_guidance, projected_text_guidance)
            # dino_corr_embed = layer(dino_corr_embed, projected_guidance, projected_text_guidance)


        # fusion_corr_embed = clip_corr_embed + dino_corr_embed

        # logit = self.conv_decoder(clip_corr_embed, projected_decoder_guidance)
        logit = self.conv_decoder(fused_corr_embed, projected_decoder_guidance)

        
        if classes is not None:
            out = torch.full((logit.size(0), orig_clases, logit.size(2), logit.size(3)), -100., device=logit.device)
            out.scatter_(dim=1, index=classes[..., None, None].expand(-1, -1, logit.size(-2), logit.size(-1)), src=logit)
            logit = out
        return logit
class FusionAggregatorVer09(nn.Module):
    def __init__(self, 
        text_guidance_dim=512,
        text_guidance_proj_dim=128,
        appearance_guidance_dim=512,
        appearance_guidance_proj_dim=128,
        decoder_dims = (64, 32),
        decoder_guidance_dims=(256, 128),
        decoder_guidance_proj_dims=(32, 16),
        num_layers=4,
        nheads=4, 
        hidden_dim=128,
        pooling_size=(6, 6),
        feature_resolution=(24, 24),
        window_size=12,
        attention_type='linear',
        prompt_channel=1,
        pad_len=256,
    ) -> None:
        """
        Cost Aggregation Model for CAT-Seg
        Args:
            text_guidance_dim: Dimension of text guidance
            text_guidance_proj_dim: Dimension of projected text guidance
            appearance_guidance_dim: Dimension of appearance guidance
            appearance_guidance_proj_dim: Dimension of projected appearance guidance
            decoder_dims: Upsampling decoder dimensions
            decoder_guidance_dims: Upsampling decoder guidance dimensions
            decoder_guidance_proj_dims: Upsampling decoder guidance projected dimensions
            num_layers: Number of layers for the aggregator
            nheads: Number of attention heads
            hidden_dim: Hidden dimension for transformer blocks
            pooling_size: Pooling size for the class aggregation layer
                          To reduce computation, we apply pooling in class aggregation blocks to reduce the number of tokens during training
            feature_resolution: Feature resolution for spatial aggregation
            window_size: Window size for Swin block in spatial aggregation
            attention_type: Attention type for the class aggregation. 
            prompt_channel: Number of prompts for ensembling text features. Default: 1
            pad_len: Padding length for the class aggregation. Default: 256
                     pad_len enforces the class aggregation block to have a fixed length of tokens for all inputs
                     This means it either pads the sequence with learnable tokens in class aggregation,
                     or truncates the classes with the initial CLIP cosine-similarity scores.
                     Set pad_len to 0 to disable this feature.
            """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.layers = nn.ModuleList([
            AggregatorLayer(
                hidden_dim=hidden_dim, text_guidance_dim=text_guidance_proj_dim, appearance_guidance=appearance_guidance_proj_dim, 
                nheads=nheads, input_resolution=feature_resolution, pooling_size=pooling_size, window_size=window_size, attention_type=attention_type, pad_len=pad_len,
            ) for _ in range(num_layers)
        ])

        # self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.fusion_corr = nn.Conv2d(2*hidden_dim, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.guidance_projection = nn.Sequential(
            nn.Conv2d(appearance_guidance_dim, appearance_guidance_proj_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        ) if appearance_guidance_dim > 0 else None
        
        self.text_guidance_projection = nn.Sequential(
            nn.Linear(text_guidance_dim, text_guidance_proj_dim),
            nn.ReLU(),
        ) if text_guidance_dim > 0 else None

        self.decoder_guidance_projection = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            ) for d, dp in zip(decoder_guidance_dims, decoder_guidance_proj_dims)
        ]) if decoder_guidance_dims[0] > 0 else None

        self.decoder1 = Up(hidden_dim, decoder_dims[0], decoder_guidance_proj_dims[0])
        self.decoder2 = Up(decoder_dims[0], decoder_dims[1], decoder_guidance_proj_dims[1])
        self.head = nn.Conv2d(decoder_dims[1], 1, kernel_size=3, stride=1, padding=1)

        self.pad_len = pad_len

    def feature_map(self, img_feats, text_feats):
        # concatenated feature volume for feature aggregation baselines
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        img_feats = repeat(img_feats, "B C H W -> B C T H W", T=text_feats.shape[1])
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        text_feats = text_feats.mean(dim=-2) # average text features over different prompts
        text_feats = F.normalize(text_feats, dim=-1) # B T C
        text_feats = repeat(text_feats, "B T C -> B C T H W", H=img_feats.shape[-2], W=img_feats.shape[-1])
        return torch.cat((img_feats, text_feats), dim=1) # B 2C T H W

    def correlation(self, img_feats, text_feats):
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        corr = torch.einsum('bchw, btpc -> bpthw', img_feats, text_feats)
        return corr

    def corr_embed(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B P T H W -> (B T) P H W')
        corr_embed = self.conv1(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed
    
    def corr_fusion_embed_minimum(self,clip_corr,dino_corr):
        # this one does not import a 1*1 conv
        # instead we modify the original embedding layer to adapt to the concatenated corr volume.
        B = clip_corr.shape[0]
        clip_corr = rearrange(clip_corr, 'B P T H W -> (B T) P H W')
        dino_corr = rearrange(dino_corr, 'B P T H W -> (B T) P H W')
        fused_corr = torch.cat([clip_corr,dino_corr],dim = 1)
        fused_corr = self.conv1_modified(fused_corr)
        fused_corr = rearrange(fused_corr, '(B T) C H W -> B C T H W', B=B)
        return fused_corr

    def corr_fusion_embed_seperate(self,clip_corr,dino_corr):
        # this one does not import a 1*1 conv
        # instead we modify the original embedding layer to adapt to the concatenated corr volume.
        B = clip_corr.shape[0]
        clip_corr = rearrange(clip_corr, 'B P T H W -> (B T) P H W')
        dino_corr = rearrange(dino_corr, 'B P T H W -> (B T) P H W')
 
        clip_embed_corr = self.conv1(clip_corr)
        dino_embed_corr = self.conv2(dino_corr)

        fused_corr = torch.cat([clip_embed_corr,dino_embed_corr],dim = 1)
        fused_corr = self.fusion_corr(fused_corr)
        fused_corr = rearrange(fused_corr, '(B T) C H W -> B C T H W', B=B)

        return fused_corr
        
        
    def corr_fusion_embed(self,clip_corr,dino_corr):
        B = clip_corr.shape[0]
        clip_corr = rearrange(clip_corr, 'B P T H W -> (B T) P H W')
        dino_corr = rearrange(dino_corr, 'B P T H W -> (B T) P H W')
        fused_corr = torch.cat([clip_corr,dino_corr],dim = 1)
        fused_corr = self.fusion_corr(fused_corr)
        fused_corr = self.conv1(fused_corr)
        fused_corr = rearrange(fused_corr, '(B T) C H W -> B C T H W', B=B)
        return fused_corr
        # exit()
        
    def corr_projection(self, x, proj):
        corr_embed = rearrange(x, 'B C T H W -> B T H W C')
        corr_embed = proj(corr_embed)
        corr_embed = rearrange(corr_embed, 'B T H W C -> B C T H W')
        return corr_embed

    def upsample(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = F.interpolate(corr_embed, scale_factor=2, mode='bilinear', align_corners=True)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed

    def conv_decoder(self, x, guidance):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = self.decoder1(corr_embed, guidance[0])
        corr_embed = self.decoder2(corr_embed, guidance[1])
        corr_embed = self.head(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
        return corr_embed
    def forward(self, img_feats,dino_feat, text_feats, appearance_guidance):
        """
        Arguments:
            img_feats: (B, C, H, W)
            text_feats: (B, T, P, C) T是类别的个数
            apperance_guidance: tuple of (B, C, H, W)
        """

        classes = None
        ################START Here modified by YCY START###############
        corr = self.correlation(img_feats, text_feats)
        dino_corr = self.correlation(dino_feat,text_feats)
        if self.pad_len > 0 and text_feats.size(1) > self.pad_len:
            avg = corr.permute(0, 2, 1, 3, 4).flatten(-3).max(dim=-1)[0] 
            avg_dino = dino_corr.permute(0, 2, 1, 3, 4).flatten(-3).max(dim=-1)[0] 
            classes = avg.topk(self.pad_len, dim=-1, sorted=False)[1]
            classes_dino = avg_dino.topk(self.pad_len, dim=-1, sorted=False)[1]
            th_text = F.normalize(text_feats, dim=-1)
            clip_th_text = torch.gather(th_text, dim=1, index=classes[..., None, None].expand(-1, -1, th_text.size(-2), th_text.size(-1)))
            dino_th_text = torch.gather(th_text, dim=1, index=classes_dino[..., None, None].expand(-1, -1, th_text.size(-2), th_text.size(-1)))
            orig_clases = text_feats.size(1)
            img_feats = F.normalize(img_feats, dim=1) # B C H W
            dino_feats = F.normalize(dino_feat, dim=1) # B C H W
            corr = torch.einsum('bchw, btpc -> bpthw', img_feats, clip_th_text)
            dino_corr = torch.einsum('bchw, btpc -> bpthw', dino_feats, dino_th_text)
        #corr = self.feature_map(img_feats, text_feats)
        #exit()
        # fused_corr = corr+dino_corr

        # exit()
        
        # fused_corr_embed = self.corr_embed(fused_corr)
        # fused_corr_embed1 = self.corr_fusion_embed_minimum(clip_corr = corr,dino_corr=dino_corr)
        fused_corr_embed = self.corr_fusion_embed_seperate(clip_corr = corr,dino_corr=dino_corr)
        # clip_corr_embed = self.corr_embed(corr)
        # dino_corr_embed = self.corr_embed(dino_corr)
        ################END Here modified by YCY END###############
        
        # print(clip_corr_embed)
        projected_guidance, projected_text_guidance, projected_decoder_guidance = None, None, [None, None]
        if self.guidance_projection is not None:
            projected_guidance = self.guidance_projection(appearance_guidance[0])
        if self.decoder_guidance_projection is not None:
            projected_decoder_guidance = [proj(g) for proj, g in zip(self.decoder_guidance_projection, appearance_guidance[1:])]

        if self.text_guidance_projection is not None:
            text_feats = text_feats.mean(dim=-2)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            projected_text_guidance = self.text_guidance_projection(text_feats)

        for layer in self.layers:
            fused_corr_embed = layer(fused_corr_embed, projected_guidance, projected_text_guidance)
            # clip_corr_embed = layer(clip_corr_embed, projected_guidance, projected_text_guidance)
            # dino_corr_embed = layer(dino_corr_embed, projected_guidance, projected_text_guidance)


        # fusion_corr_embed = clip_corr_embed + dino_corr_embed

        # logit = self.conv_decoder(clip_corr_embed, projected_decoder_guidance)
        logit = self.conv_decoder(fused_corr_embed, projected_decoder_guidance)

        
        if classes is not None:
            out = torch.full((logit.size(0), orig_clases, logit.size(2), logit.size(3)), -100., device=logit.device)
            out.scatter_(dim=1, index=classes[..., None, None].expand(-1, -1, logit.size(-2), logit.size(-1)), src=logit)
            logit = out
        return logit
class FusionAggregatorVer09e(nn.Module):
    def __init__(self, 
        text_guidance_dim=512,
        text_guidance_proj_dim=128,
        appearance_guidance_dim=512,
        appearance_guidance_proj_dim=128,
        decoder_dims = (64, 32),
        decoder_guidance_dims=(256, 128),
        decoder_guidance_proj_dims=(32, 16),
        num_layers=4,
        nheads=4, 
        hidden_dim=128,
        pooling_size=(6, 6),
        feature_resolution=(24, 24),
        window_size=12,
        attention_type='linear',
        prompt_channel=1,
        pad_len=256,
    ) -> None:
        """
        Cost Aggregation Model for CAT-Seg
        Args:
            text_guidance_dim: Dimension of text guidance
            text_guidance_proj_dim: Dimension of projected text guidance
            appearance_guidance_dim: Dimension of appearance guidance
            appearance_guidance_proj_dim: Dimension of projected appearance guidance
            decoder_dims: Upsampling decoder dimensions
            decoder_guidance_dims: Upsampling decoder guidance dimensions
            decoder_guidance_proj_dims: Upsampling decoder guidance projected dimensions
            num_layers: Number of layers for the aggregator
            nheads: Number of attention heads
            hidden_dim: Hidden dimension for transformer blocks
            pooling_size: Pooling size for the class aggregation layer
                          To reduce computation, we apply pooling in class aggregation blocks to reduce the number of tokens during training
            feature_resolution: Feature resolution for spatial aggregation
            window_size: Window size for Swin block in spatial aggregation
            attention_type: Attention type for the class aggregation. 
            prompt_channel: Number of prompts for ensembling text features. Default: 1
            pad_len: Padding length for the class aggregation. Default: 256
                     pad_len enforces the class aggregation block to have a fixed length of tokens for all inputs
                     This means it either pads the sequence with learnable tokens in class aggregation,
                     or truncates the classes with the initial CLIP cosine-similarity scores.
                     Set pad_len to 0 to disable this feature.
            """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.layers = nn.ModuleList([
            AggregatorLayerVer9e(
                hidden_dim=hidden_dim, text_guidance_dim=text_guidance_proj_dim, appearance_guidance=appearance_guidance_proj_dim, 
                nheads=nheads, input_resolution=feature_resolution, pooling_size=pooling_size, window_size=window_size, attention_type=attention_type, pad_len=pad_len,
            ) for _ in range(num_layers)
        ])

        # self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.fusion_corr = nn.Conv2d(2*hidden_dim, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.clip_guidance_projection = nn.Sequential(
            nn.Conv2d(appearance_guidance_dim, appearance_guidance_proj_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        ) if appearance_guidance_dim > 0 else None
        
        self.dino_guidance_projection = nn.Sequential(
            nn.Conv2d(appearance_guidance_dim, appearance_guidance_proj_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        ) if appearance_guidance_dim > 0 else None
        
        self.text_guidance_projection = nn.Sequential(
            nn.Linear(text_guidance_dim, text_guidance_proj_dim),
            nn.ReLU(),
        ) if text_guidance_dim > 0 else None

        self.CLIP_decoder_guidance_projection = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            ) for d, dp in zip(decoder_guidance_dims, decoder_guidance_proj_dims)
        ]) if decoder_guidance_dims[0] > 0 else None
        
        self.DINO_decoder_guidance_projection = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            ) for d, dp in zip(decoder_guidance_dims, decoder_guidance_proj_dims)
        ]) if decoder_guidance_dims[0] > 0 else None
        
        # self.decoder1 = Up(hidden_dim, decoder_dims[0], decoder_guidance_proj_dims[0])
        # self.decoder2 = Up(decoder_dims[0], decoder_dims[1], decoder_guidance_proj_dims[1])
        
        self.Fusiondecoder1=FusionUP(hidden_dim, decoder_dims[0], decoder_guidance_proj_dims[0])
        self.Fusiondecoder2=FusionUP(decoder_dims[0], decoder_dims[1], decoder_guidance_proj_dims[1])
        self.head = nn.Conv2d(decoder_dims[1], 1, kernel_size=3, stride=1, padding=1)

        self.pad_len = pad_len

    def feature_map(self, img_feats, text_feats):
        # concatenated feature volume for feature aggregation baselines
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        img_feats = repeat(img_feats, "B C H W -> B C T H W", T=text_feats.shape[1])
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        text_feats = text_feats.mean(dim=-2) # average text features over different prompts
        text_feats = F.normalize(text_feats, dim=-1) # B T C
        text_feats = repeat(text_feats, "B T C -> B C T H W", H=img_feats.shape[-2], W=img_feats.shape[-1])
        return torch.cat((img_feats, text_feats), dim=1) # B 2C T H W

    def correlation(self, img_feats, text_feats):
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        corr = torch.einsum('bchw, btpc -> bpthw', img_feats, text_feats)
        return corr

    def corr_embed(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B P T H W -> (B T) P H W')
        corr_embed = self.conv1(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed
    
    def corr_fusion_embed_minimum(self,clip_corr,dino_corr):
        # this one does not import a 1*1 conv
        # instead we modify the original embedding layer to adapt to the concatenated corr volume.
        B = clip_corr.shape[0]
        clip_corr = rearrange(clip_corr, 'B P T H W -> (B T) P H W')
        dino_corr = rearrange(dino_corr, 'B P T H W -> (B T) P H W')
        fused_corr = torch.cat([clip_corr,dino_corr],dim = 1)
        fused_corr = self.conv1_modified(fused_corr)
        fused_corr = rearrange(fused_corr, '(B T) C H W -> B C T H W', B=B)
        return fused_corr

    def corr_fusion_embed_seperate(self,clip_corr,dino_corr):
        # this one does not import a 1*1 conv
        # instead we modify the original embedding layer to adapt to the concatenated corr volume.
        B = clip_corr.shape[0]
        self.sigmoid = nn.Sigmoid()
        clip_corr = rearrange(clip_corr, 'B P T H W -> (B T) P H W')
        dino_corr = rearrange(dino_corr, 'B P T H W -> (B T) P H W')
 
        clip_embed_corr = self.conv1(clip_corr)
        dino_embed_corr = self.conv2(dino_corr)
        clip_embed_corr = self.sigmoid(clip_embed_corr)
        dino_embed_corr = self.sigmoid(dino_embed_corr)
        fused_corr = torch.cat([clip_embed_corr,dino_embed_corr],dim = 1)
        fused_corr = self.fusion_corr(fused_corr)
        fused_corr = self.sigmoid(fused_corr)
        fused_corr = rearrange(fused_corr, '(B T) C H W -> B C T H W', B=B)
        clip_embed_corr = rearrange(clip_embed_corr, '(B T) C H W -> B C T H W', B=B)
        dino_embed_corr = rearrange(dino_embed_corr, '(B T) C H W -> B C T H W', B=B)
        return fused_corr, clip_embed_corr, dino_embed_corr
        
        
    def corr_fusion_embed(self,clip_corr,dino_corr):
        B = clip_corr.shape[0]
        clip_corr = rearrange(clip_corr, 'B P T H W -> (B T) P H W')
        dino_corr = rearrange(dino_corr, 'B P T H W -> (B T) P H W')
        fused_corr = torch.cat([clip_corr,dino_corr],dim = 1)
        fused_corr = self.fusion_corr(fused_corr)
        fused_corr = self.conv1(fused_corr)
        fused_corr = rearrange(fused_corr, '(B T) C H W -> B C T H W', B=B)
        return fused_corr
        # exit()
        
    def corr_projection(self, x, proj):
        corr_embed = rearrange(x, 'B C T H W -> B T H W C')
        corr_embed = proj(corr_embed)
        corr_embed = rearrange(corr_embed, 'B T H W C -> B C T H W')
        return corr_embed

    def upsample(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = F.interpolate(corr_embed, scale_factor=2, mode='bilinear', align_corners=True)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed

    def conv_decoder(self, x, guidance):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = self.decoder1(corr_embed, guidance[0])
        corr_embed = self.decoder2(corr_embed, guidance[1])
        corr_embed = self.head(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
        return corr_embed
    
    def Fusion_conv_decoer(self, x, clip_guidance,dino_guidance):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = self.Fusiondecoder1(corr_embed, clip_guidance[0],dino_guidance[0])
        corr_embed = self.Fusiondecoder2(corr_embed, clip_guidance[1],dino_guidance[1])
        corr_embed = self.head(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
        return corr_embed
    def forward(self, img_feats,dino_feat, text_feats, appearance_guidance,dino_guidance):
        """
        Arguments:
            img_feats: (B, C, H, W)
            text_feats: (B, T, P, C) T是类别的个数
            apperance_guidance: tuple of (B, C, H, W)
        """

        classes = None
        ################START Here modified by YCY START###############
        corr = self.correlation(img_feats, text_feats)
        dino_corr = self.correlation(dino_feat,text_feats)
        if self.pad_len > 0 and text_feats.size(1) > self.pad_len:
            avg = corr.permute(0, 2, 1, 3, 4).flatten(-3).max(dim=-1)[0] 
            avg_dino = dino_corr.permute(0, 2, 1, 3, 4).flatten(-3).max(dim=-1)[0] 
            classes = avg.topk(self.pad_len, dim=-1, sorted=False)[1]
            classes_dino = avg_dino.topk(self.pad_len, dim=-1, sorted=False)[1]
            th_text = F.normalize(text_feats, dim=-1)
            clip_th_text = torch.gather(th_text, dim=1, index=classes[..., None, None].expand(-1, -1, th_text.size(-2), th_text.size(-1)))
            dino_th_text = torch.gather(th_text, dim=1, index=classes_dino[..., None, None].expand(-1, -1, th_text.size(-2), th_text.size(-1)))
            orig_clases = text_feats.size(1)
            img_feats = F.normalize(img_feats, dim=1) # B C H W
            dino_feats = F.normalize(dino_feat, dim=1) # B C H W
            corr = torch.einsum('bchw, btpc -> bpthw', img_feats, clip_th_text)
            dino_corr = torch.einsum('bchw, btpc -> bpthw', dino_feats, dino_th_text)
        #corr = self.feature_map(img_feats, text_feats)
        #exit()
        # fused_corr = corr+dino_corr

        # exit()
        
        # fused_corr_embed = self.corr_embed(fused_corr)
        # fused_corr_embed1 = self.corr_fusion_embed_minimum(clip_corr = corr,dino_corr=dino_corr)
        fused_corr_embed,clip_embed_corr, dino_embed_corr  = self.corr_fusion_embed_seperate(clip_corr = corr,dino_corr=dino_corr)
        # add the res here #

        fused_corr_embed = fused_corr_embed+clip_embed_corr

        # add the res here #
        # print(23333333333)
        # clip_corr_embed = self.corr_embed(corr)
        # dino_corr_embed = self.corr_embed(dino_corr)
        ################END Here modified by YCY END###############
        
        # print(clip_corr_embed)
        clip_projected_guidance, projected_text_guidance, CLIP_projected_decoder_guidance,DINO_projected_decoder_guidance  = None, None, [None, None], [None,None]
        dino_projected_guidance = None
        if self.clip_guidance_projection is not None:
            clip_projected_guidance = self.clip_guidance_projection(appearance_guidance[0])
        if self.dino_guidance_projection is not None:
            dino_projected_guidance = self.dino_guidance_projection(dino_feat)
        if self.CLIP_decoder_guidance_projection is not None:
            CLIP_projected_decoder_guidance = [proj(g) for proj, g in zip(self.CLIP_decoder_guidance_projection, appearance_guidance[1:])]
        if self.DINO_decoder_guidance_projection is not None:
            DINO_projected_decoder_guidance = [proj(g) for proj, g in zip(self.DINO_decoder_guidance_projection, dino_guidance)]
        if self.text_guidance_projection is not None:
            text_feats = text_feats.mean(dim=-2)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            projected_text_guidance = self.text_guidance_projection(text_feats)

        for layer in self.layers:
            fused_corr_embed = layer(fused_corr_embed, clip_projected_guidance, dino_projected_guidance, projected_text_guidance)
            # clip_corr_embed = layer(clip_corr_embed, projected_guidance, projected_text_guidance)
            # dino_corr_embed = layer(dino_corr_embed, projected_guidance, projected_text_guidance)
        

        # fusion_corr_embed = clip_corr_embed + dino_corr_embed

        # logit = self.conv_decoder(clip_corr_embed, projected_decoder_guidance)

        logit = self.Fusion_conv_decoer(fused_corr_embed, CLIP_projected_decoder_guidance,DINO_projected_decoder_guidance)

        
        if classes is not None:
            out = torch.full((logit.size(0), orig_clases, logit.size(2), logit.size(3)), -100., device=logit.device)
            out.scatter_(dim=1, index=classes[..., None, None].expand(-1, -1, logit.size(-2), logit.size(-1)), src=logit)
            logit = out
        return logit

class FusionAggregatorVer09d(nn.Module):
    def __init__(self, 
        text_guidance_dim=512,
        text_guidance_proj_dim=128,
        appearance_guidance_dim=512,
        appearance_guidance_proj_dim=128,
        decoder_dims = (64, 32),
        decoder_guidance_dims=(256, 128),
        decoder_guidance_proj_dims=(32, 16),
        num_layers=4,
        nheads=4, 
        hidden_dim=128,
        pooling_size=(6, 6),
        feature_resolution=(24, 24),
        window_size=12,
        attention_type='linear',
        prompt_channel=1,
        pad_len=256,
    ) -> None:
        """
        Cost Aggregation Model for CAT-Seg
        Args:
            text_guidance_dim: Dimension of text guidance
            text_guidance_proj_dim: Dimension of projected text guidance
            appearance_guidance_dim: Dimension of appearance guidance
            appearance_guidance_proj_dim: Dimension of projected appearance guidance
            decoder_dims: Upsampling decoder dimensions
            decoder_guidance_dims: Upsampling decoder guidance dimensions
            decoder_guidance_proj_dims: Upsampling decoder guidance projected dimensions
            num_layers: Number of layers for the aggregator
            nheads: Number of attention heads
            hidden_dim: Hidden dimension for transformer blocks
            pooling_size: Pooling size for the class aggregation layer
                          To reduce computation, we apply pooling in class aggregation blocks to reduce the number of tokens during training
            feature_resolution: Feature resolution for spatial aggregation
            window_size: Window size for Swin block in spatial aggregation
            attention_type: Attention type for the class aggregation. 
            prompt_channel: Number of prompts for ensembling text features. Default: 1
            pad_len: Padding length for the class aggregation. Default: 256
                     pad_len enforces the class aggregation block to have a fixed length of tokens for all inputs
                     This means it either pads the sequence with learnable tokens in class aggregation,
                     or truncates the classes with the initial CLIP cosine-similarity scores.
                     Set pad_len to 0 to disable this feature.
            """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.layers = nn.ModuleList([
            AggregatorLayer(
                hidden_dim=hidden_dim, text_guidance_dim=text_guidance_proj_dim, appearance_guidance=appearance_guidance_proj_dim, 
                nheads=nheads, input_resolution=feature_resolution, pooling_size=pooling_size, window_size=window_size, attention_type=attention_type, pad_len=pad_len,
            ) for _ in range(num_layers)
        ])

        # self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.fusion_corr = nn.Conv2d(2*hidden_dim, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.guidance_projection = nn.Sequential(
            nn.Conv2d(appearance_guidance_dim, appearance_guidance_proj_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        ) if appearance_guidance_dim > 0 else None
        
        self.text_guidance_projection = nn.Sequential(
            nn.Linear(text_guidance_dim, text_guidance_proj_dim),
            nn.ReLU(),
        ) if text_guidance_dim > 0 else None

        self.CLIP_decoder_guidance_projection = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            ) for d, dp in zip(decoder_guidance_dims, decoder_guidance_proj_dims)
        ]) if decoder_guidance_dims[0] > 0 else None
        
        self.DINO_decoder_guidance_projection = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            ) for d, dp in zip(decoder_guidance_dims, decoder_guidance_proj_dims)
        ]) if decoder_guidance_dims[0] > 0 else None
        
        self.decoder1 = Up(hidden_dim, decoder_dims[0], decoder_guidance_proj_dims[0])
        self.decoder2 = Up(decoder_dims[0], decoder_dims[1], decoder_guidance_proj_dims[1])
        
        self.Fusiondecoder1=FusionUP(hidden_dim, decoder_dims[0], decoder_guidance_proj_dims[0])
        self.Fusiondecoder2=FusionUP(decoder_dims[0], decoder_dims[1], decoder_guidance_proj_dims[1])
        self.head = nn.Conv2d(decoder_dims[1], 1, kernel_size=3, stride=1, padding=1)

        self.pad_len = pad_len

    def feature_map(self, img_feats, text_feats):
        # concatenated feature volume for feature aggregation baselines
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        img_feats = repeat(img_feats, "B C H W -> B C T H W", T=text_feats.shape[1])
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        text_feats = text_feats.mean(dim=-2) # average text features over different prompts
        text_feats = F.normalize(text_feats, dim=-1) # B T C
        text_feats = repeat(text_feats, "B T C -> B C T H W", H=img_feats.shape[-2], W=img_feats.shape[-1])
        return torch.cat((img_feats, text_feats), dim=1) # B 2C T H W

    def correlation(self, img_feats, text_feats):
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        corr = torch.einsum('bchw, btpc -> bpthw', img_feats, text_feats)
        return corr

    def corr_embed(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B P T H W -> (B T) P H W')
        corr_embed = self.conv1(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed
    
    def corr_fusion_embed_minimum(self,clip_corr,dino_corr):
        # this one does not import a 1*1 conv
        # instead we modify the original embedding layer to adapt to the concatenated corr volume.
        B = clip_corr.shape[0]
        clip_corr = rearrange(clip_corr, 'B P T H W -> (B T) P H W')
        dino_corr = rearrange(dino_corr, 'B P T H W -> (B T) P H W')
        fused_corr = torch.cat([clip_corr,dino_corr],dim = 1)
        fused_corr = self.conv1_modified(fused_corr)
        fused_corr = rearrange(fused_corr, '(B T) C H W -> B C T H W', B=B)
        return fused_corr

    def corr_fusion_embed_seperate(self,clip_corr,dino_corr):
        # this one does not import a 1*1 conv
        # instead we modify the original embedding layer to adapt to the concatenated corr volume.
        B = clip_corr.shape[0]
        clip_corr = rearrange(clip_corr, 'B P T H W -> (B T) P H W')
        dino_corr = rearrange(dino_corr, 'B P T H W -> (B T) P H W')
 
        clip_embed_corr = self.conv1(clip_corr)
        dino_embed_corr = self.conv2(dino_corr)

        fused_corr = torch.cat([clip_embed_corr,dino_embed_corr],dim = 1)
        fused_corr = self.fusion_corr(fused_corr)
        fused_corr = rearrange(fused_corr, '(B T) C H W -> B C T H W', B=B)
        clip_embed_corr = rearrange(clip_embed_corr, '(B T) C H W -> B C T H W', B=B)
        dino_embed_corr = rearrange(dino_embed_corr, '(B T) C H W -> B C T H W', B=B)
        return fused_corr, clip_embed_corr, dino_embed_corr
        
        
    def corr_fusion_embed(self,clip_corr,dino_corr):
        B = clip_corr.shape[0]
        clip_corr = rearrange(clip_corr, 'B P T H W -> (B T) P H W')
        dino_corr = rearrange(dino_corr, 'B P T H W -> (B T) P H W')
        fused_corr = torch.cat([clip_corr,dino_corr],dim = 1)
        fused_corr = self.fusion_corr(fused_corr)
        fused_corr = self.conv1(fused_corr)
        fused_corr = rearrange(fused_corr, '(B T) C H W -> B C T H W', B=B)
        return fused_corr
        # exit()
        
    def corr_projection(self, x, proj):
        corr_embed = rearrange(x, 'B C T H W -> B T H W C')
        corr_embed = proj(corr_embed)
        corr_embed = rearrange(corr_embed, 'B T H W C -> B C T H W')
        return corr_embed

    def upsample(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = F.interpolate(corr_embed, scale_factor=2, mode='bilinear', align_corners=True)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed

    def conv_decoder(self, x, guidance):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = self.decoder1(corr_embed, guidance[0])
        corr_embed = self.decoder2(corr_embed, guidance[1])
        corr_embed = self.head(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
        return corr_embed
    
    def corr_fusion_embed_summation_res(self,clip_corr,dino_corr):
        B = clip_corr.shape[0]
        self.sigmoid = nn.Sigmoid()
        clip_corr = rearrange(clip_corr, 'B P T H W -> (B T) P H W')
        dino_corr = rearrange(dino_corr, 'B P T H W -> (B T) P H W')

        clip_embed_corr = self.conv1(clip_corr)
        dino_embed_corr = self.conv2(dino_corr)
        
        #   last Modification here  #
        clip_embed_corr = self.sigmoid(clip_embed_corr)

        dino_embed_corr = self.sigmoid(dino_embed_corr)
        #   last Modification here  #
        fused_corr = clip_embed_corr + dino_embed_corr
        # fused_corr = self.fusion_corr(fused_corr)
        fused_corr = fused_corr + clip_embed_corr
        fused_corr = rearrange(fused_corr, '(B T) C H W -> B C T H W', B=B)
        # print('success!!')
        return fused_corr
    def Fusion_conv_decoer(self, x, clip_guidance,dino_guidance):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = self.Fusiondecoder1(corr_embed, clip_guidance[0],dino_guidance[0])
        corr_embed = self.Fusiondecoder2(corr_embed, clip_guidance[1],dino_guidance[1])
        corr_embed = self.head(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
        return corr_embed
    def forward(self, img_feats,dino_feat, text_feats, appearance_guidance,dino_guidance):
        """
        Arguments:
            img_feats: (B, C, H, W)
            text_feats: (B, T, P, C) T是类别的个数
            apperance_guidance: tuple of (B, C, H, W)
        """

        classes = None
        ################START Here modified by YCY START###############
        corr = self.correlation(img_feats, text_feats)
        dino_corr = self.correlation(dino_feat,text_feats)
        if self.pad_len > 0 and text_feats.size(1) > self.pad_len:
            avg = corr.permute(0, 2, 1, 3, 4).flatten(-3).max(dim=-1)[0] 
            avg_dino = dino_corr.permute(0, 2, 1, 3, 4).flatten(-3).max(dim=-1)[0] 
            classes = avg.topk(self.pad_len, dim=-1, sorted=False)[1]
            classes_dino = avg_dino.topk(self.pad_len, dim=-1, sorted=False)[1]
            th_text = F.normalize(text_feats, dim=-1)
            clip_th_text = torch.gather(th_text, dim=1, index=classes[..., None, None].expand(-1, -1, th_text.size(-2), th_text.size(-1)))
            dino_th_text = torch.gather(th_text, dim=1, index=classes_dino[..., None, None].expand(-1, -1, th_text.size(-2), th_text.size(-1)))
            orig_clases = text_feats.size(1)
            img_feats = F.normalize(img_feats, dim=1) # B C H W
            dino_feats = F.normalize(dino_feat, dim=1) # B C H W
            corr = torch.einsum('bchw, btpc -> bpthw', img_feats, clip_th_text)
            dino_corr = torch.einsum('bchw, btpc -> bpthw', dino_feats, dino_th_text)
        #corr = self.feature_map(img_feats, text_feats)
        #exit()
        # fused_corr = corr+dino_corr

        # exit()
        
        # fused_corr_embed = self.corr_embed(fused_corr)
        # fused_corr_embed1 = self.corr_fusion_embed_minimum(clip_corr = corr,dino_corr=dino_corr)
        # fused_corr_embed,clip_embed_corr, dino_embed_corr  = self.corr_fusion_embed_seperate(clip_corr = corr,dino_corr=dino_corr)
        # add the res here #
        fused_corr_embed = self.corr_fusion_embed_summation_res(clip_corr = corr,dino_corr=dino_corr)
        # fused_corr_embed = fused_corr_embed+clip_embed_corr

        # add the res here #
        # print(23333333333)
        # clip_corr_embed = self.corr_embed(corr)
        # dino_corr_embed = self.corr_embed(dino_corr)
        ################END Here modified by YCY END###############
        
        # print(clip_corr_embed)
        projected_guidance, projected_text_guidance, CLIP_projected_decoder_guidance,DINO_projected_decoder_guidance  = None, None, [None, None], [None,None]
        if self.guidance_projection is not None:
            projected_guidance = self.guidance_projection(appearance_guidance[0])
        if self.CLIP_decoder_guidance_projection is not None:
            CLIP_projected_decoder_guidance = [proj(g) for proj, g in zip(self.CLIP_decoder_guidance_projection, appearance_guidance[1:])]
        if self.DINO_decoder_guidance_projection is not None:
            DINO_projected_decoder_guidance = [proj(g) for proj, g in zip(self.DINO_decoder_guidance_projection, dino_guidance)]
        if self.text_guidance_projection is not None:
            text_feats = text_feats.mean(dim=-2)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            projected_text_guidance = self.text_guidance_projection(text_feats)

        for layer in self.layers:
            fused_corr_embed = layer(fused_corr_embed, projected_guidance, projected_text_guidance)
            # clip_corr_embed = layer(clip_corr_embed, projected_guidance, projected_text_guidance)
            # dino_corr_embed = layer(dino_corr_embed, projected_guidance, projected_text_guidance)


        # fusion_corr_embed = clip_corr_embed + dino_corr_embed

        # logit = self.conv_decoder(clip_corr_embed, projected_decoder_guidance)

        logit = self.Fusion_conv_decoer(fused_corr_embed, CLIP_projected_decoder_guidance,DINO_projected_decoder_guidance)

        
        if classes is not None:
            out = torch.full((logit.size(0), orig_clases, logit.size(2), logit.size(3)), -100., device=logit.device)
            out.scatter_(dim=1, index=classes[..., None, None].expand(-1, -1, logit.size(-2), logit.size(-1)), src=logit)
            logit = out
        return logit

class FusionAggregatorVer09c(nn.Module):
    def __init__(self, 
        text_guidance_dim=512,
        text_guidance_proj_dim=128,
        appearance_guidance_dim=512,
        appearance_guidance_proj_dim=128,
        decoder_dims = (64, 32),
        decoder_guidance_dims=(256, 128),
        decoder_guidance_proj_dims=(32, 16),
        num_layers=4,
        nheads=4, 
        hidden_dim=128,
        pooling_size=(6, 6),
        feature_resolution=(24, 24),
        window_size=12,
        attention_type='linear',
        prompt_channel=1,
        pad_len=256,
    ) -> None:
        """
        Cost Aggregation Model for CAT-Seg
        Args:
            text_guidance_dim: Dimension of text guidance
            text_guidance_proj_dim: Dimension of projected text guidance
            appearance_guidance_dim: Dimension of appearance guidance
            appearance_guidance_proj_dim: Dimension of projected appearance guidance
            decoder_dims: Upsampling decoder dimensions
            decoder_guidance_dims: Upsampling decoder guidance dimensions
            decoder_guidance_proj_dims: Upsampling decoder guidance projected dimensions
            num_layers: Number of layers for the aggregator
            nheads: Number of attention heads
            hidden_dim: Hidden dimension for transformer blocks
            pooling_size: Pooling size for the class aggregation layer
                          To reduce computation, we apply pooling in class aggregation blocks to reduce the number of tokens during training
            feature_resolution: Feature resolution for spatial aggregation
            window_size: Window size for Swin block in spatial aggregation
            attention_type: Attention type for the class aggregation. 
            prompt_channel: Number of prompts for ensembling text features. Default: 1
            pad_len: Padding length for the class aggregation. Default: 256
                     pad_len enforces the class aggregation block to have a fixed length of tokens for all inputs
                     This means it either pads the sequence with learnable tokens in class aggregation,
                     or truncates the classes with the initial CLIP cosine-similarity scores.
                     Set pad_len to 0 to disable this feature.
            """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.layers = nn.ModuleList([
            AggregatorLayer(
                hidden_dim=hidden_dim, text_guidance_dim=text_guidance_proj_dim, appearance_guidance=appearance_guidance_proj_dim, 
                nheads=nheads, input_resolution=feature_resolution, pooling_size=pooling_size, window_size=window_size, attention_type=attention_type, pad_len=pad_len,
            ) for _ in range(num_layers)
        ])

        # self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.fusion_corr = nn.Conv2d(2*hidden_dim, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.guidance_projection = nn.Sequential(
            nn.Conv2d(appearance_guidance_dim, appearance_guidance_proj_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        ) if appearance_guidance_dim > 0 else None
        
        self.text_guidance_projection = nn.Sequential(
            nn.Linear(text_guidance_dim, text_guidance_proj_dim),
            nn.ReLU(),
        ) if text_guidance_dim > 0 else None

        self.CLIP_decoder_guidance_projection = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            ) for d, dp in zip(decoder_guidance_dims, decoder_guidance_proj_dims)
        ]) if decoder_guidance_dims[0] > 0 else None
        
        self.DINO_decoder_guidance_projection = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            ) for d, dp in zip(decoder_guidance_dims, decoder_guidance_proj_dims)
        ]) if decoder_guidance_dims[0] > 0 else None
        
        # self.decoder1 = Up(hidden_dim, decoder_dims[0], decoder_guidance_proj_dims[0])
        # self.decoder2 = Up(decoder_dims[0], decoder_dims[1], decoder_guidance_proj_dims[1])
        
        self.Fusiondecoder1=FusionUP(hidden_dim, decoder_dims[0], decoder_guidance_proj_dims[0])
        self.Fusiondecoder2=FusionUP(decoder_dims[0], decoder_dims[1], decoder_guidance_proj_dims[1])
        self.head = nn.Conv2d(decoder_dims[1], 1, kernel_size=3, stride=1, padding=1)

        self.pad_len = pad_len

    def feature_map(self, img_feats, text_feats):
        # concatenated feature volume for feature aggregation baselines
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        img_feats = repeat(img_feats, "B C H W -> B C T H W", T=text_feats.shape[1])
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        text_feats = text_feats.mean(dim=-2) # average text features over different prompts
        text_feats = F.normalize(text_feats, dim=-1) # B T C
        text_feats = repeat(text_feats, "B T C -> B C T H W", H=img_feats.shape[-2], W=img_feats.shape[-1])
        return torch.cat((img_feats, text_feats), dim=1) # B 2C T H W

    def correlation(self, img_feats, text_feats):
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        corr = torch.einsum('bchw, btpc -> bpthw', img_feats, text_feats)
        return corr

    def corr_embed(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B P T H W -> (B T) P H W')
        corr_embed = self.conv1(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed
    
    def corr_fusion_embed_minimum(self,clip_corr,dino_corr):
        # this one does not import a 1*1 conv
        # instead we modify the original embedding layer to adapt to the concatenated corr volume.
        B = clip_corr.shape[0]
        clip_corr = rearrange(clip_corr, 'B P T H W -> (B T) P H W')
        dino_corr = rearrange(dino_corr, 'B P T H W -> (B T) P H W')
        fused_corr = torch.cat([clip_corr,dino_corr],dim = 1)
        fused_corr = self.conv1_modified(fused_corr)
        fused_corr = rearrange(fused_corr, '(B T) C H W -> B C T H W', B=B)
        return fused_corr

    def corr_fusion_embed_seperate(self,clip_corr,dino_corr):
        # this one does not import a 1*1 conv
        # instead we modify the original embedding layer to adapt to the concatenated corr volume.
        B = clip_corr.shape[0]
        self.sigmoid = nn.Sigmoid()
        clip_corr = rearrange(clip_corr, 'B P T H W -> (B T) P H W')
        dino_corr = rearrange(dino_corr, 'B P T H W -> (B T) P H W')
 
        clip_embed_corr = self.conv1(clip_corr)
        dino_embed_corr = self.conv2(dino_corr)
        clip_embed_corr = self.sigmoid(clip_embed_corr)
        dino_embed_corr = self.sigmoid(dino_embed_corr)
        fused_corr = torch.cat([clip_embed_corr,dino_embed_corr],dim = 1)
        fused_corr = self.fusion_corr(fused_corr)
        fused_corr = self.sigmoid(fused_corr)
        fused_corr = rearrange(fused_corr, '(B T) C H W -> B C T H W', B=B)
        clip_embed_corr = rearrange(clip_embed_corr, '(B T) C H W -> B C T H W', B=B)
        dino_embed_corr = rearrange(dino_embed_corr, '(B T) C H W -> B C T H W', B=B)
        return fused_corr, clip_embed_corr, dino_embed_corr
        
        
    def corr_fusion_embed(self,clip_corr,dino_corr):
        B = clip_corr.shape[0]
        clip_corr = rearrange(clip_corr, 'B P T H W -> (B T) P H W')
        dino_corr = rearrange(dino_corr, 'B P T H W -> (B T) P H W')
        fused_corr = torch.cat([clip_corr,dino_corr],dim = 1)
        fused_corr = self.fusion_corr(fused_corr)
        fused_corr = self.conv1(fused_corr)
        fused_corr = rearrange(fused_corr, '(B T) C H W -> B C T H W', B=B)
        return fused_corr
        # exit()
        
    def corr_projection(self, x, proj):
        corr_embed = rearrange(x, 'B C T H W -> B T H W C')
        corr_embed = proj(corr_embed)
        corr_embed = rearrange(corr_embed, 'B T H W C -> B C T H W')
        return corr_embed

    def upsample(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = F.interpolate(corr_embed, scale_factor=2, mode='bilinear', align_corners=True)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed

    def conv_decoder(self, x, guidance):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = self.decoder1(corr_embed, guidance[0])
        corr_embed = self.decoder2(corr_embed, guidance[1])
        corr_embed = self.head(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
        return corr_embed
    
    def Fusion_conv_decoer(self, x, clip_guidance,dino_guidance):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = self.Fusiondecoder1(corr_embed, clip_guidance[0],dino_guidance[0])
        corr_embed = self.Fusiondecoder2(corr_embed, clip_guidance[1],dino_guidance[1])
        corr_embed = self.head(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
        return corr_embed
    def forward(self, img_feats,dino_feat, text_feats, appearance_guidance,dino_guidance):
        """
        Arguments:
            img_feats: (B, C, H, W)
            text_feats: (B, T, P, C) T是类别的个数
            apperance_guidance: tuple of (B, C, H, W)
        """

        classes = None
        ################START Here modified by YCY START###############
        corr = self.correlation(img_feats, text_feats)
        dino_corr = self.correlation(dino_feat,text_feats)
        if self.pad_len > 0 and text_feats.size(1) > self.pad_len:
            avg = corr.permute(0, 2, 1, 3, 4).flatten(-3).max(dim=-1)[0] 
            avg_dino = dino_corr.permute(0, 2, 1, 3, 4).flatten(-3).max(dim=-1)[0] 
            classes = avg.topk(self.pad_len, dim=-1, sorted=False)[1]
            classes_dino = avg_dino.topk(self.pad_len, dim=-1, sorted=False)[1]
            th_text = F.normalize(text_feats, dim=-1)
            clip_th_text = torch.gather(th_text, dim=1, index=classes[..., None, None].expand(-1, -1, th_text.size(-2), th_text.size(-1)))
            dino_th_text = torch.gather(th_text, dim=1, index=classes_dino[..., None, None].expand(-1, -1, th_text.size(-2), th_text.size(-1)))
            orig_clases = text_feats.size(1)
            img_feats = F.normalize(img_feats, dim=1) # B C H W
            dino_feats = F.normalize(dino_feat, dim=1) # B C H W
            corr = torch.einsum('bchw, btpc -> bpthw', img_feats, clip_th_text)
            dino_corr = torch.einsum('bchw, btpc -> bpthw', dino_feats, dino_th_text)
        #corr = self.feature_map(img_feats, text_feats)
        #exit()
        # fused_corr = corr+dino_corr

        # exit()
        
        # fused_corr_embed = self.corr_embed(fused_corr)
        # fused_corr_embed1 = self.corr_fusion_embed_minimum(clip_corr = corr,dino_corr=dino_corr)
        fused_corr_embed,clip_embed_corr, dino_embed_corr  = self.corr_fusion_embed_seperate(clip_corr = corr,dino_corr=dino_corr)
        # add the res here #

        fused_corr_embed = fused_corr_embed+clip_embed_corr

        # add the res here #
        # print(23333333333)
        # clip_corr_embed = self.corr_embed(corr)
        # dino_corr_embed = self.corr_embed(dino_corr)
        ################END Here modified by YCY END###############
        
        # print(clip_corr_embed)
        projected_guidance, projected_text_guidance, CLIP_projected_decoder_guidance,DINO_projected_decoder_guidance  = None, None, [None, None], [None,None]
        if self.guidance_projection is not None:
            projected_guidance = self.guidance_projection(appearance_guidance[0])
        if self.CLIP_decoder_guidance_projection is not None:
            CLIP_projected_decoder_guidance = [proj(g) for proj, g in zip(self.CLIP_decoder_guidance_projection, appearance_guidance[1:])]
        if self.DINO_decoder_guidance_projection is not None:
            DINO_projected_decoder_guidance = [proj(g) for proj, g in zip(self.DINO_decoder_guidance_projection, dino_guidance)]
        if self.text_guidance_projection is not None:
            text_feats = text_feats.mean(dim=-2)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            projected_text_guidance = self.text_guidance_projection(text_feats)

        for layer in self.layers:
            fused_corr_embed = layer(fused_corr_embed, projected_guidance, projected_text_guidance)
            # clip_corr_embed = layer(clip_corr_embed, projected_guidance, projected_text_guidance)
            # dino_corr_embed = layer(dino_corr_embed, projected_guidance, projected_text_guidance)


        # fusion_corr_embed = clip_corr_embed + dino_corr_embed

        # logit = self.conv_decoder(clip_corr_embed, projected_decoder_guidance)

        logit = self.Fusion_conv_decoer(fused_corr_embed, CLIP_projected_decoder_guidance,DINO_projected_decoder_guidance)

        
        if classes is not None:
            out = torch.full((logit.size(0), orig_clases, logit.size(2), logit.size(3)), -100., device=logit.device)
            out.scatter_(dim=1, index=classes[..., None, None].expand(-1, -1, logit.size(-2), logit.size(-1)), src=logit)
            logit = out
        return logit

class FusionAggregatorVer09b(nn.Module):
    def __init__(self, 
        text_guidance_dim=512,
        text_guidance_proj_dim=128,
        appearance_guidance_dim=512,
        appearance_guidance_proj_dim=128,
        decoder_dims = (64, 32),
        decoder_guidance_dims=(256, 128),
        decoder_guidance_proj_dims=(32, 16),
        num_layers=4,
        nheads=4, 
        hidden_dim=128,
        pooling_size=(6, 6),
        feature_resolution=(24, 24),
        window_size=12,
        attention_type='linear',
        prompt_channel=1,
        pad_len=256,
    ) -> None:
        """
        Cost Aggregation Model for CAT-Seg
        Args:
            text_guidance_dim: Dimension of text guidance
            text_guidance_proj_dim: Dimension of projected text guidance
            appearance_guidance_dim: Dimension of appearance guidance
            appearance_guidance_proj_dim: Dimension of projected appearance guidance
            decoder_dims: Upsampling decoder dimensions
            decoder_guidance_dims: Upsampling decoder guidance dimensions
            decoder_guidance_proj_dims: Upsampling decoder guidance projected dimensions
            num_layers: Number of layers for the aggregator
            nheads: Number of attention heads
            hidden_dim: Hidden dimension for transformer blocks
            pooling_size: Pooling size for the class aggregation layer
                          To reduce computation, we apply pooling in class aggregation blocks to reduce the number of tokens during training
            feature_resolution: Feature resolution for spatial aggregation
            window_size: Window size for Swin block in spatial aggregation
            attention_type: Attention type for the class aggregation. 
            prompt_channel: Number of prompts for ensembling text features. Default: 1
            pad_len: Padding length for the class aggregation. Default: 256
                     pad_len enforces the class aggregation block to have a fixed length of tokens for all inputs
                     This means it either pads the sequence with learnable tokens in class aggregation,
                     or truncates the classes with the initial CLIP cosine-similarity scores.
                     Set pad_len to 0 to disable this feature.
            """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.layers = nn.ModuleList([
            AggregatorLayer(
                hidden_dim=hidden_dim, text_guidance_dim=text_guidance_proj_dim, appearance_guidance=appearance_guidance_proj_dim, 
                nheads=nheads, input_resolution=feature_resolution, pooling_size=pooling_size, window_size=window_size, attention_type=attention_type, pad_len=pad_len,
            ) for _ in range(num_layers)
        ])

        # self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)
        # self.fusion_corr = nn.Conv2d(2*hidden_dim, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.guidance_projection = nn.Sequential(
            nn.Conv2d(appearance_guidance_dim, appearance_guidance_proj_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        ) if appearance_guidance_dim > 0 else None
        
        self.text_guidance_projection = nn.Sequential(
            nn.Linear(text_guidance_dim, text_guidance_proj_dim),
            nn.ReLU(),
        ) if text_guidance_dim > 0 else None

        self.decoder_guidance_projection = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            ) for d, dp in zip(decoder_guidance_dims, decoder_guidance_proj_dims)
        ]) if decoder_guidance_dims[0] > 0 else None

        self.decoder1 = Up(hidden_dim, decoder_dims[0], decoder_guidance_proj_dims[0])
        self.decoder2 = Up(decoder_dims[0], decoder_dims[1], decoder_guidance_proj_dims[1])
        self.head = nn.Conv2d(decoder_dims[1], 1, kernel_size=3, stride=1, padding=1)

        self.pad_len = pad_len

    def feature_map(self, img_feats, text_feats):
        # concatenated feature volume for feature aggregation baselines
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        img_feats = repeat(img_feats, "B C H W -> B C T H W", T=text_feats.shape[1])
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        text_feats = text_feats.mean(dim=-2) # average text features over different prompts
        text_feats = F.normalize(text_feats, dim=-1) # B T C
        text_feats = repeat(text_feats, "B T C -> B C T H W", H=img_feats.shape[-2], W=img_feats.shape[-1])
        return torch.cat((img_feats, text_feats), dim=1) # B 2C T H W

    def correlation(self, img_feats, text_feats):
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        corr = torch.einsum('bchw, btpc -> bpthw', img_feats, text_feats)
        return corr

    def corr_embed(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B P T H W -> (B T) P H W')
        corr_embed = self.conv1(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed
    
    def corr_fusion_embed_minimum(self,clip_corr,dino_corr):
        # this one does not import a 1*1 conv
        # instead we modify the original embedding layer to adapt to the concatenated corr volume.
        B = clip_corr.shape[0]
        clip_corr = rearrange(clip_corr, 'B P T H W -> (B T) P H W')
        dino_corr = rearrange(dino_corr, 'B P T H W -> (B T) P H W')
        fused_corr = torch.cat([clip_corr,dino_corr],dim = 1)
        fused_corr = self.conv1_modified(fused_corr)
        fused_corr = rearrange(fused_corr, '(B T) C H W -> B C T H W', B=B)
        return fused_corr

    def corr_fusion_embed_seperate(self,clip_corr,dino_corr):
        # this one does not import a 1*1 conv
        # instead we modify the original embedding layer to adapt to the concatenated corr volume.
        B = clip_corr.shape[0]
        clip_corr = rearrange(clip_corr, 'B P T H W -> (B T) P H W')
        dino_corr = rearrange(dino_corr, 'B P T H W -> (B T) P H W')
 
        clip_embed_corr = self.conv1(clip_corr)
        dino_embed_corr = self.conv2(dino_corr)

        fused_corr = torch.cat([clip_embed_corr,dino_embed_corr],dim = 1)
        fused_corr = self.fusion_corr(fused_corr)
        fused_corr = rearrange(fused_corr, '(B T) C H W -> B C T H W', B=B)

        return fused_corr
    
    def corr_fusion_embed_summation_res(self,clip_corr,dino_corr):
        B = clip_corr.shape[0]
        clip_corr = rearrange(clip_corr, 'B P T H W -> (B T) P H W')
        dino_corr = rearrange(dino_corr, 'B P T H W -> (B T) P H W')
 
        clip_embed_corr = self.conv1(clip_corr)
        dino_embed_corr = self.conv2(dino_corr)

        fused_corr = clip_embed_corr + dino_embed_corr
        # fused_corr = self.fusion_corr(fused_corr)
        fused_corr = fused_corr + clip_embed_corr
        fused_corr = rearrange(fused_corr, '(B T) C H W -> B C T H W', B=B)

        return fused_corr
        
        
    def corr_fusion_embed(self,clip_corr,dino_corr):
        B = clip_corr.shape[0]
        clip_corr = rearrange(clip_corr, 'B P T H W -> (B T) P H W')
        dino_corr = rearrange(dino_corr, 'B P T H W -> (B T) P H W')
        fused_corr = torch.cat([clip_corr,dino_corr],dim = 1)
        fused_corr = self.fusion_corr(fused_corr)
        fused_corr = self.conv1(fused_corr)
        fused_corr = rearrange(fused_corr, '(B T) C H W -> B C T H W', B=B)
        return fused_corr
        # exit()
        
    def corr_projection(self, x, proj):
        corr_embed = rearrange(x, 'B C T H W -> B T H W C')
        corr_embed = proj(corr_embed)
        corr_embed = rearrange(corr_embed, 'B T H W C -> B C T H W')
        return corr_embed

    def upsample(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = F.interpolate(corr_embed, scale_factor=2, mode='bilinear', align_corners=True)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed

    def conv_decoder(self, x, guidance):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = self.decoder1(corr_embed, guidance[0])
        corr_embed = self.decoder2(corr_embed, guidance[1])
        corr_embed = self.head(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
        return corr_embed
    def forward(self, img_feats,dino_feat, text_feats, appearance_guidance):
        """
        Arguments:
            img_feats: (B, C, H, W)
            text_feats: (B, T, P, C) T是类别的个数
            apperance_guidance: tuple of (B, C, H, W)
        """

        classes = None
        ################START Here modified by YCY START###############
        corr = self.correlation(img_feats, text_feats)
        dino_corr = self.correlation(dino_feat,text_feats)
        if self.pad_len > 0 and text_feats.size(1) > self.pad_len:
            avg = corr.permute(0, 2, 1, 3, 4).flatten(-3).max(dim=-1)[0] 
            avg_dino = dino_corr.permute(0, 2, 1, 3, 4).flatten(-3).max(dim=-1)[0] 
            classes = avg.topk(self.pad_len, dim=-1, sorted=False)[1]
            classes_dino = avg_dino.topk(self.pad_len, dim=-1, sorted=False)[1]
            th_text = F.normalize(text_feats, dim=-1)
            clip_th_text = torch.gather(th_text, dim=1, index=classes[..., None, None].expand(-1, -1, th_text.size(-2), th_text.size(-1)))
            dino_th_text = torch.gather(th_text, dim=1, index=classes_dino[..., None, None].expand(-1, -1, th_text.size(-2), th_text.size(-1)))
            orig_clases = text_feats.size(1)
            img_feats = F.normalize(img_feats, dim=1) # B C H W
            dino_feats = F.normalize(dino_feat, dim=1) # B C H W
            corr = torch.einsum('bchw, btpc -> bpthw', img_feats, clip_th_text)
            dino_corr = torch.einsum('bchw, btpc -> bpthw', dino_feats, dino_th_text)
        #corr = self.feature_map(img_feats, text_feats)
        #exit()
        # fused_corr = corr+dino_corr

        # exit()
        
        # fused_corr_embed = self.corr_embed(fused_corr)
        # fused_corr_embed1 = self.corr_fusion_embed_minimum(clip_corr = corr,dino_corr=dino_corr)
        fused_corr_embed = self.corr_fusion_embed_summation_res(clip_corr = corr,dino_corr=dino_corr)
        # print(233333)
        # clip_corr_embed = self.corr_embed(corr)
        # dino_corr_embed = self.corr_embed(dino_corr)
        ################END Here modified by YCY END###############
        
        # print(clip_corr_embed)
        projected_guidance, projected_text_guidance, projected_decoder_guidance = None, None, [None, None]
        if self.guidance_projection is not None:
            projected_guidance = self.guidance_projection(appearance_guidance[0])
        if self.decoder_guidance_projection is not None:
            projected_decoder_guidance = [proj(g) for proj, g in zip(self.decoder_guidance_projection, appearance_guidance[1:])]

        if self.text_guidance_projection is not None:
            text_feats = text_feats.mean(dim=-2)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            projected_text_guidance = self.text_guidance_projection(text_feats)

        for layer in self.layers:
            fused_corr_embed = layer(fused_corr_embed, projected_guidance, projected_text_guidance)
            # clip_corr_embed = layer(clip_corr_embed, projected_guidance, projected_text_guidance)
            # dino_corr_embed = layer(dino_corr_embed, projected_guidance, projected_text_guidance)


        # fusion_corr_embed = clip_corr_embed + dino_corr_embed

        # logit = self.conv_decoder(clip_corr_embed, projected_decoder_guidance)
        logit = self.conv_decoder(fused_corr_embed, projected_decoder_guidance)

        
        if classes is not None:
            out = torch.full((logit.size(0), orig_clases, logit.size(2), logit.size(3)), -100., device=logit.device)
            out.scatter_(dim=1, index=classes[..., None, None].expand(-1, -1, logit.size(-2), logit.size(-1)), src=logit)
            logit = out
        return logit
class FusionAggregatorVer09a(nn.Module):
    def __init__(self, 
        text_guidance_dim=512,
        text_guidance_proj_dim=128,
        appearance_guidance_dim=512,
        appearance_guidance_proj_dim=128,
        decoder_dims = (64, 32),
        decoder_guidance_dims=(256, 128),
        decoder_guidance_proj_dims=(32, 16),
        num_layers=4,
        nheads=4, 
        hidden_dim=128,
        pooling_size=(6, 6),
        feature_resolution=(24, 24),
        window_size=12,
        attention_type='linear',
        prompt_channel=1,
        pad_len=256,
    ) -> None:
        """
        Cost Aggregation Model for CAT-Seg
        Args:
            text_guidance_dim: Dimension of text guidance
            text_guidance_proj_dim: Dimension of projected text guidance
            appearance_guidance_dim: Dimension of appearance guidance
            appearance_guidance_proj_dim: Dimension of projected appearance guidance
            decoder_dims: Upsampling decoder dimensions
            decoder_guidance_dims: Upsampling decoder guidance dimensions
            decoder_guidance_proj_dims: Upsampling decoder guidance projected dimensions
            num_layers: Number of layers for the aggregator
            nheads: Number of attention heads
            hidden_dim: Hidden dimension for transformer blocks
            pooling_size: Pooling size for the class aggregation layer
                          To reduce computation, we apply pooling in class aggregation blocks to reduce the number of tokens during training
            feature_resolution: Feature resolution for spatial aggregation
            window_size: Window size for Swin block in spatial aggregation
            attention_type: Attention type for the class aggregation. 
            prompt_channel: Number of prompts for ensembling text features. Default: 1
            pad_len: Padding length for the class aggregation. Default: 256
                     pad_len enforces the class aggregation block to have a fixed length of tokens for all inputs
                     This means it either pads the sequence with learnable tokens in class aggregation,
                     or truncates the classes with the initial CLIP cosine-similarity scores.
                     Set pad_len to 0 to disable this feature.
            """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.layers = nn.ModuleList([
            AggregatorLayer(
                hidden_dim=hidden_dim, text_guidance_dim=text_guidance_proj_dim, appearance_guidance=appearance_guidance_proj_dim, 
                nheads=nheads, input_resolution=feature_resolution, pooling_size=pooling_size, window_size=window_size, attention_type=attention_type, pad_len=pad_len,
            ) for _ in range(num_layers)
        ])

        # self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.fusion_corr = nn.Conv2d(2*hidden_dim, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.guidance_projection = nn.Sequential(
            nn.Conv2d(appearance_guidance_dim, appearance_guidance_proj_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        ) if appearance_guidance_dim > 0 else None
        
        self.text_guidance_projection = nn.Sequential(
            nn.Linear(text_guidance_dim, text_guidance_proj_dim),
            nn.ReLU(),
        ) if text_guidance_dim > 0 else None

        self.decoder_guidance_projection = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            ) for d, dp in zip(decoder_guidance_dims, decoder_guidance_proj_dims)
        ]) if decoder_guidance_dims[0] > 0 else None

        self.decoder1 = Up(hidden_dim, decoder_dims[0], decoder_guidance_proj_dims[0])
        self.decoder2 = Up(decoder_dims[0], decoder_dims[1], decoder_guidance_proj_dims[1])
        self.head = nn.Conv2d(decoder_dims[1], 1, kernel_size=3, stride=1, padding=1)

        self.pad_len = pad_len

    def feature_map(self, img_feats, text_feats):
        # concatenated feature volume for feature aggregation baselines
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        img_feats = repeat(img_feats, "B C H W -> B C T H W", T=text_feats.shape[1])
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        text_feats = text_feats.mean(dim=-2) # average text features over different prompts
        text_feats = F.normalize(text_feats, dim=-1) # B T C
        text_feats = repeat(text_feats, "B T C -> B C T H W", H=img_feats.shape[-2], W=img_feats.shape[-1])
        return torch.cat((img_feats, text_feats), dim=1) # B 2C T H W

    def correlation(self, img_feats, text_feats):
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        corr = torch.einsum('bchw, btpc -> bpthw', img_feats, text_feats)
        return corr

    def corr_embed(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B P T H W -> (B T) P H W')
        corr_embed = self.conv1(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed
    
    def corr_fusion_embed_minimum(self,clip_corr,dino_corr):
        # this one does not import a 1*1 conv
        # instead we modify the original embedding layer to adapt to the concatenated corr volume.
        B = clip_corr.shape[0]
        clip_corr = rearrange(clip_corr, 'B P T H W -> (B T) P H W')
        dino_corr = rearrange(dino_corr, 'B P T H W -> (B T) P H W')
        fused_corr = torch.cat([clip_corr,dino_corr],dim = 1)
        fused_corr = self.conv1_modified(fused_corr)
        fused_corr = rearrange(fused_corr, '(B T) C H W -> B C T H W', B=B)
        return fused_corr

    def corr_fusion_embed_seperate(self,clip_corr,dino_corr):
        # this one does not import a 1*1 conv
        # instead we modify the original embedding layer to adapt to the concatenated corr volume.
        B = clip_corr.shape[0]
        clip_corr = rearrange(clip_corr, 'B P T H W -> (B T) P H W')
        dino_corr = rearrange(dino_corr, 'B P T H W -> (B T) P H W')
 
        clip_embed_corr = self.conv1(clip_corr)
        dino_embed_corr = self.conv2(dino_corr)

        fused_corr = torch.cat([clip_embed_corr,dino_embed_corr],dim = 1)
        fused_corr = self.fusion_corr(fused_corr)
        fused_corr = rearrange(fused_corr, '(B T) C H W -> B C T H W', B=B)
        clip_embed_corr = rearrange(clip_embed_corr, '(B T) C H W -> B C T H W', B=B)
        dino_embed_corr = rearrange(dino_embed_corr, '(B T) C H W -> B C T H W', B=B)
        return fused_corr, clip_embed_corr, dino_embed_corr
        
        
    def corr_fusion_embed(self,clip_corr,dino_corr):
        B = clip_corr.shape[0]
        clip_corr = rearrange(clip_corr, 'B P T H W -> (B T) P H W')
        dino_corr = rearrange(dino_corr, 'B P T H W -> (B T) P H W')
        fused_corr = torch.cat([clip_corr,dino_corr],dim = 1)
        fused_corr = self.fusion_corr(fused_corr)
        fused_corr = self.conv1(fused_corr)
        fused_corr = rearrange(fused_corr, '(B T) C H W -> B C T H W', B=B)
        return fused_corr
        # exit()
        
    def corr_projection(self, x, proj):
        corr_embed = rearrange(x, 'B C T H W -> B T H W C')
        corr_embed = proj(corr_embed)
        corr_embed = rearrange(corr_embed, 'B T H W C -> B C T H W')
        return corr_embed

    def upsample(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = F.interpolate(corr_embed, scale_factor=2, mode='bilinear', align_corners=True)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed

    def conv_decoder(self, x, guidance):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = self.decoder1(corr_embed, guidance[0])
        corr_embed = self.decoder2(corr_embed, guidance[1])
        corr_embed = self.head(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
        return corr_embed
    def forward(self, img_feats,dino_feat, text_feats, appearance_guidance):
        """
        Arguments:
            img_feats: (B, C, H, W)
            text_feats: (B, T, P, C) T是类别的个数
            apperance_guidance: tuple of (B, C, H, W)
        """

        classes = None
        ################START Here modified by YCY START###############
        corr = self.correlation(img_feats, text_feats)
        dino_corr = self.correlation(dino_feat,text_feats)
        if self.pad_len > 0 and text_feats.size(1) > self.pad_len:
            avg = corr.permute(0, 2, 1, 3, 4).flatten(-3).max(dim=-1)[0] 
            avg_dino = dino_corr.permute(0, 2, 1, 3, 4).flatten(-3).max(dim=-1)[0] 
            classes = avg.topk(self.pad_len, dim=-1, sorted=False)[1]
            classes_dino = avg_dino.topk(self.pad_len, dim=-1, sorted=False)[1]
            th_text = F.normalize(text_feats, dim=-1)
            clip_th_text = torch.gather(th_text, dim=1, index=classes[..., None, None].expand(-1, -1, th_text.size(-2), th_text.size(-1)))
            dino_th_text = torch.gather(th_text, dim=1, index=classes_dino[..., None, None].expand(-1, -1, th_text.size(-2), th_text.size(-1)))
            orig_clases = text_feats.size(1)
            img_feats = F.normalize(img_feats, dim=1) # B C H W
            dino_feats = F.normalize(dino_feat, dim=1) # B C H W
            corr = torch.einsum('bchw, btpc -> bpthw', img_feats, clip_th_text)
            dino_corr = torch.einsum('bchw, btpc -> bpthw', dino_feats, dino_th_text)
        #corr = self.feature_map(img_feats, text_feats)
        #exit()
        # fused_corr = corr+dino_corr

        # exit()
        
        # fused_corr_embed = self.corr_embed(fused_corr)
        # fused_corr_embed1 = self.corr_fusion_embed_minimum(clip_corr = corr,dino_corr=dino_corr)
        fused_corr_embed,clip_embed_corr, dino_embed_corr  = self.corr_fusion_embed_seperate(clip_corr = corr,dino_corr=dino_corr)
        # add the res here #

        fused_corr_embed = fused_corr_embed+clip_embed_corr

        # add the res here #
        # print(23333333333)
        # clip_corr_embed = self.corr_embed(corr)
        # dino_corr_embed = self.corr_embed(dino_corr)
        ################END Here modified by YCY END###############
        
        # print(clip_corr_embed)
        projected_guidance, projected_text_guidance, projected_decoder_guidance = None, None, [None, None]
        if self.guidance_projection is not None:
            projected_guidance = self.guidance_projection(appearance_guidance[0])
        if self.decoder_guidance_projection is not None:
            projected_decoder_guidance = [proj(g) for proj, g in zip(self.decoder_guidance_projection, appearance_guidance[1:])]

        if self.text_guidance_projection is not None:
            text_feats = text_feats.mean(dim=-2)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            projected_text_guidance = self.text_guidance_projection(text_feats)

        for layer in self.layers:
            fused_corr_embed = layer(fused_corr_embed, projected_guidance, projected_text_guidance)
            # clip_corr_embed = layer(clip_corr_embed, projected_guidance, projected_text_guidance)
            # dino_corr_embed = layer(dino_corr_embed, projected_guidance, projected_text_guidance)


        # fusion_corr_embed = clip_corr_embed + dino_corr_embed

        # logit = self.conv_decoder(clip_corr_embed, projected_decoder_guidance)
        logit = self.conv_decoder(fused_corr_embed, projected_decoder_guidance)

        
        if classes is not None:
            out = torch.full((logit.size(0), orig_clases, logit.size(2), logit.size(3)), -100., device=logit.device)
            out.scatter_(dim=1, index=classes[..., None, None].expand(-1, -1, logit.size(-2), logit.size(-1)), src=logit)
            logit = out
        return logit
class FusionAggregatorVer08(nn.Module):
    def __init__(self, 
        text_guidance_dim=512,
        text_guidance_proj_dim=128,
        appearance_guidance_dim=512,
        appearance_guidance_proj_dim=128,
        decoder_dims = (64, 32),
        decoder_guidance_dims=(256, 128),
        decoder_guidance_proj_dims=(32, 16),
        num_layers=4,
        nheads=4, 
        hidden_dim=128,
        pooling_size=(6, 6),
        feature_resolution=(24, 24),
        window_size=12,
        attention_type='linear',
        prompt_channel=1,
        pad_len=256,
    ) -> None:
        """
        Cost Aggregation Model for CAT-Seg
        Args:
            text_guidance_dim: Dimension of text guidance
            text_guidance_proj_dim: Dimension of projected text guidance
            appearance_guidance_dim: Dimension of appearance guidance
            appearance_guidance_proj_dim: Dimension of projected appearance guidance
            decoder_dims: Upsampling decoder dimensions
            decoder_guidance_dims: Upsampling decoder guidance dimensions
            decoder_guidance_proj_dims: Upsampling decoder guidance projected dimensions
            num_layers: Number of layers for the aggregator
            nheads: Number of attention heads
            hidden_dim: Hidden dimension for transformer blocks
            pooling_size: Pooling size for the class aggregation layer
                          To reduce computation, we apply pooling in class aggregation blocks to reduce the number of tokens during training
            feature_resolution: Feature resolution for spatial aggregation
            window_size: Window size for Swin block in spatial aggregation
            attention_type: Attention type for the class aggregation. 
            prompt_channel: Number of prompts for ensembling text features. Default: 1
            pad_len: Padding length for the class aggregation. Default: 256
                     pad_len enforces the class aggregation block to have a fixed length of tokens for all inputs
                     This means it either pads the sequence with learnable tokens in class aggregation,
                     or truncates the classes with the initial CLIP cosine-similarity scores.
                     Set pad_len to 0 to disable this feature.
            """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.layers = nn.ModuleList([
            AggregatorLayer(
                hidden_dim=hidden_dim, text_guidance_dim=text_guidance_proj_dim, appearance_guidance=appearance_guidance_proj_dim, 
                nheads=nheads, input_resolution=feature_resolution, pooling_size=pooling_size, window_size=window_size, attention_type=attention_type, pad_len=pad_len,
            ) for _ in range(num_layers)
        ])

        self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.conv1_modified = nn.Conv2d(prompt_channel*2, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.fusion_corr = nn.Conv2d(2*prompt_channel, prompt_channel, kernel_size=1, stride=1, padding=0)
        self.guidance_projection = nn.Sequential(
            nn.Conv2d(appearance_guidance_dim, appearance_guidance_proj_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        ) if appearance_guidance_dim > 0 else None
        
        self.text_guidance_projection = nn.Sequential(
            nn.Linear(text_guidance_dim, text_guidance_proj_dim),
            nn.ReLU(),
        ) if text_guidance_dim > 0 else None

        self.decoder_guidance_projection = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            ) for d, dp in zip(decoder_guidance_dims, decoder_guidance_proj_dims)
        ]) if decoder_guidance_dims[0] > 0 else None

        self.decoder1 = Up(hidden_dim, decoder_dims[0], decoder_guidance_proj_dims[0])
        self.decoder2 = Up(decoder_dims[0], decoder_dims[1], decoder_guidance_proj_dims[1])
        self.head = nn.Conv2d(decoder_dims[1], 1, kernel_size=3, stride=1, padding=1)

        self.pad_len = pad_len

    def feature_map(self, img_feats, text_feats):
        # concatenated feature volume for feature aggregation baselines
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        img_feats = repeat(img_feats, "B C H W -> B C T H W", T=text_feats.shape[1])
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        text_feats = text_feats.mean(dim=-2) # average text features over different prompts
        text_feats = F.normalize(text_feats, dim=-1) # B T C
        text_feats = repeat(text_feats, "B T C -> B C T H W", H=img_feats.shape[-2], W=img_feats.shape[-1])
        return torch.cat((img_feats, text_feats), dim=1) # B 2C T H W

    def correlation(self, img_feats, text_feats):
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        corr = torch.einsum('bchw, btpc -> bpthw', img_feats, text_feats)
        return corr

    def corr_embed(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B P T H W -> (B T) P H W')
        corr_embed = self.conv1(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed
    
    def corr_fusion_embed_minimum(self,clip_corr,dino_corr):
        # this one does not import a 1*1 conv
        # instead we modify the original embedding layer to adapt to the concatenated corr volume.
        B = clip_corr.shape[0]
        clip_corr = rearrange(clip_corr, 'B P T H W -> (B T) P H W')
        dino_corr = rearrange(dino_corr, 'B P T H W -> (B T) P H W')
        fused_corr = torch.cat([clip_corr,dino_corr],dim = 1)
        fused_corr = self.conv1_modified(fused_corr)
        fused_corr = rearrange(fused_corr, '(B T) C H W -> B C T H W', B=B)
        return fused_corr

        
        
    def corr_fusion_embed(self,clip_corr,dino_corr):
        B = clip_corr.shape[0]
        clip_corr = rearrange(clip_corr, 'B P T H W -> (B T) P H W')
        dino_corr = rearrange(dino_corr, 'B P T H W -> (B T) P H W')
        fused_corr = torch.cat([clip_corr,dino_corr],dim = 1)
        fused_corr = self.fusion_corr(fused_corr)
        fused_corr = self.conv1(fused_corr)
        fused_corr = rearrange(fused_corr, '(B T) C H W -> B C T H W', B=B)
        return fused_corr
        # exit()
        
    def corr_projection(self, x, proj):
        corr_embed = rearrange(x, 'B C T H W -> B T H W C')
        corr_embed = proj(corr_embed)
        corr_embed = rearrange(corr_embed, 'B T H W C -> B C T H W')
        return corr_embed

    def upsample(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = F.interpolate(corr_embed, scale_factor=2, mode='bilinear', align_corners=True)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed

    def conv_decoder(self, x, guidance):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = self.decoder1(corr_embed, guidance[0])
        corr_embed = self.decoder2(corr_embed, guidance[1])
        corr_embed = self.head(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
        return corr_embed
    def forward(self, img_feats,dino_feat, text_feats, appearance_guidance):
        """
        Arguments:
            img_feats: (B, C, H, W)
            text_feats: (B, T, P, C) T是类别的个数
            apperance_guidance: tuple of (B, C, H, W)
        """

        classes = None
        ################START Here modified by YCY START###############
        corr = self.correlation(img_feats, text_feats)
        dino_corr = self.correlation(dino_feat,text_feats)
        if self.pad_len > 0 and text_feats.size(1) > self.pad_len:
            avg = corr.permute(0, 2, 1, 3, 4).flatten(-3).max(dim=-1)[0] 
            avg_dino = dino_corr.permute(0, 2, 1, 3, 4).flatten(-3).max(dim=-1)[0] 
            classes = avg.topk(self.pad_len, dim=-1, sorted=False)[1]
            classes_dino = avg_dino.topk(self.pad_len, dim=-1, sorted=False)[1]
            th_text = F.normalize(text_feats, dim=-1)
            clip_th_text = torch.gather(th_text, dim=1, index=classes[..., None, None].expand(-1, -1, th_text.size(-2), th_text.size(-1)))
            dino_th_text = torch.gather(th_text, dim=1, index=classes_dino[..., None, None].expand(-1, -1, th_text.size(-2), th_text.size(-1)))
            orig_clases = text_feats.size(1)
            img_feats = F.normalize(img_feats, dim=1) # B C H W
            dino_feats = F.normalize(dino_feat, dim=1) # B C H W
            corr = torch.einsum('bchw, btpc -> bpthw', img_feats, clip_th_text)
            dino_corr = torch.einsum('bchw, btpc -> bpthw', dino_feats, dino_th_text)
        #corr = self.feature_map(img_feats, text_feats)
        #exit()
        # fused_corr = corr+dino_corr

        # exit()
        
        # fused_corr_embed = self.corr_embed(fused_corr)
        fused_corr_embed = self.corr_fusion_embed_minimum(clip_corr = corr,dino_corr=dino_corr)
        # clip_corr_embed = self.corr_embed(corr)
        # dino_corr_embed = self.corr_embed(dino_corr)
        ################END Here modified by YCY END###############
        
        # print(clip_corr_embed)
        projected_guidance, projected_text_guidance, projected_decoder_guidance = None, None, [None, None]
        if self.guidance_projection is not None:
            projected_guidance = self.guidance_projection(appearance_guidance[0])
        if self.decoder_guidance_projection is not None:
            projected_decoder_guidance = [proj(g) for proj, g in zip(self.decoder_guidance_projection, appearance_guidance[1:])]

        if self.text_guidance_projection is not None:
            text_feats = text_feats.mean(dim=-2)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            projected_text_guidance = self.text_guidance_projection(text_feats)

        for layer in self.layers:
            fused_corr_embed = layer(fused_corr_embed, projected_guidance, projected_text_guidance)
            # clip_corr_embed = layer(clip_corr_embed, projected_guidance, projected_text_guidance)
            # dino_corr_embed = layer(dino_corr_embed, projected_guidance, projected_text_guidance)


        # fusion_corr_embed = clip_corr_embed + dino_corr_embed

        # logit = self.conv_decoder(clip_corr_embed, projected_decoder_guidance)
        logit = self.conv_decoder(fused_corr_embed, projected_decoder_guidance)

        
        if classes is not None:
            out = torch.full((logit.size(0), orig_clases, logit.size(2), logit.size(3)), -100., device=logit.device)
            out.scatter_(dim=1, index=classes[..., None, None].expand(-1, -1, logit.size(-2), logit.size(-1)), src=logit)
            logit = out
        return logit

class FusionAggregatorVer07(nn.Module):
    def __init__(self, 
        text_guidance_dim=512,
        text_guidance_proj_dim=128,
        appearance_guidance_dim=512,
        appearance_guidance_proj_dim=128,
        decoder_dims = (64, 32),
        decoder_guidance_dims=(256, 128),
        decoder_guidance_proj_dims=(32, 16),
        num_layers=4,
        nheads=4, 
        hidden_dim=128,
        pooling_size=(6, 6),
        feature_resolution=(24, 24),
        window_size=12,
        attention_type='linear',
        prompt_channel=1,
        pad_len=256,
    ) -> None:
        """
        Cost Aggregation Model for CAT-Seg
        Args:
            text_guidance_dim: Dimension of text guidance
            text_guidance_proj_dim: Dimension of projected text guidance
            appearance_guidance_dim: Dimension of appearance guidance
            appearance_guidance_proj_dim: Dimension of projected appearance guidance
            decoder_dims: Upsampling decoder dimensions
            decoder_guidance_dims: Upsampling decoder guidance dimensions
            decoder_guidance_proj_dims: Upsampling decoder guidance projected dimensions
            num_layers: Number of layers for the aggregator
            nheads: Number of attention heads
            hidden_dim: Hidden dimension for transformer blocks
            pooling_size: Pooling size for the class aggregation layer
                          To reduce computation, we apply pooling in class aggregation blocks to reduce the number of tokens during training
            feature_resolution: Feature resolution for spatial aggregation
            window_size: Window size for Swin block in spatial aggregation
            attention_type: Attention type for the class aggregation. 
            prompt_channel: Number of prompts for ensembling text features. Default: 1
            pad_len: Padding length for the class aggregation. Default: 256
                     pad_len enforces the class aggregation block to have a fixed length of tokens for all inputs
                     This means it either pads the sequence with learnable tokens in class aggregation,
                     or truncates the classes with the initial CLIP cosine-similarity scores.
                     Set pad_len to 0 to disable this feature.
            """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.layers = nn.ModuleList([
            AggregatorLayer(
                hidden_dim=hidden_dim, text_guidance_dim=text_guidance_proj_dim, appearance_guidance=appearance_guidance_proj_dim, 
                nheads=nheads, input_resolution=feature_resolution, pooling_size=pooling_size, window_size=window_size, attention_type=attention_type, pad_len=pad_len,
            ) for _ in range(num_layers)
        ])

        self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)

        self.guidance_projection = nn.Sequential(
            nn.Conv2d(appearance_guidance_dim, appearance_guidance_proj_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        ) if appearance_guidance_dim > 0 else None
        
        self.text_guidance_projection = nn.Sequential(
            nn.Linear(text_guidance_dim, text_guidance_proj_dim),
            nn.ReLU(),
        ) if text_guidance_dim > 0 else None

        self.decoder_guidance_projection = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            ) for d, dp in zip(decoder_guidance_dims, decoder_guidance_proj_dims)
        ]) if decoder_guidance_dims[0] > 0 else None

        self.decoder1 = Up(hidden_dim, decoder_dims[0], decoder_guidance_proj_dims[0])
        self.decoder2 = Up(decoder_dims[0], decoder_dims[1], decoder_guidance_proj_dims[1])
        self.head = nn.Conv2d(decoder_dims[1], 1, kernel_size=3, stride=1, padding=1)

        self.pad_len = pad_len

    def feature_map(self, img_feats, text_feats):
        # concatenated feature volume for feature aggregation baselines
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        img_feats = repeat(img_feats, "B C H W -> B C T H W", T=text_feats.shape[1])
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        text_feats = text_feats.mean(dim=-2) # average text features over different prompts
        text_feats = F.normalize(text_feats, dim=-1) # B T C
        text_feats = repeat(text_feats, "B T C -> B C T H W", H=img_feats.shape[-2], W=img_feats.shape[-1])
        return torch.cat((img_feats, text_feats), dim=1) # B 2C T H W

    def correlation(self, img_feats, text_feats):
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        corr = torch.einsum('bchw, btpc -> bpthw', img_feats, text_feats)
        return corr

    def corr_embed(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B P T H W -> (B T) P H W')
        corr_embed = self.conv1(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed
    
    def corr_projection(self, x, proj):
        corr_embed = rearrange(x, 'B C T H W -> B T H W C')
        corr_embed = proj(corr_embed)
        corr_embed = rearrange(corr_embed, 'B T H W C -> B C T H W')
        return corr_embed

    def upsample(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = F.interpolate(corr_embed, scale_factor=2, mode='bilinear', align_corners=True)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed

    def conv_decoder(self, x, guidance):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = self.decoder1(corr_embed, guidance[0])
        corr_embed = self.decoder2(corr_embed, guidance[1])
        corr_embed = self.head(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
        return corr_embed
    def forward(self, img_feats,dino_feat, text_feats, appearance_guidance):
        """
        Arguments:
            img_feats: (B, C, H, W)
            text_feats: (B, T, P, C) T是类别的个数
            apperance_guidance: tuple of (B, C, H, W)
        """

        classes = None
        ################START Here modified by YCY START###############
        corr = self.correlation(img_feats, text_feats)
        dino_corr = self.correlation(dino_feat,text_feats)
        if self.pad_len > 0 and text_feats.size(1) > self.pad_len:
            avg = corr.permute(0, 2, 1, 3, 4).flatten(-3).max(dim=-1)[0] 
            avg_dino = dino_corr.permute(0, 2, 1, 3, 4).flatten(-3).max(dim=-1)[0] 
            classes = avg.topk(self.pad_len, dim=-1, sorted=False)[1]
            classes_dino = avg_dino.topk(self.pad_len, dim=-1, sorted=False)[1]
            th_text = F.normalize(text_feats, dim=-1)
            clip_th_text = torch.gather(th_text, dim=1, index=classes[..., None, None].expand(-1, -1, th_text.size(-2), th_text.size(-1)))
            dino_th_text = torch.gather(th_text, dim=1, index=classes_dino[..., None, None].expand(-1, -1, th_text.size(-2), th_text.size(-1)))
            orig_clases = text_feats.size(1)
            img_feats = F.normalize(img_feats, dim=1) # B C H W
            dino_feats = F.normalize(dino_feat, dim=1) # B C H W
            corr = torch.einsum('bchw, btpc -> bpthw', img_feats, clip_th_text)
            dino_corr = torch.einsum('bchw, btpc -> bpthw', dino_feats, dino_th_text)
        #corr = self.feature_map(img_feats, text_feats)
        fused_corr = corr+dino_corr

        # exit()
        
        fused_corr_embed = self.corr_embed(fused_corr)
        # clip_corr_embed = self.corr_embed(corr)
        # dino_corr_embed = self.corr_embed(dino_corr)
        ################END Here modified by YCY END###############
        
        # print(clip_corr_embed)
        projected_guidance, projected_text_guidance, projected_decoder_guidance = None, None, [None, None]
        if self.guidance_projection is not None:
            projected_guidance = self.guidance_projection(appearance_guidance[0])
        if self.decoder_guidance_projection is not None:
            projected_decoder_guidance = [proj(g) for proj, g in zip(self.decoder_guidance_projection, appearance_guidance[1:])]

        if self.text_guidance_projection is not None:
            text_feats = text_feats.mean(dim=-2)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            projected_text_guidance = self.text_guidance_projection(text_feats)

        for layer in self.layers:
            fused_corr_embed = layer(fused_corr_embed, projected_guidance, projected_text_guidance)
            # clip_corr_embed = layer(clip_corr_embed, projected_guidance, projected_text_guidance)
            # dino_corr_embed = layer(dino_corr_embed, projected_guidance, projected_text_guidance)


        # fusion_corr_embed = clip_corr_embed + dino_corr_embed

        # logit = self.conv_decoder(clip_corr_embed, projected_decoder_guidance)
        logit = self.conv_decoder(fused_corr_embed, projected_decoder_guidance)
        

        if classes is not None:
            out = torch.full((logit.size(0), orig_clases, logit.size(2), logit.size(3)), -100., device=logit.device)
            out.scatter_(dim=1, index=classes[..., None, None].expand(-1, -1, logit.size(-2), logit.size(-1)), src=logit)
            logit = out
        return logit
