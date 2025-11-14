import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
import math
from einops import repeat, rearrange

# 设置全局标志，表示mamba不可用
MAMBA_AVAILABLE = False
selective_scan_fn = None
mamba_inner_fn = None
causal_conv1d_fn = None

#print("Using simplified TransMamba implementation (mamba-ssm not required)")

__all__ = ['TransMambaBlock', 'SpectralEnhancedFFN']

# 修复ChannelAttention类
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 修复：简化逻辑，避免条件判断
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        attention = self.sigmoid(out)
        return attention

# 修复SpatialAttention类
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 修复：简化逻辑，避免条件判断
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_concat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv1(x_concat))
        return attention

##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        """确保LayerNorm正确处理4D张量"""
        if len(x.shape) == 4:
            h, w = x.shape[-2:]
            return to_4d(self.body(to_3d(x)), h, w)
        else:
            return self.body(x)

##########################################################################
# filepath: [transMamba.py](http://_vscodecontentref_/0)

class SimpleMamba(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        
        # 修复：使用Conv2d代替Linear进行输入投影
        self.in_proj = nn.Conv2d(self.d_model, self.d_inner * 2, kernel_size=1, bias=False)
        
        # 深度卷积
        self.dwconv = nn.Conv2d(self.d_inner * 2, self.d_inner * 2, 
                               kernel_size=3, padding=1, groups=self.d_inner * 2)
        
        # 注意力机制
        self.ca = ChannelAttention(self.d_inner)
        self.sa = SpatialAttention()
        
        # 修复：使用Conv2d代替Linear进行输出投影
        self.out_proj = nn.Conv2d(self.d_inner, self.d_model, kernel_size=1, bias=False)
        
        # 激活函数
        self.act = nn.SiLU()

    def forward(self, hidden_states):
        """
        简化的Mamba实现，使用注意力机制替代状态空间模型
        Args:
            hidden_states: (B, C, H, W) 4D张量
        Returns:
            out: (B, C, H, W) 4D张量
        """
        # 修复：直接处理4D张量，不需要reshape
        batch, channels, height, width = hidden_states.shape
        
        # 输入投影和深度卷积 - 都是4D操作
        xz = self.in_proj(hidden_states)  # (B, C, H, W) -> (B, d_inner*2, H, W)
        xz = self.dwconv(xz)              # (B, d_inner*2, H, W) -> (B, d_inner*2, H, W)
        
        # 分离为x和z（门控机制）
        x, z = xz.chunk(2, dim=1)  # 每个都是 (B, d_inner, H, W)
        
        # 应用通道和空间注意力
        x_att = self.ca(x) * x
        x_att = self.sa(x_att) * x_att
        
        # 门控机制：z作为门控信号
        gated = x_att * torch.sigmoid(z)
        
        # 输出投影 - 直接使用Conv2d
        out = self.out_proj(gated)  # (B, d_inner, H, W) -> (B, d_model, H, W)
        
        return out

##########################################################################
# Spectral Enhanced Feed-Forward
class SpectralEnhancedFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(SpectralEnhancedFFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

##########################################################################
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = SpectralEnhancedFFN(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class MambaBlock(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super(MambaBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.mamba1 = SimpleMamba(dim)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.mamba2 = SimpleMamba(dim)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) 4D张量
        Returns:
            out: (B, C, H, W) 4D张量
        """
        # 确保所有操作都保持4D张量格式
        x = x + self.mamba1(self.norm1(x))
        x = x + self.mamba2(self.norm2(x))
        return x

class DRMamba(nn.Module):
    def __init__(self, dim, reverse):
        super(DRMamba, self).__init__()
        self.mamba = SimpleMamba(dim)  # 使用简化的Mamba
        self.reverse = reverse

    def forward(self, x):
        if self.reverse:
            # 简单的特征翻转
            x = x.flip(dims=[2])  # 沿高度维度翻转
        
        x = self.mamba(x)
        
        if self.reverse:
            # 翻转回来
            x = x.flip(dims=[2])
        
        return x

class TransMambaBlock(nn.Module):
    def __init__(self, dim, num_heads=8, ffn_expansion_factor=1.5, bias=False, LayerNorm_type='BiasFree'):
        super(TransMambaBlock, self).__init__()

        self.transformer_block = TransformerBlock(dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type)
        self.mamba_block = MambaBlock(dim, LayerNorm_type)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) 4D张量
        Returns:
            out: (B, C, H, W) 4D张量
        """
        # 确保输入是4D张量
        if len(x.shape) != 4:
            raise ValueError(f"Expected 4D input, got {x.shape}")
            
        # Transformer分支
        x1 = self.transformer_block(x)
        
        # Mamba分支  
        x2 = self.mamba_block(x)
        
        # 融合两个分支的输出
        out = x1 + x2
        
        return out