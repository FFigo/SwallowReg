import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
import numpy as np
import ml_collections
from torch.utils import checkpoint
from torch.distributions.normal import Normal
import Utils.Decoder as Decoder

class CoordinateAttention(nn.Module):
    def __init__(self, in_channels, reduction=32):
        super().__init__()
        self.in_channels = in_channels
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mid_channels = max(in_channels // reduction, 8)
        self.conv1 = nn.Conv2d(2 * in_channels, mid_channels, 1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act = nn.GELU()
        self.conv_h = nn.Conv2d(mid_channels, in_channels, 1)
        self.conv_w = nn.Conv2d(mid_channels, in_channels, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.sigmoid = nn.Sigmoid()
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, y=None):
        identity = x
        B, C, H, W = x.shape

        # 坐标注意力计算
        x_h = self.pool_h(x)
        x_w = self.pool_w(x)
        x_h_expanded = x_h.expand(-1, -1, -1, W)
        x_w_expanded = x_w.expand(-1, -1, H, -1)
        y_feat = torch.cat([x_h_expanded, x_w_expanded], dim=1)
        y_feat = self.act(self.bn1(self.conv1(y_feat)))
        att_h = self.sigmoid(self.conv_h(y_feat))
        att_w = self.sigmoid(self.conv_w(y_feat))
        coord_attn = att_h * att_w

        # 特征增强
        if y is not None:
            value = self.value_conv(y)
            enhanced = coord_attn * value
        else:
            enhanced = coord_attn * self.value_conv(x)

        return identity + self.gamma * enhanced


# ======================== 2D 核心模块 ========================
class Mlp(nn.Module):
    """MLP 模块"""

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


def window_partition_2d(x, window_size):
    """
    2D 窗口划分
    Args:
        x: (B, H, W, C)
        window_size (tuple): 窗口大小 (H_w, W_w)
    Returns:
        windows: (num_windows*B, window_size[0], window_size[1], C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


def window_reverse_2d(windows, window_size, H, W):
    """
    2D 窗口重建
    Args:
        windows: (num_windows*B, window_size[0], window_size[1], C)
        window_size (tuple): 窗口大小 (H_w, W_w)
        H (int): 图像高度
        W (int): 图像宽度
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention2D(nn.Module):
    """2D 窗口注意力机制"""

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # 相对位置偏置表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # 相对位置索引
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # 应用相对位置偏置
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock2D(nn.Module):
    """2D Swin Transformer 块"""

    def __init__(self, dim, num_heads, window_size=(7, 7), shift_size=(0, 0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention2D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        H, W = self.H, self.W
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # 填充到窗口大小的倍数
        pad_l = pad_t = 0
        pad_r = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        pad_b = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        x = F.pad(x, (0, 0, pad_t, pad_b, pad_l, pad_r))
        _, Hp, Wp, _ = x.shape

        # 循环移位
        if min(self.shift_size) > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # 窗口划分
        x_windows = window_partition_2d(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1], C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # 合并窗口
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        shifted_x = window_reverse_2d(attn_windows, self.window_size, Hp, Wp)

        # 逆循环移位
        if min(self.shift_size) > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x

        # 移除填充
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging2D(nn.Module):
    """2D 块合并 (修复版本)"""

    def __init__(self, dim, norm_layer=nn.LayerNorm, reduce_factor=2):
        super().__init__()
        self.dim = dim
        # 修复：确保输出维度翻倍
        out_dim = 2 * dim  # 标准Swin Transformer中，块合并后维度翻倍
        self.reduction = nn.Linear(4 * dim, out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W
        W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        # 填充
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)
        return x


class BasicLayer2D(nn.Module):
    """2D 基础层"""

    def __init__(self, dim, depth, num_heads, window_size=(7, 7),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                 downsample=None, use_checkpoint=False, pat_merg_rf=2):
        super().__init__()
        self.window_size = window_size
        self.shift_size = (window_size[0] // 2, window_size[1] // 2)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.pat_merg_rf = pat_merg_rf

        # 构建块
        self.blocks = nn.ModuleList([
            SwinTransformerBlock2D(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0) if (i % 2 == 0) else (window_size[0] // 2, window_size[1] // 2),
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # 下采样层
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer, reduce_factor=self.pat_merg_rf)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        # 生成 SW-MSA 的注意力掩码
        Hp = int(np.ceil(H / self.window_size[0])) * self.window_size[0]
        Wp = int(np.ceil(W / self.window_size[1])) * self.window_size[1]
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)

        h_slices = (slice(0, -self.window_size[0]),
                    slice(-self.window_size[0], -self.shift_size[0]),
                    slice(-self.shift_size[0], None))
        w_slices = (slice(0, -self.window_size[1]),
                    slice(-self.window_size[1], -self.shift_size[1]),
                    slice(-self.shift_size[1], None))

        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition_2d(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1])
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)

        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class PatchEmbed2D(nn.Module):
    """2D 图像块嵌入"""

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # 填充到块大小的倍数
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C H//P W//P
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)
        return x


class SinPositionalEncoding2D(nn.Module):
    """2D 正弦位置编码"""

    def __init__(self, channels):
        super().__init__()
        channels = int(np.ceil(channels / 4) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        self.inv_freq = 1. / (10000 ** (torch.arange(0, channels, 2).float() / channels))

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 1)
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
        emb = torch.zeros((x, y, self.channels * 2), device=tensor.device).type(tensor.type())
        emb[:, :, :self.channels] = emb_x
        emb[:, :, self.channels:] = emb_y
        emb = emb[None, :, :, :orig_ch].repeat(batch_size, 1, 1, 1)
        return emb.permute(0, 3, 1, 2)


class SwinTransformer2D(nn.Module):
    """2D Swin Transformer"""

    def __init__(self, pretrain_img_size=224, patch_size=4, in_chans=3, embed_dim=96,
                 depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=(7, 7),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.2, norm_layer=nn.LayerNorm, ape=False, spe=False,
                 patch_norm=True, out_indices=(0, 1, 2, 3), frozen_stages=-1, use_checkpoint=False,
                 pat_merg_rf=2):
        super().__init__()
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.spe = spe
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # 图像块嵌入
        self.patch_embed = PatchEmbed2D(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # 位置编码
        if self.ape:
            pretrain_img_size = (pretrain_img_size, pretrain_img_size)
            patch_size = (patch_size, patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0],
                                  pretrain_img_size[1] // patch_size[1]]
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        elif self.spe:
            self.pos_embd = SinPositionalEncoding2D(embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # 随机深度
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # 构建层
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer2D(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                pat_merg_rf=pat_merg_rf)
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # 为每个输出添加归一化层
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def forward(self, x):
        x = self.patch_embed(x)
        Wh, Ww = x.size(2), x.size(3)

        if self.ape:
            # 插值位置嵌入
            absolute_pos_embed = F.interpolate(
                self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic', align_corners=False)
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)
        elif self.spe:
            x = (x + self.pos_embd(x)).flatten(2).transpose(1, 2)
        else:
            x = x.flatten(2).transpose(1, 2)

        x = self.pos_drop(x)
        outs = []

        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)
                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return outs


class Conv2dReLU(nn.Sequential):
    """2D 卷积+ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, use_batchnorm=True):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_batchnorm,
        )
        relu = nn.LeakyReLU(inplace=True)
        if use_batchnorm:
            nm = nn.BatchNorm2d(out_channels)
        else:
            nm = nn.Identity()
        super(Conv2dReLU, self).__init__(conv, nm, relu)


class DecoderBlock2D(nn.Module):
    """2D 解码器块"""

    def __init__(self, in_channels, out_channels, skip_channels=0, use_batchnorm=True):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm
        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class RegistrationHead2D(nn.Sequential):
    """2D 配准头"""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        conv2d.weight = nn.Parameter(Normal(0, 1e-5).sample(conv2d.weight.shape))
        conv2d.bias = nn.Parameter(torch.zeros(conv2d.bias.shape))
        super().__init__(conv2d)


class SpatialTransformer2D(nn.Module):
    """2D 空间变换器"""

    def __init__(self, size, mode='bilinear'):
        super().__init__()
        self.mode = mode
        self.size = size

        # 创建采样网格
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # 新位置 = 原始网格 + 位移场
        new_locs = self.grid.to(flow.device) + flow

        # 归一化到 [-1, 1]
        new_locs[:, 0, :, :] = 2.0 * new_locs[:, 0, :, :] / (self.size[1] - 1) - 1.0
        new_locs[:, 1, :, :] = 2.0 * new_locs[:, 1, :, :] / (self.size[0] - 1) - 1.0

        # 调整维度顺序 (B, 2, H, W) -> (B, H, W, 2)
        new_locs = new_locs.permute(0, 2, 3, 1)

        # 交换x,y顺序 (PyTorch grid_sample 要求 y,x)
        new_locs = new_locs[..., [1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)



class TransMorph(nn.Module):
    # baseline 仅支持 full（融合 SAM 特征）与 enc_only（原生 Transformer 解码）两种模式
    FUSION_MODES = ('full', 'enc_only')

    def __init__(self, config, S1Channel, S2Channel, S3Channel, S4Channel, Start_Channel=32, img_size=(256, 256)):
        super().__init__()
        self.if_convskip = config.if_convskip
        self.if_transskip = config.if_transskip
        self.img_size = img_size

        # 2D Swin Transformer 编码器
        self.transformer = SwinTransformer2D(
            patch_size=config.patch_size,
            in_chans=2,  # moving和fixed图像拼接
            embed_dim=config.embed_dim,
            depths=config.depths,
            num_heads=config.num_heads,
            window_size=config.window_size,
            mlp_ratio=config.mlp_ratio,
            qkv_bias=config.qkv_bias,
            drop_rate=config.drop_rate,
            attn_drop_rate=config.attn_drop_rate,
            drop_path_rate=config.drop_path_rate,
            ape=config.ape,
            spe=config.spe,
            patch_norm=config.patch_norm,
            out_indices=config.out_indices,
            pat_merg_rf=config.pat_merg_rf
        )

        # 空间变换层
        self.spatial_transformer = SpatialTransformer2D(img_size)

        # 四尺度语义特征适配层
        self.CB1 = self.ConvModule(in_channels=S1Channel, out_channels=Start_Channel,
                                   kernel_size=3, stride=1, padding=1, batchnorm=True)
        self.CB2 = self.ConvModule(in_channels=S2Channel, out_channels=Start_Channel * 2,
                                   kernel_size=3, stride=1, padding=1, batchnorm=True)
        self.CB3 = self.ConvModule(in_channels=S3Channel, out_channels=Start_Channel * 4,
                                   kernel_size=3, stride=1, padding=1, batchnorm=True)
        self.CB4 = self.ConvModule(in_channels=S4Channel, out_channels=Start_Channel * 8,
                                   kernel_size=3, stride=1, padding=1, batchnorm=True)

        # Coordinate Attention模块
        self.attn1 = CoordinateAttention(Start_Channel)
        self.attn2 = CoordinateAttention(Start_Channel * 2)
        self.attn3 = CoordinateAttention(Start_Channel * 4)
        self.attn4 = CoordinateAttention(Start_Channel * 8)

        # 解码器
        self.up_0 = Decoder.DecoderBlock(Start_Channel * 8, Start_Channel * 4,
                                         skip_channels=Start_Channel * 4, use_batchnorm=False)
        self.up_1 = Decoder.DecoderBlock(Start_Channel * 4, Start_Channel * 2,
                                         skip_channels=Start_Channel * 2, use_batchnorm=False)
        self.up_2 = Decoder.DecoderBlock(Start_Channel * 2, Start_Channel,
                                         skip_channels=Start_Channel, use_batchnorm=False)

        # 形变场预测头
        self.reg_head = nn.Conv2d(Start_Channel, 2, kernel_size=1, padding=0)

        # Transformer特征调整层
        # 原始输出: [96, 64, 64], [192, 32, 32], [384, 16, 16], [768, 8, 8]
        # 目标尺寸: 32 * 256 * 256, 64 * 128 * 128, 128 * 64 * 64, 256 * 32 * 32

        # 调整通道数和分辨率
        self.adjust1 = nn.Sequential(
            nn.Conv2d(96, Start_Channel, kernel_size=1),  # 96 -> 32
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)  # 64 -> 256
        )

        self.adjust2 = nn.Sequential(
            nn.Conv2d(192, Start_Channel * 2, kernel_size=1),  # 192 -> 64
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)  # 32 -> 128
        )

        self.adjust3 = nn.Sequential(
            nn.Conv2d(384, Start_Channel * 4, kernel_size=1),  # 384 -> 128
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)  # 16 -> 64
        )

        self.adjust4 = nn.Sequential(
            nn.Conv2d(768, Start_Channel * 8, kernel_size=1),  # 768 -> 256
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)  # 8 -> 32
        )

    def ConvModule(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                   bias=False, batchnorm=False):
        layers = []
        # 1. 卷积层
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size,
                                stride=stride, padding=padding, bias=bias))
        # 2. BatchNorm
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        # 3. 激活函数
        layers.append(nn.PReLU())
        # 4. GroupNorm
        num_groups = 16 if out_channels >= 16 else out_channels
        layers.append(nn.GroupNorm(num_groups, out_channels))
        return nn.Sequential(*layers)

    def forward(self, moving_img, fixed_img, score1, score2, score3, score4, fusion_mode='full'):
        if fusion_mode not in self.FUSION_MODES:
            raise ValueError(
                f"TransMorph only supports fusion_mode {self.FUSION_MODES}, got '{fusion_mode}'."
            )
        # 拼接移动和固定图像
        x = torch.cat([moving_img, fixed_img], dim=1)  # (B, 2, H, W)

        # Transformer编码器特征提取
        transformer_features = self.transformer(x)

        # 确保输出四个尺度的特征
        assert len(transformer_features) == 4, f"Transformer should output 4 scales, got {len(transformer_features)}"

        # Transformer特征调整
        # 原始: f1:96 * 64 * 64, f2:192 * 32 * 32, f3:384 * 16 * 16, f4:768 * 8 * 8
        # 目标: 32 * 256 * 256, 64 * 128 * 128, 128 * 64 * 64, 256 * 32 * 32
        f1 = self.adjust1(transformer_features[0])  # -> 32 * 256 * 256
        f2 = self.adjust2(transformer_features[1])  # -> 64 * 128 * 128
        f3 = self.adjust3(transformer_features[2])  # -> 128 * 64 * 64
        f4 = self.adjust4(transformer_features[3])  # -> 256 * 32 * 32

        # 处理SAM特征
        vscore1 = self.CB1(score1)  # -> 32 * 256 * 256
        vscore2 = self.CB2(score2)  # -> 64 * 128 * 128
        vscore3 = self.CB3(score3)  # -> 128 * 64 * 64
        vscore4 = self.CB4(score4)  # -> 256 * 32 * 32

        if fusion_mode == 'full':
            # CoordinateAttention融合
            fix_fused_f1 = self.attn1(vscore1, f1)  # 32 * 256 * 256
            fix_fused_f2 = self.attn2(vscore2, f2)  # 64 * 128 * 128
            fix_fused_f3 = self.attn3(vscore3, f3)  # 128 * 64 * 64
            fix_fused_f4 = self.attn4(vscore4, f4)  # 256 * 32 * 32

            mov_fused_f1 = self.attn1(f1, vscore1)  # 32 * 256 * 256
            mov_fused_f2 = self.attn2(f2, vscore2)  # 64 * 128 * 128
            mov_fused_f3 = self.attn3(f3, vscore3)  # 128 * 64 * 64
            # mov_fused_f4 = self.attn4(f4, vscore4)  # 256 * 32 * 32

            # 解码器上采样
            x = self.up_0(fix_fused_f4, fix_fused_f3, mov_fused_f3)  # 32 * 32 -> 64 * 64
            x = self.up_1(x, fix_fused_f2, mov_fused_f2)  # 64 * 64 -> 128 * 128
            x = self.up_2(x, fix_fused_f1, mov_fused_f1)  # 128 * 128 -> 256 * 256

            # 生成流场
            v = self.reg_head(x)  # 256 * 256 * 2
            return v
        else:

            x = self.up_0(f4, f3, None)  # 32 * 32 -> 64 * 64
            x = self.up_1(x, f2, None)  # 64 * 64 -> 128 * 128
            x = self.up_2(x, f1, None)  # 128 * 128 -> 256 * 256

            # 生成流场
            flow = self.reg_head(x)  # (B, 2, H, W)
            return flow


# 配置文件
def get_transmorph_config():
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = False  # 在您的版本中可能不需要卷积跳跃连接
    config.patch_size = 4
    config.in_chans = 2
    config.embed_dim = 96
    config.depths = [2, 2, 4, 2]
    config.num_heads = [4, 4, 8, 8]
    config.window_size = (7, 7)
    config.mlp_ratio = 4.0
    config.qkv_bias = True
    config.drop_rate = 0.0
    config.attn_drop_rate = 0.0
    config.drop_path_rate = 0.2
    config.ape = False
    config.spe = False
    config.patch_norm = True
    config.use_checkpoint = False
    config.out_indices = (0, 1, 2, 3)
    config.reg_head_chan = 16
    config.img_size = (256, 256)
    config.pat_merg_rf = 2
    return config



def create_transmorph_model(S1Channel, S2Channel, S3Channel, S4Channel, Start_Channel=32, img_size=(256, 256)):
    config = get_transmorph_config()
    model = TransMorph(config, S1Channel, S2Channel, S3Channel, S4Channel,
                       Start_Channel=Start_Channel, img_size=img_size)
    return model


# 使用示例
if __name__ == "__main__":
    # 假设SAM特征通道数
    S1Channel = 256
    S2Channel = 256
    S3Channel = 256
    S4Channel = 256

    # 创建模型
    model = create_transmorph_model(S1Channel, S2Channel, S3Channel, S4Channel)

    # 测试输入
    batch_size = 2
    moving_img = torch.randn(batch_size, 1, 256, 256)
    fixed_img = torch.randn(batch_size, 1, 256, 256)
    score1 = torch.randn(batch_size, S1Channel, 256, 256)
    score2 = torch.randn(batch_size, S2Channel, 128, 128)
    score3 = torch.randn(batch_size, S3Channel, 64, 64)
    score4 = torch.randn(batch_size, S4Channel, 32, 32)

    # 前向传播
    flow = model(moving_img, fixed_img, score1, score2, score3, score4, use_feature=True)
    print(f"Output flow shape: {flow.shape}")  # 应该是 [2, 2, 256, 256]

    # 测试不使用SAM特征
    flow_no_sam = model(moving_img, fixed_img, score1, score2, score3, score4, use_feature=False)
    print(f"Output flow (no SAM) shape: {flow_no_sam.shape}")





