import Utils.Conv2dReLU as Conv2dReLU
import Utils.Decoder as Decoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from ops_dcnv3 import modules as opsm

# 窗口注意力
class WindowAttention(nn.Module):
    """窗口注意力机制 - 优化实现"""

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 相对位置偏置表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))

        # 生成相对位置索引
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        # 投影层
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # 添加相对位置偏置
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(
            self.window_size * self.window_size,
            self.window_size * self.window_size,
            -1).permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = F.softmax(attn, dim=-1)
        else:
            attn = F.softmax(attn, dim=-1)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# 窗口划分与还原
def window_partition(x, window_size):
    B, H, W, C = x.shape
    H_padded = (H + window_size - 1) // window_size * window_size
    W_padded = (W + window_size - 1) // window_size * window_size

    if H_padded != H or W_padded != W:
        x = F.pad(x, (0, 0, 0, W_padded - W, 0, H_padded - H))

    x = x.view(B, H_padded // window_size, window_size, W_padded // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    H_padded = (H + window_size - 1) // window_size * window_size
    W_padded = (W + window_size - 1) // window_size * window_size

    B = windows.shape[0] // ((H_padded // window_size) * (W_padded // window_size))

    x = windows.view(B, H_padded // window_size, W_padded // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H_padded, W_padded, -1)
    x = x[:, :H, :W, :]
    return x


# Swin-Intern Block：DCNv3 与窗口注意力并行融合
class SwinInternBlock(nn.Module):
    """混合DCNv3和窗口注意力的核心模块"""

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.0,
                 drop=0.0,
                 drop_path=0.0,
                 layer_type='shallow'):
        super().__init__()
        self.dim = dim
        self.layer_type = layer_type
        self.window_size = window_size

        # 归一化层
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # 根据层类型调整两种注意力的权重
        if layer_type == 'shallow':
            self.dcn_weight = 0.7
            self.window_weight = 0.3
            dcn_channels = dim
            window_channels = dim
        else:
            self.dcn_weight = 0.3
            self.window_weight = 0.7
            dcn_channels = max(dim // 2, 32)
            window_channels = dim

        # DCNv3分支
        self.dcn = opsm.DCNv3(
            channels=dcn_channels,
            kernel_size=3,
            stride=1,
            pad=1,
            dilation=1,
            group=dcn_channels // 16,
            offset_scale=1.0,
            act_layer='GELU',
            norm_layer='LN',
            center_feature_scale=False
        )
        self.dcn_input_proj = nn.Linear(dim, dcn_channels)
        self.dcn_output_proj = nn.Linear(dcn_channels, dim)

        # 窗口注意力分支
        self.window_attn = WindowAttention(
            dim=window_channels,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=True,
            attn_drop=drop,
            proj_drop=drop
        )
        self.window_proj = nn.Linear(window_channels, dim) if window_channels != dim else nn.Identity()

        # 动态门控融合机制
        self.fusion_gate = nn.Sequential(
            nn.Linear(dim * 2, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 2),
            nn.Softmax(dim=-1)
        )

        # MLP和DropPath
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, H, W):
        shortcut = x
        x = self.norm1(x)

        B, L, C = x.shape
        assert L == H * W, f"特征尺寸不匹配: {L} != {H}*{W}"

        # 转换为空间格式
        x_spatial = x.view(B, H, W, C)

        # DCNv3分支
        dcn_input = self.dcn_input_proj(x_spatial)
        dcn_feat = self.dcn(dcn_input)
        dcn_feat = self.dcn_output_proj(dcn_feat)
        dcn_feat = dcn_feat.reshape(B, L, -1)

        # 窗口注意力分支
        x_windows = window_partition(x_spatial, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.window_attn(x_windows)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        attn_feat = window_reverse(attn_windows, self.window_size, H, W)
        attn_feat = attn_feat.reshape(B, L, -1)
        attn_feat = self.window_proj(attn_feat)

        # 智能融合
        fusion_input = torch.cat([dcn_feat, attn_feat], dim=-1)
        gate_weights = self.fusion_gate(fusion_input)

        dcn_weight = gate_weights[..., 0:1] * self.dcn_weight
        attn_weight = gate_weights[..., 1:2] * self.window_weight
        total_weight = dcn_weight + attn_weight + 1e-6
        dcn_weight = dcn_weight / total_weight
        attn_weight = attn_weight / total_weight
        fused_feat = dcn_weight * dcn_feat + attn_weight * attn_feat

        # 残差连接
        x = shortcut + self.drop_path(fused_feat)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# 下采样
class PatchMerging(nn.Module):
    """特征图下采样（2倍下采样）"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, f"输入特征尺寸错误: L={L}, H*W={H * W}"

        # 转换为空间格式
        x = x.view(B, H, W, C)

        # 按2×2区域划分
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        # 拼接特征
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class SwinInternEncoder(nn.Module):
    """Swin-Intern编码器：分层混合注意力"""

    def __init__(self,
                 in_chans=2,
                 embed_dim=32,
                 depths=[2, 2, 6, 2],
                 num_heads=[1, 2, 4, 8],
                 window_size=7,
                 mlp_ratio=4.0,
                 drop_rate=0.0,
                 drop_path_rate=0.2):
        super().__init__()

        # 初始嵌入层（无下采样）
        self.conv1 = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1)
        self.act1 = nn.GELU()
        self.norm = nn.LayerNorm(embed_dim)

        # 随机深度衰减
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # 构建混合注意力阶段
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        current_dim = embed_dim
        for i, depth in enumerate(depths):
            stage_blocks = nn.ModuleList()

            # 确定当前阶段的层类型
            layer_type = 'shallow' if i < len(depths) // 2 else 'deep'

            for j in range(depth):
                block = SwinInternBlock(
                    dim=current_dim,
                    num_heads=num_heads[i],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_path=dpr[sum(depths[:i]) + j],
                    layer_type=layer_type
                )
                stage_blocks.append(block)

            self.stages.append(stage_blocks)

            # 下采样层（最后一阶段除外）
            if i < len(depths) - 1:
                downsample = PatchMerging(current_dim)
                self.downsamples.append(downsample)
                current_dim *= 2

    def forward(self, x):
        """前向传播：返回多尺度特征图 [256, 128, 64, 32]"""
        # 初始嵌入 - 无下采样
        x = self.conv1(x)
        x = self.act1(x)

        # 应用LayerNorm
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]

        # 转换为序列格式
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        x = x.view(B, H * W, C)  # [B, L, C]

        features = []
        current_H, current_W = H, W

        # 逐阶段处理
        for i, stage in enumerate(self.stages):
            for block in stage:
                x = block(x, current_H, current_W)

            # 保存特征图 [B, C, H, W]
            feat = x.view(B, current_H, current_W, -1).permute(0, 3, 1, 2).contiguous()
            features.append(feat)

            # 下采样（除了最后一层）
            if i < len(self.stages) - 1:
                x = self.downsamples[i](x, current_H, current_W)
                current_H //= 2
                current_W //= 2

        return features

class BCAF(nn.Module):
    """Bidirectional Coordinate Attention Fusion"""

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

class SwinInternRegNet(nn.Module):
    def ConvModule(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                   bias=False, batchnorm=False):
        layers = []
        # 1. 卷积层
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias))

        # 2.BatchNorm
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))

        # 3. 激活函数
        layers.append(nn.PReLU())

        # GroupNorm：对齐 SAM 特征分布
        num_groups = 16 if out_channels >= 16 else out_channels
        layers.append(nn.GroupNorm(num_groups, out_channels))

        return nn.Sequential(*layers)

    def __init__(self, S1Channel, S2Channel, S3Channel, S4Channel, Start_Channel,
                 in_channel=2):
        super().__init__()
        # 编码器
        self.enc = SwinInternEncoder(
            in_chans=2,
            embed_dim=32,
            depths=[2, 2, 6, 2],
            num_heads=[1, 2, 4, 8],
            window_size=7,
            mlp_ratio=4.0,
            drop_rate=0.1,
            drop_path_rate=0.2
        )

        # 调整通道映射
        self.CB1 = self.ConvModule(in_channels=S1Channel, out_channels=Start_Channel,
                                   kernel_size=3, stride=1, padding=1, batchnorm=True)
        self.CB2 = self.ConvModule(in_channels=S2Channel, out_channels=Start_Channel * 2,
                                   kernel_size=3, stride=1, padding=1, batchnorm=True)
        self.CB3 = self.ConvModule(in_channels=S3Channel, out_channels=Start_Channel * 4,
                                   kernel_size=3, stride=1, padding=1, batchnorm=True)
        self.CB4 = self.ConvModule(in_channels=S4Channel, out_channels=Start_Channel * 8,
                                   kernel_size=3, stride=1, padding=1, batchnorm=True)

        self.ec1 = Conv2dReLU.Conv2dReLU(2, 32, 3, 1, use_batchnorm=False)

        # 注意力模块
        self.attn1 = BCAF(Start_Channel)
        self.attn2 = BCAF(Start_Channel * 2)
        self.attn3 = BCAF(Start_Channel * 4)
        self.attn4 = BCAF(Start_Channel * 8)

        # 修改解码器块，使其输出256×256
        # 移除上采样块，改为直接卷积处理
        # 解码器模块 - 调整输入输出通道
        self.up0 = Decoder.DecoderBlock(256, 128, skip_channels=128, use_batchnorm=False)  # 32×32 -> 64×64
        self.up1 = Decoder.DecoderBlock(128, 64, skip_channels=64, use_batchnorm=False)  # 64×64 -> 128×128
        self.up2 = Decoder.DecoderBlock(64, 32, skip_channels=32, use_batchnorm=False)  # 128×128 -> 256×256

        # 输出前的特征整合
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

        # 回归头：输出 2 通道形变场
        self.reg_head = Decoder.RegistrationHead(
            in_channels=16,
            out_channels=2,
            kernel_size=3,
        )

    # 支持的特征融合模式（对应论文 Table 1 消融实验）
    FUSION_MODES = ('full', 'concat', 'sam_only', 'enc_only')

    def forward(self, source, target, score0, score1, score2, score3, fusion_mode='full'):
        """
        Args:
            source, target: 待配准的运动图像与固定图像 (B, 1, 256, 256)
            score0~score3:  SAM 特征适配器输出的 4 尺度特征
            fusion_mode:    特征融合模式，对应论文 Table 1 消融实验
                'full'     - BCAF 双向融合 SAM 特征与编码器特征（完整模型, Row 6）
                'concat'   - SAM 特征与编码器特征直接拼接, 不经过 BCAF（Row 5）
                'sam_only' - 仅使用 SAM 特征 FSAM（Row 2）
                'enc_only' - 仅使用配准编码器特征 FEn（Row 1）
        Returns:
            v: 形变场 (B, 2, 256, 256)
        """
        if fusion_mode not in self.FUSION_MODES:
            raise ValueError(
                f"Unknown fusion_mode '{fusion_mode}'. "
                f"Choose from {self.FUSION_MODES}."
            )

        input_fusion = torch.cat([source, target], dim=1)

        # 编码器前向传播，输出尺寸: 256×256, 128×128, 64×64, 32×32
        intern_f1, intern_f2, intern_f3, intern_f4 = self.enc(input_fusion)

        # 对 SAM 特征进行通道适配
        vscore1 = self.CB1(score0)  # 256×256
        vscore2 = self.CB2(score1)  # 128×128
        vscore3 = self.CB3(score2)  # 64×64
        vscore4 = self.CB4(score3)  # 32×32

        # 根据融合模式，确定送入解码器各层的 (主特征, skip, skip2)
        if fusion_mode == 'full':
            # BCAF 双向融合：固定方向 attn(vscore, intern)，运动方向 attn(intern, vscore)
            x4 = self.attn4(vscore4, intern_f4)
            s3, s3b = self.attn3(vscore3, intern_f3), self.attn3(intern_f3, vscore3)
            s2, s2b = self.attn2(vscore2, intern_f2), self.attn2(intern_f2, vscore2)
            s1, s1b = self.attn1(vscore1, intern_f1), self.attn1(intern_f1, vscore1)
        elif fusion_mode == 'concat':
            # 编码器特征与 SAM 特征直接作为两路 skip 拼接，不经过 BCAF
            x4 = intern_f4
            s3, s3b = intern_f3, vscore3
            s2, s2b = intern_f2, vscore2
            s1, s1b = intern_f1, vscore1
        elif fusion_mode == 'sam_only':
            x4 = vscore4
            s3, s3b = vscore3, vscore3
            s2, s2b = vscore2, vscore2
            s1, s1b = vscore1, vscore1
        else:  # 'enc_only'
            x4 = intern_f4
            s3, s3b = intern_f3, intern_f3
            s2, s2b = intern_f2, intern_f2
            s1, s1b = intern_f1, intern_f1

        x = self.up0(x4, s3, s3b)  # 32×32 -> 64×64
        x = self.up1(x, s2, s2b)   # 64×64 -> 128×128
        x = self.up2(x, s1, s1b)   # 128×128 -> 256×256
        x = self.final_conv(x)     # 256×256×16
        v = self.reg_head(x)       # 256×256×2
        return v


if __name__ == "__main__":
    # 自测：4 个尺度的 SAM 特征通道数需与 CB1~CB4 的输入一致
    S1Channel, S2Channel, S3Channel, S4Channel = 128, 256, 512, 1024
    Start_Channel = 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = SwinInternRegNet(
        S1Channel, S2Channel, S3Channel, S4Channel, Start_Channel
    ).to(device)

    source = torch.randn(2, 1, 256, 256).to(device)
    target = torch.randn(2, 1, 256, 256).to(device)
    score0 = torch.randn(2, S1Channel, 256, 256).to(device)
    score1 = torch.randn(2, S2Channel, 128, 128).to(device)
    score2 = torch.randn(2, S3Channel, 64, 64).to(device)
    score3 = torch.randn(2, S4Channel, 32, 32).to(device)

    # 逐一验证 4 种融合模式都能正常前向并输出 (2, 2, 256, 256)
    for mode in SwinInternRegNet.FUSION_MODES:
        flow = model(source, target, score0, score1, score2, score3, fusion_mode=mode)
        assert flow.shape == (2, 2, 256, 256), f"{mode}: {flow.shape}"
        print(f"fusion_mode='{mode}' -> flow {tuple(flow.shape)}  OK")