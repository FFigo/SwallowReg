import torch
import torch.nn as nn
import Utils.Decoder as Decoder

# 空间变换层
class SpatialTransformer(nn.Module):
    def __init__(self, size, mode='bilinear'):
        super(SpatialTransformer, self).__init__()
        self.mode = mode
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        batch_size, _, H, W = flow.shape
        new_grid = self.grid + flow
        new_grid[:, 0, :, :] = 2 * (new_grid[:, 0, :, :] / (W - 1) - 0.5)
        new_grid[:, 1, :, :] = 2 * (new_grid[:, 1, :, :] / (H - 1) - 0.5)
        new_grid = new_grid.permute(0, 2, 3, 1)
        warped = torch.nn.functional.grid_sample(src, new_grid, mode=self.mode, padding_mode='border',
                                                 align_corners=True)
        return warped

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

# VoxelMorph baseline：接收 4 尺度 SAM 特征，输出形变场
class VoxelMorph(nn.Module):
    # baseline 仅支持 full（融合 SAM 特征）与 enc_only（原生 UNet 解码）两种模式
    FUSION_MODES = ('full', 'enc_only')

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

    def __init__(self,S1Channel, S2Channel, S3Channel, S4Channel, Start_Channel, in_channels=2, out_channels=2,img_size=(256, 256)):
        super(VoxelMorph, self).__init__()
        # VoxelMorph 原生 UNet（编码器-解码器）
        self.unet = self._build_unet(in_channels)
        # 空间变换层
        self.spatial_transformer = SpatialTransformer(img_size)
        # 四尺度 SAM 特征通道适配
        self.CB1 = self.ConvModule(in_channels=S1Channel, out_channels=Start_Channel,
                                   kernel_size=3, stride=1, padding=1, batchnorm=True)
        self.CB2 = self.ConvModule(in_channels=S2Channel, out_channels=Start_Channel * 2,
                                   kernel_size=3, stride=1, padding=1, batchnorm=True)
        self.CB3 = self.ConvModule(in_channels=S3Channel, out_channels=Start_Channel * 4,
                                   kernel_size=3, stride=1, padding=1, batchnorm=True)
        self.CB4 = self.ConvModule(in_channels=S4Channel, out_channels=Start_Channel * 8,
                                   kernel_size=3, stride=1, padding=1, batchnorm=True)



        self.attn1 = CoordinateAttention(Start_Channel)
        self.attn2 = CoordinateAttention(Start_Channel * 2)
        self.attn3 = CoordinateAttention(Start_Channel * 4)
        self.attn4 = CoordinateAttention(Start_Channel * 8)

    def _build_unet(self, in_channels):
        """
        UNet：四尺度编码-解码
        四尺度对应：enc1(64)、enc2(128)、enc3(256)、enc4(512)
        """

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        def up_conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        # 编码器（四尺度对应输出通道：64→128→256→512）
        self.enc1 = conv_block(in_channels, 32)
        self.enc2 = conv_block(32, 64)
        self.enc3 = conv_block(64, 128)
        self.enc4 = conv_block(128, 256)
        # 池化层
        self.pool = nn.MaxPool2d(2)
        # 解码器
        self.up4 = up_conv_block(256, 128)
        self.dec4 = conv_block(256, 128)
        self.up3 = up_conv_block(128, 64)
        self.dec3 = conv_block(128, 64)
        self.up2 = up_conv_block(64, 32)
        self.dec2 = conv_block(64, 32)


        # 融合 SAM 特征的解码分支
        self.up0 = Decoder.DecoderBlock(256, 128, skip_channels=128, use_batchnorm=False)  # 32×32 -> 64×64
        self.up1 = Decoder.DecoderBlock(128, 64, skip_channels=64, use_batchnorm=False)  # 64×64 -> 128×128
        self.up2 = Decoder.DecoderBlock(64, 32, skip_channels=32, use_batchnorm=False)  # 128×128 -> 256×256
        # 形变场预测头（输出2通道flow：x/y方向）
        self.reg_head = nn.Conv2d(32, 2, kernel_size=1, padding=0)

        return nn.ModuleList([self.enc1, self.enc2, self.enc3, self.enc4,
                              self.up4, self.dec4, self.up3, self.dec3,
                              self.up2, self.dec2, self.reg_head])

    def forward(self, moving_img, fixed_img, score1, score2, score3, score4, fusion_mode='full'):
        if fusion_mode not in self.FUSION_MODES:
            raise ValueError(
                f"VoxelMorph only supports fusion_mode {self.FUSION_MODES}, got '{fusion_mode}'."
            )
        # 原始图像拼接
        x = torch.cat([moving_img, fixed_img], dim=1)  # (B, 2, H, W)

        # 四尺度编码
        enc1 = self.enc1(x)  # (B, 32, H, W)
        enc2 = self.enc2(self.pool(enc1))  # (B, 64, H/2, W/2)
        enc3 = self.enc3(self.pool(enc2))  # (B, 128, H/4, W/4)
        enc4 = self.enc4(self.pool(enc3))  # (B, 256, H/8, W/8)

        vscore1 = self.CB1(score1)  # (B, 64, H, W)
        vscore2 = self.CB2(score2)  # (B, 64, H, W)
        vscore3 = self.CB3(score3)  # (B, 64, H, W)
        vscore4 = self.CB4(score4)  # (B, 64, H, W)
        if fusion_mode == 'full':
            fix_fused_f1 = self.attn1(vscore1, enc1)  # 256×256×32
            fix_fused_f2 = self.attn2(vscore2, enc2)  # 128×128×64
            fix_fused_f3 = self.attn3(vscore3, enc3)  # 64×64×128
            fix_fused_f4 = self.attn4(vscore4, enc4)  # 32×32×256
            mov_fused_f1 = self.attn1(enc1,vscore1)
            mov_fused_f2 = self.attn2(enc2,vscore2)
            mov_fused_f3 = self.attn3(enc3,vscore3)
            x = self.up0(fix_fused_f4, fix_fused_f3, mov_fused_f3)  # 32×32 -> 64×64
            x = self.up1(x, fix_fused_f2, mov_fused_f2)  # 64×64 -> 128×128
            x = self.up2(x, fix_fused_f1, mov_fused_f1)  # 128×128 -> 256×256
            v = self.reg_head(x)  # 256×256×2
            return v

        else:
            up4 = self.up4(enc4)
            up4 = torch.cat([up4, enc3], dim=1)
            dec4 = self.dec4(up4)
            up3 = self.up3(dec4)
            up3 = torch.cat([up3, enc2], dim=1)
            dec3 = self.dec3(up3)
            up2 = self.up2(dec3)
            up2 = torch.cat([up2, enc1], dim=1)
            dec2 = self.dec2(up2)
            flow = self.reg_head(dec2)  # (B, 2, H, W)
            return flow
if __name__ == '__main__':
    # 模拟 4 尺度 SAM 特征
    batch_size = 1
    img_size = (256, 256)
    moving_img = torch.randn(batch_size, 1, *img_size)
    fixed_img = torch.randn(batch_size, 1, *img_size)

    # 模拟SAM适配器输出的四尺度特征（通道需与 CB1~CB4 输入一致）
    semantic_64 = torch.randn(batch_size, 128, *img_size)
    semantic_128 = torch.randn(batch_size, 256, img_size[0] // 2, img_size[1] // 2)
    semantic_256 = torch.randn(batch_size, 512, img_size[0] // 4, img_size[1] // 4)
    semantic_512 = torch.randn(batch_size, 1024, img_size[0] // 8, img_size[1] // 8)

    # 初始化模型
    model = VoxelMorph(S1Channel=128, S2Channel=256, S3Channel=512, S4Channel=1024,
                       Start_Channel=32, img_size=img_size)

    # 逐一验证支持的融合模式
    for mode in VoxelMorph.FUSION_MODES:
        flow = model(moving_img, fixed_img, semantic_64, semantic_128, semantic_256,
                     semantic_512, fusion_mode=mode)
        print(f"fusion_mode='{mode}' -> flow {tuple(flow.shape)}")