import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry
import numpy as np


class FeaturePyramidUNet(nn.Module):
    """
    深度监督的UNet风格金字塔
    每个尺度都有独立的分割监督
    """

    def __init__(self, in_channels=256):
        super(FeaturePyramidUNet, self).__init__()

        # 编码器路径
        self.enc1 = self._conv_block(in_channels, 64, downsample=False)  # 保持64×64
        self.enc2 = self._conv_block(64, 128, downsample=True)  # 64×64 → 32×32
        self.enc3 = self._conv_block(128, 256, downsample=True)  # 32×32 → 16×16
        self.enc4 = self._conv_block(256, 512, downsample=True)  # 16×16 → 8×8

        # 解码器路径
        self.up4 = self._upconv_block(512, 256)  # 8×8 → 16×16
        self.dec4 = self._conv_block(512, 256, downsample=False)  # 拼接enc3

        self.up3 = self._upconv_block(256, 128)  # 16×16 → 32×32
        self.dec3 = self._conv_block(256, 128, downsample=False)  # 拼接enc2

        self.up2 = self._upconv_block(128, 64)  # 32×32 → 64×64
        self.dec2 = self._conv_block(128, 64, downsample=False)  # 拼接enc1

        # 最终解码
        self.up1 = self._upconv_block(64, 32)  # 64×64 → 64×64
        self.dec1 = self._conv_block(32, 32, downsample=False)

        # 每个尺度的分割头（输出特征，不直接输出分割）
        self.scale_heads = nn.ModuleList([
            self._build_scale_head(64, 64),  # 第1尺度：32→64
            self._build_scale_head(128, 128),  # 第2尺度：64→128
            self._build_scale_head(256, 256),  # 第3尺度：128→256
            self._build_scale_head(512, 512)  # 第4尺度：256→512
        ])

        # 用于深度监督的分割头（每个尺度单独的分割网络）
        self.deep_seg_heads = nn.ModuleList([
            self._build_deep_seg_head(64),  # 第1尺度
            self._build_deep_seg_head(128),  # 第2尺度
            self._build_deep_seg_head(256),  # 第3尺度
            self._build_deep_seg_head(512)  # 第4尺度
        ])

    def _conv_block(self, in_ch, out_ch, downsample=True):
        """卷积块"""
        layers = []
        if downsample:
            layers.append(nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1))
        else:
            layers.append(nn.Conv2d(in_ch, out_ch, 3, padding=1))

        layers.extend([
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        ])
        return nn.Sequential(*layers)

    def _upconv_block(self, in_ch, out_ch):
        """上采样块"""
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def _build_scale_head(self, in_ch, out_ch):
        """构建尺度特征头"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def _build_deep_seg_head(self, in_ch):
        """构建深度监督分割头"""
        return nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, 3, padding=1),
            nn.BatchNorm2d(in_ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // 2, in_ch // 4, 3, padding=1),
            nn.BatchNorm2d(in_ch // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // 4, 1, 1)
        )

    def forward(self, x, return_deep_supervision=False):
        """
        前向传播

        Args:
            x: 输入特征 [B, 256, 64, 64]
            return_deep_supervision: 是否返回深度监督结果

        Returns:
            如果return_deep_supervision=False: 返回4个尺度的特征
            如果return_deep_supervision=True: 返回(特征列表, 分割结果列表)
        """
        # 编码路径
        e1 = self.enc1(x)  # [B, 64, 256, 256]
        e2 = self.enc2(e1)  # [B, 128, 128, 128]
        e3 = self.enc3(e2)  # [B, 256, 64, 64]
        e4 = self.enc4(e3)  #

        # 解码路径 + 跳跃连接
        d3_up = self.up4(e4)  # [B, 256, 16, 16]
        d3_cat = torch.cat([d3_up, e3], dim=1)  # [B, 512, 16, 16]
        d3 = self.dec4(d3_cat)  # [B, 256, 16, 16]

        d2_up = self.up3(d3)  # [B, 128, 32, 32]
        d2_cat = torch.cat([d2_up, e2], dim=1)  # [B, 256, 32, 32]
        d2 = self.dec3(d2_cat)  # [B, 128, 32, 32]

        d1_up = self.up2(d2)  # [B, 64, 64, 64]
        d1_cat = torch.cat([d1_up, e1], dim=1)  # [B, 128, 64, 64]
        d1 = self.dec2(d1_cat)  # [B, 64, 64, 64]


        # 多尺度特征
        D_scale_features = []
        E_scale_features = []
        D_scale_features.append(self.scale_heads[0](d1))  # [B, 64, , 64]
        D_scale_features.append(self.scale_heads[1](d2))  # [B, 128, 32, 32]
        D_scale_features.append(self.scale_heads[2](d3))  # [B, 256, 16, 16]
        D_scale_features.append(self.scale_heads[3](e4))  # [B, 512, 8, 8]
        E_scale_features.append(e1)
        E_scale_features.append(e2)
        E_scale_features.append(e3)
        E_scale_features.append(e4)


        if not return_deep_supervision:
            return E_scale_features

        # 深度监督分割结果
        deep_seg_outputs = []
        for i, (feat, seg_head) in enumerate(zip(D_scale_features, self.deep_seg_heads)):
            seg_output = seg_head(feat)
            deep_seg_outputs.append(seg_output)

        return E_scale_features, deep_seg_outputs





class PixelShuffleUpsample4x(nn.Module):
    """PixelShuffle 4倍上采样模块"""

    def __init__(self, in_channels=256, out_channels=256):
        super().__init__()
        # 16 = 4 × 4 (宽高各4倍)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * 16,  # 扩展通道数
            kernel_size=3,
            padding=1
        )
        self.pixel_shuffle = nn.PixelShuffle(4)  # 4倍上采样
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)  # 通道数减少16倍，分辨率增加4倍
        x = self.norm(x)
        x = self.act(x)
        return x



class SAMFeatureAdaptor(nn.Module):
    """
    SAM特征适配器
    每个金字塔尺度都有独立的分割监督
    """

    def __init__(self, model_type='vit_b', checkpoint_path=None, image_size=224):
        super(SAMFeatureAdaptor, self).__init__()

        self.image_size = image_size
        self.model_type = model_type

        # 加载SAM：提供了有效的预训练权重则加载，否则随机初始化
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
            print(f"✅ 已加载 SAM 预训练权重: {checkpoint_path}")
        elif checkpoint_path:
            print(f"⚠️ 未找到 SAM 预训练权重: {checkpoint_path}，将使用随机初始化"
                  f"（若随后加载完整 checkpoint 可忽略此警告）")
            self.sam = sam_model_registry[model_type]()
        else:
            print("⚠️ 未指定 SAM 预训练权重路径，SAM 编码器将随机初始化")
            self.sam = sam_model_registry[model_type]()

        # 冻结SAM
        self._freeze_sam()

        # 通道适配
        self.channel_adapter = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )

        # 深度监督的金字塔网络
        self.pyramid = FeaturePyramidUNet(in_channels=256)

        # 主解码路径
        self.main_decoder = nn.Sequential(
            # 上采样1: 64×64 → 128×128
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 上采样2: 128×128 → 256×256
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        # 主分割头
        self.main_seg_head = nn.Sequential(
            nn.Conv2d(16, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid()
        )

        # 特征融合层 - 处理SAM编码器输出 256通道——64通道
        self.encoder_fusion = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        #图像编码器处理，将64*64调整到256*256
        self.emb_upsample = PixelShuffleUpsample4x(in_channels=256, out_channels=256)

        # 记录状态
        self.sam_frozen = True

    def _freeze_sam(self):
        """冻结SAM"""
        for param in self.sam.parameters():
            param.requires_grad = False
        self.sam_frozen = True




    def unfreeze_image_encoder(self):
        """解冻SAM图像编码器"""
        print("解冻SAM图像编码器...")
        for param in self.sam.image_encoder.parameters():
            param.requires_grad = True
        self.sam_frozen = False
        print("✅ SAM图像编码器已解冻")

    def get_trainable_parameters(self):
        """获取可训练参数"""
        trainable_params = []
        trainable_params.extend(list(self.channel_adapter.parameters()))
        trainable_params.extend(list(self.pyramid.parameters()))
        trainable_params.extend(list(self.main_decoder.parameters()))
        trainable_params.extend(list(self.main_seg_head.parameters()))

        if not self.sam_frozen:
            trainable_params.extend([p for p in self.sam.parameters() if p.requires_grad])

        return trainable_params

    def forward(self, x, return_pyramid_features=False, deep_supervision=True):
        """
        前向传播

        Args:
            x: 输入图像 [B, 1, H, W]
            return_pyramid_features: 是否返回金字塔特征
            deep_supervision: 是否使用深度监督

        Returns:
            如果return_pyramid_features=True: 返回金字塔特征
            如果deep_supervision=True: 返回(主输出, 深度监督输出列表)
            否则: 返回主输出
        """
        batch_size = x.size(0)

        # 1. 通道适配
        x_rgb = self.channel_adapter(x)  # [B, 3, H, W]

        # 2. 调整尺寸
        x_resized = F.interpolate(x_rgb, size=(1024, 1024),
                                  mode='bilinear', align_corners=False)

        # 3. SAM编码器
        if self.sam_frozen:
            with torch.no_grad():
                image_embeddings = self.sam.image_encoder(x_resized)  # [B, 256, 64, 64]
        else:
            image_embeddings = self.sam.image_encoder(x_resized)  # [B, 256, 64, 64]

        image_emb_upsample =self.emb_upsample(image_embeddings)



        # 4. 金字塔网络
        if deep_supervision:
            pyramid_features, deep_seg_outputs = self.pyramid(
                image_emb_upsample, return_deep_supervision=True
            )
        else:
            pyramid_features = self.pyramid(image_emb_upsample, return_deep_supervision=False)

        if return_pyramid_features:
            return pyramid_features

        # 5. 主分割路径
        main_features = self.encoder_fusion(image_embeddings)  # [B, 64, 64, 64] # [B, 64, 64, 64]
        main_features = self.main_decoder(main_features)  # [B, 16, 256, 256]
        main_output = self.main_seg_head(main_features)  # [B, 1, 256, 256]

        # 调整到原始尺寸
        main_output = F.interpolate(main_output, size=(self.image_size, self.image_size),
                                    mode='bilinear', align_corners=False)

        if not deep_supervision:
            return main_output

        # 6. 处理深度监督输出
        # 将深度监督输出上采样到图像尺寸
        processed_deep_outputs = []
        for i, seg_output in enumerate(deep_seg_outputs):
            # 上采样到图像尺寸
            if i == 0:  # 64×64
                output_up = F.interpolate(seg_output, size=(self.image_size, self.image_size),
                                          mode='bilinear', align_corners=False)
            elif i == 1:  # 32×32
                output_up = F.interpolate(seg_output, size=(self.image_size, self.image_size),
                                          mode='bilinear', align_corners=False)
            elif i == 2:  # 16×16
                output_up = F.interpolate(seg_output, size=(self.image_size, self.image_size),
                                          mode='bilinear', align_corners=False)
            else:  # 8×8
                output_up = F.interpolate(seg_output, size=(self.image_size, self.image_size),
                                          mode='bilinear', align_corners=False)

            # 应用sigmoid激活
            output_sigmoid = torch.sigmoid(output_up)
            processed_deep_outputs.append(output_sigmoid)

        return main_output, processed_deep_outputs


# 损失函数
class DeepSupervisionLoss(nn.Module):
    """
    深度监督损失
    结合主损失和辅助损失
    """

    def __init__(self, main_weight=0.6, aux_weights=[0.1, 0.1, 0.1, 0.1],
                 dice_weight=0.7, bce_weight=0.3):
        super(DeepSupervisionLoss, self).__init__()
        self.main_weight = main_weight
        self.aux_weights = aux_weights

        # Dice损失
        self.dice_loss_fn = lambda pred, target: 1 - self._dice_coefficient(pred, target)

        # BCE损失
        self.bce_loss_fn = nn.BCELoss()

        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def _dice_coefficient(self, pred, target, smooth=1e-6):
        """计算Dice系数"""
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)

        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

        return dice

    def _combined_loss(self, pred, target):
        """组合损失：Dice + BCE"""
        dice_loss = self.dice_loss_fn(pred, target)
        bce_loss = self.bce_loss_fn(pred, target)
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss

    def forward(self, main_output, aux_outputs, targets):
        """
        计算深度监督损失

        Args:
            main_output: 主输出 [B, 1, H, W]
            aux_outputs: 辅助输出列表，4个元素
            targets: 目标掩码 [B, 1, H, W]

        Returns:
            总损失
        """
        # 主损失
        main_loss = self._combined_loss(main_output, targets)

        # 辅助损失
        aux_loss = 0
        for i, (aux_output, weight) in enumerate(zip(aux_outputs, self.aux_weights)):
            loss = self._combined_loss(aux_output, targets)
            aux_loss += weight * loss

        # 总损失
        total_loss = self.main_weight * main_loss + aux_loss

        # 记录各项损失
        loss_dict = {
            'total': total_loss.item(),
            'main': main_loss.item(),
            'aux': aux_loss.item()
        }

        # 记录每个辅助损失
        for i, aux_output in enumerate(aux_outputs):
            loss_dict[f'aux_{i}'] = self._combined_loss(aux_output, targets).item()

        return total_loss, loss_dict


# 可选：渐进式深度监督权重
class ProgressiveDeepSupervisionLoss(nn.Module):
    """
    渐进式深度监督损失
    早期更关注浅层，后期更关注深层
    """

    def __init__(self, total_epochs, dice_weight=0.7, bce_weight=0.3):
        super(ProgressiveDeepSupervisionLoss, self).__init__()
        self.total_epochs = total_epochs
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

        # 基础损失函数
        self.dice_loss_fn = lambda pred, target: 1 - self._dice_coefficient(pred, target)
        self.bce_loss_fn = nn.BCELoss()

    def _dice_coefficient(self, pred, target, smooth=1e-6):
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)

        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

        return dice

    def _combined_loss(self, pred, target):
        dice_loss = self.dice_loss_fn(pred, target)
        bce_loss = self.bce_loss_fn(pred, target)
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss

    def forward(self, main_output, aux_outputs, targets, current_epoch):
        """
        计算渐进式深度监督损失

        Args:
            current_epoch: 当前epoch，用于调整权重
        """
        # 计算主损失
        main_loss = self._combined_loss(main_output, targets)

        # 计算渐进权重
        # 早期：更关注浅层（大尺度）
        # 后期：更关注深层（小尺度）
        epoch_ratio = current_epoch / self.total_epochs

        # 权重计算
        aux_weights = []
        for i in range(4):  # 4个辅助输出
            if i == 0:  # 最浅层
                weight = max(0.1, 0.2 * (1 - epoch_ratio))
            elif i == 1:
                weight = 0.2
            elif i == 2:
                weight = 0.2
            else:  # 最深层
                weight = max(0.1, 0.2 * epoch_ratio)
            aux_weights.append(weight)

        # 归一化权重
        weight_sum = sum(aux_weights)
        aux_weights = [w / weight_sum * 0.4 for w in aux_weights]  # 总辅助权重为0.4

        # 计算辅助损失
        aux_loss = 0
        aux_losses = []
        for i, (aux_output, weight) in enumerate(zip(aux_outputs, aux_weights)):
            loss = self._combined_loss(aux_output, targets)
            aux_loss += weight * loss
            aux_losses.append(loss.item())

        # 总损失（主损失权重0.6）
        total_loss = 0.6 * main_loss + aux_loss

        # 记录
        loss_dict = {
            'total': total_loss.item(),
            'main': main_loss.item(),
            'aux': aux_loss.item(),
            'epoch_ratio': epoch_ratio,
            'aux_weights': aux_weights
        }

        for i, loss in enumerate(aux_losses):
            loss_dict[f'aux_{i}'] = loss

        return total_loss, loss_dict