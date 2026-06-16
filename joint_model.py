import torch
import torch.nn as nn
import torch.nn.functional as F
from RegModel.reg_network import SwinInternRegNet
from RegModel.VoxelMorph import VoxelMorph
from RegModel.TransMorph import create_transmorph_model
from SAMModel.sam_feature_adaptor import SAMFeatureAdaptor, DeepSupervisionLoss

# ===================== 边缘加权 IoU 损失 =====================
class EdgeWeightedIoULoss(nn.Module):
    def __init__(self, edge_weight=2.0, device='cuda'):
        super().__init__()
        self.edge_weight = edge_weight
        self.device = device
        # 边缘检测卷积核，修改为浮点型张量，适配卷积层权重类型
        self.edge_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False).to(device)
        # 方案1：直接定义浮点数值（推荐）
        kernel = torch.tensor([[[[-1.0, -1.0, -1.0],
                                 [-1.0,  8.0, -1.0],
                                 [-1.0, -1.0, -1.0]]]],
                             dtype=torch.float32,  # 强制指定浮点类型
                             device=device)
        self.edge_conv.weight.data = kernel
        # 冻结核参数，禁止训练更新
        self.edge_conv.requires_grad_(False)

    def forward(self, warped_seg, target_seg):
        """
        Args:
            warped_seg: 扭曲后的SAM伪标签 (B, C, H, W)
            target_seg: 固定图像真实标签 (B, C, H, W) → 支持单通道（二分类）/多通道（多分类）
        Return:
            边缘加权IoU损失（越小越好）
        """
        B, C, H, W = warped_seg.shape

        # 1. 统一处理为概率分布（适配SAM输出logits/概率）
        if C > 1:
            warped_seg = F.softmax(warped_seg, dim=1)
            # 真实标签若为单通道类别编码，转为one-hot
            if target_seg.shape[1] == 1:
                target_seg = F.one_hot(target_seg.long().squeeze(1), num_classes=C).permute(0,3,1,2).float()
        else:
            warped_seg = torch.sigmoid(warped_seg)
            target_seg = target_seg.float()

        # 2. 提取真实标签的边缘（只基于真实标签，避免伪标签噪声干扰）
        target_edge = self.edge_conv(target_seg.sum(dim=1, keepdim=True))  # (B,1,H,W)
        target_edge = torch.abs(target_edge)  # 取绝对值，突出边缘
        target_edge = (target_edge > 0.1).float()  # 二值化，只保留强边缘

        # 3. 计算加权IoU（边缘区域权重翻倍，背景权重1）
        weight_map = 1 + self.edge_weight * target_edge  # (B,1,H,W)
        # 交、并集都乘权重，聚焦边缘区域
        intersection = (warped_seg * target_seg * weight_map).sum(dim=[2,3])  # (B,C)
        union = (warped_seg + target_seg - warped_seg * target_seg) * weight_map
        union = union.sum(dim=[2,3])  # (B,C)

        # 避免除零，计算IoU
        iou = (intersection + 1e-8) / (union + 1e-8)
        return 1 - iou.mean()  # 损失=1-IoU，越小表示对齐越好

# ===================== 模型主体 =====================
class SwallowReg(nn.Module):
    def __init__(self,
                 sam_config,
                 reg_config,
                 beta=0.5,
                 device='cuda',
                 training_mode='joint',
                 use_edge_iou_loss=True,
                 use_feature=True,
                 fusion_mode=None,
                 reg_net='swin_intern'):
        super().__init__()

        self.beta = beta
        self.device = device
        self.training_mode = training_mode
        self.use_edge_iou_loss = use_edge_iou_loss
        self.reg_net_name = reg_net

        # 特征融合模式（对应论文 Table 1 消融实验）。
        # 若未显式指定 fusion_mode，则按旧参数 use_feature 推导以保持向后兼容：
        #   use_feature=True  -> 'full'（融合 SAM 特征，完整方案）
        #   use_feature=False -> 'enc_only'（仅配准编码器特征）
        if fusion_mode is None:
            fusion_mode = 'full' if use_feature else 'enc_only'
        self.fusion_mode = fusion_mode
        self.use_feature = use_feature

        print(f"模型训练模式: {self.training_mode}")
        print(f"配准网络: {self.reg_net_name}")
        print(f"特征融合模式: {self.fusion_mode}")

        # 1. SAM适配器
        self.sam_adapter = SAMFeatureAdaptor(
            model_type=sam_config['model_type'],
            checkpoint_path=sam_config.get('checkpoint_path'),
            image_size=sam_config.get('image_size', 224)
        ).to(device)

        # 2. 配准网络（可选 SwinInternRegNet / VoxelMorph / TransMorph）
        #    三者均接收 4 尺度 SAM 特征并支持 fusion_mode 接口；
        #    其中 VoxelMorph / TransMorph 为 baseline，仅支持 'full' 与 'enc_only'。
        reg_channels = dict(S1Channel=128, S2Channel=256, S3Channel=512, S4Channel=1024,
                            Start_Channel=32)
        if reg_net == 'swin_intern':
            self.reg_net = SwinInternRegNet(**reg_channels).to(device)
        elif reg_net == 'voxelmorph':
            self.reg_net = VoxelMorph(**reg_channels, img_size=(256, 256)).to(device)
        elif reg_net == 'transmorph':
            self.reg_net = create_transmorph_model(**reg_channels, img_size=(256, 256)).to(device)
        else:
            raise ValueError(
                f"Unknown reg_net '{reg_net}'. "
                f"Choose from 'swin_intern', 'voxelmorph', 'transmorph'."
            )

        # 校验 fusion_mode 是否被所选配准网络支持
        if self.fusion_mode not in type(self.reg_net).FUSION_MODES:
            raise ValueError(
                f"reg_net='{reg_net}' does not support fusion_mode='{self.fusion_mode}'. "
                f"Supported: {type(self.reg_net).FUSION_MODES}."
            )

        # 3. 空间变换层
        from Utils.utils import SpatialTransform
        self.spatial_transform = SpatialTransform().to(device)

        # 4. 分割损失函数
        self.seg_criterion = DeepSupervisionLoss(
            main_weight=0.6,
            aux_weights=[0.1, 0.1, 0.1, 0.1],
            dice_weight=0.7,
            bce_weight=0.3
        ).to(device)

        # 5. 配准损失函数
        from Utils.utils import MSE, smoothloss
        self.reg_similarity_loss = MSE().loss
        self.reg_smooth_loss = smoothloss
        self.Anatomy_loss = EdgeWeightedIoULoss(edge_weight=2.0, device=device).to(device)  # 边缘加权 IoU 损失

        # 6. 设置训练模式
        self._setup_training_mode(training_mode)

    def _setup_training_mode(self, training_mode):
        print(f"设置训练模式: {training_mode}")
        if training_mode == 'seg_only':
            for param in self.reg_net.parameters():
                param.requires_grad = False
            print("✅ 只训练SAM适配器")
        elif training_mode == 'reg_only':
            for param in self.sam_adapter.parameters():
                param.requires_grad = False
            print("✅ 只训练配准网络")
        elif training_mode == 'joint':
            print("✅ 联合训练SAM和配准网络")
        else:
            raise ValueError(f"未知的训练模式: {training_mode}")

    def get_trainable_parameters(self):
        params = []
        for param in self.parameters():
            if param.requires_grad:
                params.append(param)
        return params

    def forward(self, mov_img_sam, movlab_sam, mov_img_reg, fix_img_reg, movlab_reg, fixlab_reg, training=True):
        seg_output = None
        seg_aux_outputs = None
        seg_loss = torch.tensor(0.0, device=self.device)
        seg_loss_dict = {}

        if training:
            if self.training_mode != 'reg_only':
                seg_output, seg_aux_outputs = self.sam_adapter(mov_img_sam, deep_supervision=True)
                if movlab_sam is not None:
                    seg_loss, seg_loss_dict = self.seg_criterion(seg_output, seg_aux_outputs, movlab_sam)
        else:
            seg_output, seg_aux_outputs = self.sam_adapter(mov_img_sam, deep_supervision=True)

        # 配准分支
        mov_features = self.sam_adapter(mov_img_reg, return_pyramid_features=True)
        fix_features = self.sam_adapter(fix_img_reg, return_pyramid_features=True)
        score0 = torch.cat((mov_features[0], fix_features[0]), dim=1)
        score1 = torch.cat((mov_features[1], fix_features[1]), dim=1)
        score2 = torch.cat((mov_features[2], fix_features[2]), dim=1)
        score3 = torch.cat((mov_features[3], fix_features[3]), dim=1)
        flow = self.reg_net(mov_img_reg, fix_img_reg, score0, score1, score2, score3, fusion_mode=self.fusion_mode)

        # 空间变换
        _, wrap_seg = self.spatial_transform(movlab_reg, flow.permute(0, 2, 3, 1))
        _, wrap_img = self.spatial_transform(mov_img_reg, flow.permute(0, 2, 3, 1))

        # 基础配准损失
        sim_loss = self.reg_similarity_loss(wrap_img, fix_img_reg)
        smooth_loss = self.reg_smooth_loss(flow)

        # 边缘加权 IoU 损失：用 SAM 分割伪标签经形变场扭曲后与固定图像标签对齐
        edge_iou_loss = torch.tensor(0.0, device=self.device)
        if self.use_edge_iou_loss and seg_output is not None:
            seg_output = F.interpolate(seg_output, size=(256,256), mode='nearest')
            _, wrap_sam_seg = self.spatial_transform(seg_output, flow.permute(0, 2, 3, 1))
            edge_iou_loss = self.Anatomy_loss(wrap_sam_seg, fixlab_reg)


        reg_loss = sim_loss  + 0.01 * smooth_loss

        reg_loss_dict = {
            'sim': sim_loss.item(),
            'smooth': smooth_loss.item(),
            'total': reg_loss.item(),
            'edge_iou': edge_iou_loss.item()
        }

        if training:
            if self.training_mode == 'joint':
                if self.use_edge_iou_loss:
                    total_loss = self.beta * seg_loss + self.beta * reg_loss + 0.5*self.beta * edge_iou_loss
                else:
                    total_loss = self.beta * seg_loss + self.beta * reg_loss
            elif self.training_mode == 'reg_only':
                total_loss = reg_loss
            else:
                total_loss = seg_loss

            results = {
                'flow': flow,
                'wrap_seg': wrap_seg,
                'wrap_img': wrap_img,
                'seg_output': seg_output,
                'seg_aux_outputs': seg_aux_outputs,
                'seg_loss': seg_loss.item(),
                'reg_loss': reg_loss.item(),
                'total_loss': total_loss.item(),
                'seg_loss_dict': seg_loss_dict,
                'reg_loss_dict': reg_loss_dict
            }
            return total_loss, results
        else:
            results = {
                'flow': flow,
                'wrap_seg': wrap_seg,
                'wrap_img': wrap_img,
                'seg_out': seg_output
            }
            return results