#!/usr/bin/env python3
# test_joint_model.py
"""
独立测试联合SAM分割和配准网络
加载最佳模型并在测试集上评估
"""

import argparse
import os

# 设置环境变量，禁用oneDNN和TensorFlow日志
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=显示所有, 1=INFO, 2=WARNING, 3=ERROR
import sys
import glob
import yaml
import torch
import numpy as np
from pathlib import Path
import warnings
import time
import csv
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import nibabel as nib

warnings.filterwarnings('ignore')

# 添加当前目录到路径
sys.path.append(str(Path(__file__).parent))

# 导入模型
from joint_model import SwallowReg
# 导入数据加载器
from Utils.utils import Dataset_epoch_with_name, jacobian_determinant_vxm


class JointSAMRegistrationTester:
    """独立的测试器，用于加载最佳模型并进行全面评估"""

    def __init__(self, config, checkpoint_path):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"初始化测试器，设备: {self.device}")
        print(f"加载检查点: {checkpoint_path}")

        # 导入模型
        from joint_model import SwallowReg

        # 创建模型
        self.model = SwallowReg(
            sam_config=config['sam_config'],
            reg_config=config['reg_config'],
            beta=config.get('beta', 0.5),
            device=self.device,
            training_mode='joint',  # 测试模式
            use_edge_iou_loss = config.get('use_edge_iou_loss', config.get('use_sam_label_loss', True)),
            use_feature = config.get('use_feature', True),
            fusion_mode = config.get('fusion_mode', None),
            reg_net = config.get('reg_net', 'swin_intern')
        )

        # 加载检查点
        self.load_checkpoint(checkpoint_path)

        # 将模型设置为评估模式
        self.model.eval()

        print("✅ 测试器初始化完成！")

    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"检查点不存在: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device,weights_only=False)

        # 处理可能的键名不匹配
        model_state_dict = checkpoint['model_state_dict']

        # 移除可能的'module.'前缀（如果模型是在DataParallel中训练的）
        new_state_dict = {}
        for k, v in model_state_dict.items():
            if k.startswith('module.'):
                new_key = k[7:]  # 移除'module.'
            else:
                new_key = k
            new_state_dict[new_key] = v

        # 加载模型权重
        self.model.load_state_dict(new_state_dict, strict=False)

        print(f"✅ 成功加载模型检查点: {checkpoint_path}")

        # 获取最佳Dice分数
        if 'best_val_dice' in checkpoint:
            self.best_val_dice = checkpoint['best_val_dice']
            print(f"   最佳验证Dice: {self.best_val_dice:.4f}")

        if 'epoch' in checkpoint:
            self.trained_epochs = checkpoint['epoch'] + 1
            print(f"   训练轮数: {self.trained_epochs}")

    def dice_coefficient(self, predictions, targets, threshold=0.5):
        """计算Dice系数"""
        predictions = (predictions > threshold).float()
        targets = targets.float()

        intersection = (predictions * targets).sum()
        dice = (2. * intersection) / (predictions.sum() + targets.sum() + 1e-8)
        return dice.item()

    def gncc(self, I, J, eps=1e-5):
        """计算归一化互相关"""
        I = I - I.mean()
        J = J - J.mean()
        cross = torch.sum(I * J)
        I_var = torch.sum(I ** 2)
        J_var = torch.sum(J ** 2)
        return cross / (torch.sqrt(I_var) * torch.sqrt(J_var) + eps)

    def mk_grid_img(self, grid_step=8, line_thickness=1, grid_sz=(256, 256)):
        """创建网格图像用于可视化变形"""
        grid_img = np.zeros(grid_sz)
        for j in range(0, grid_img.shape[0], grid_step):
            grid_img[j:j + line_thickness, :] = 1
        for i in range(0, grid_img.shape[1], grid_step):
            grid_img[:, i:i + line_thickness] = 1
        grid_img = grid_img[None, None, ...]
        return torch.from_numpy(grid_img).to(self.device)

    def evaluate(self, test_data_dir, save_outputs=True, output_dir=None):
        """
        综合评估模型
        Args:
            test_data_dir: 测试数据目录
            save_outputs: 是否保存输出结果
            output_dir: 输出目录
        Returns:
            评估结果字典
        """
        if not os.path.exists(test_data_dir):
            print(f"❌ 错误: 测试目录不存在: {test_data_dir}")
            return None

        # 获取测试文件
        test_paths = glob.glob(os.path.join(test_data_dir, "*.npz"))

        if len(test_paths) == 0:
            print(f"❌ 错误: 在 {test_data_dir} 中没有找到npz文件")
            return None

        print(f"✅ 找到 {len(test_paths)} 个测试样本")

        # 设置输出目录
        if output_dir is None:
            output_dir = os.path.join("test_results", os.path.basename(os.path.normpath(test_data_dir)))

        os.makedirs(output_dir, exist_ok=True)

        # 创建测试数据集
        test_dataset = Dataset_epoch_with_name(
            test_paths,
            sam_image_size=self.config['sam_config']['image_size'],
            reg_image_size=self.config.get('reg_image_size', 256),
            augment=False,
            debug=False
        )

        # 创建数据加载器
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2
        )

        # 初始化指标收集
        metrics = {
            'samples': [],
            'dice_def': [],
            'dice_raw': [],
            'gncc_def': [],
            'gncc_raw': [],
            'ssim_def': [],
            'ssim_raw': [],
            'jacobian_neg': [],
            'inference_time': []
        }

        print("\n" + "=" * 60)
        print("开始评估...")
        print("=" * 60)

        with torch.no_grad():
            pbar = tqdm(test_loader, desc='测试评估')

            for batch_idx, batch in enumerate(pbar):
                # 获取数据
                mov_sam = batch['movimg_sam'].to(self.device).float()
                fix_sam = batch['tarimg_sam'].to(self.device).float()
                mov_reg = batch['movimg_reg'].to(self.device).float()
                fix_reg = batch['tarimg_reg'].to(self.device).float()
                movlab_sam = batch['movlab_sam'].to(self.device).float()
                fixlab_sam = batch['tarlab_sam'].to(self.device).float()
                movlab_reg = batch['movlab_reg'].to(self.device).float()
                fixlab_reg = batch['tarlab_reg'].to(self.device).float()

                # 获取文件名
                filename = batch.get('filename', [f'sample_{batch_idx:03d}'])[0]
                sample_name = os.path.splitext(os.path.basename(filename))[0]
                metrics['samples'].append(sample_name)

                # 计时
                start_time = time.time()

                # 前向传播
                results = self.model(
                    mov_img_sam=mov_sam,
                    movlab_sam=movlab_sam,
                    mov_img_reg=mov_reg,
                    fix_img_reg=fix_reg,
                    movlab_reg=movlab_reg,
                    fixlab_reg=fixlab_reg,
                    training=False
                )

                end_time = time.time()
                inference_time = end_time - start_time
                metrics['inference_time'].append(inference_time)

                # 获取输出
                flow = results['flow']
                wrap_seg = results.get('wrap_seg')
                wrap_img = results.get('wrap_img')

                # 如果没有直接输出wrap_seg，通过spatial_transform计算
                if wrap_seg is None:
                    wrap_seg = self.model.spatial_transform(
                        movlab_reg,
                        flow.permute(0, 2, 3, 1),
                        mod='nearest'
                    )

                # 如果没有直接输出wrap_img，通过spatial_transform计算
                if wrap_img is None:
                    wrap_img = self.model.spatial_transform(mov_reg, flow.permute(0, 2, 3, 1))

                # 1. 计算Dice系数
                dice_def = self.dice_coefficient(wrap_seg, fixlab_reg)
                dice_raw = self.dice_coefficient(movlab_reg, fixlab_reg)
                metrics['dice_def'].append(dice_def)
                metrics['dice_raw'].append(dice_raw)

                # 2. 计算GNCC
                gncc_def = self.gncc(wrap_img.float(), fix_reg.float()).item()
                gncc_raw = self.gncc(mov_reg.float(), fix_reg.float()).item()
                metrics['gncc_def'].append(gncc_def)
                metrics['gncc_raw'].append(gncc_raw)

                # 3. 计算SSIM
                mov_img_np = mov_reg[0, 0].cpu().numpy()
                fix_img_np = fix_reg[0, 0].cpu().numpy()
                wrap_img_np = wrap_img[0, 0].cpu().numpy()

                ssim_def = ssim(wrap_img_np, fix_img_np, data_range=1.0)
                ssim_raw = ssim(mov_img_np, fix_img_np, data_range=1.0)
                metrics['ssim_def'].append(ssim_def)
                metrics['ssim_raw'].append(ssim_raw)

                # 4. 计算雅可比行列式（负值比例）
                flow_cpu = flow.cpu().numpy()
                jac_det = jacobian_determinant_vxm(flow_cpu[0])
                jac_neg_ratio = np.sum(jac_det <= 0) / np.prod(jac_det.shape)
                metrics['jacobian_neg'].append(jac_neg_ratio)

                # 创建变形网格
                grid_img = self.mk_grid_img(grid_step=8, grid_sz=(256, 256))
                _, deformed_grid = self.model.spatial_transform(
                    grid_img.float(),
                    flow.permute(0, 2, 3, 1)
                )

                # 更新进度条
                pbar.set_postfix({
                    'Dice': f'{dice_def:.3f}',
                    'SSIM': f'{ssim_def:.3f}',
                    'Time': f'{inference_time:.2f}s'
                })

                # 保存输出结果
                if save_outputs:
                    self._save_sample_outputs(
                        sample_name=sample_name,
                        output_dir=output_dir,
                        mov_img=mov_reg[0, 0].cpu().numpy(),
                        fix_img=fix_reg[0, 0].cpu().numpy(),
                        mov_seg=movlab_reg[0, 0].cpu().numpy(),
                        fix_seg=fixlab_reg[0, 0].cpu().numpy(),
                        wrap_img=wrap_img[0, 0].cpu().numpy(),
                        wrap_seg=wrap_seg[0, 0].cpu().numpy(),
                        flow=flow[0, 0].cpu().numpy(),
                        deformed_grid=deformed_grid[0, 0].cpu().numpy(),
                        jacobian=jac_det,
                        sample_metrics={
                            'dice_def': dice_def,
                            'dice_raw': dice_raw,
                            'gncc_def': gncc_def,
                            'gncc_raw': gncc_raw,
                            'ssim_def': ssim_def,
                            'ssim_raw': ssim_raw,
                            'jacobian_neg': jac_neg_ratio,
                            'inference_time': inference_time
                        }
                    )

        # 计算总体统计
        overall_stats = self._compute_overall_statistics(metrics)

        # 保存评估结果
        self._save_evaluation_results(metrics, overall_stats, output_dir)

        # 打印评估摘要
        self._print_evaluation_summary(overall_stats, len(test_paths))

        return {
            'metrics': metrics,
            'overall_stats': overall_stats,
            'output_dir': output_dir
        }

    def _save_sample_outputs(self, sample_name, output_dir, mov_img, fix_img, mov_seg, fix_seg,
                             wrap_img, wrap_seg, flow, deformed_grid, jacobian, sample_metrics):
        """保存单个样本的所有输出结果"""
        sample_dir = os.path.join(output_dir, 'samples', sample_name)
        os.makedirs(sample_dir, exist_ok=True)

        # 保存为numpy文件
        np.savez_compressed(
            os.path.join(sample_dir, f'{sample_name}_results.npz'),
            mov_img=mov_img,
            fix_img=fix_img,
            mov_seg=mov_seg,
            fix_seg=fix_seg,
            wrap_img=wrap_img,
            wrap_seg=wrap_seg,
            flow=flow,
            deformed_grid=deformed_grid,
            jacobian=jacobian
        )

        # 创建NIfTI图像（2D图像扩展为3D）
        def create_nifti(data_2d):
            # 添加z维度
            data_3d = data_2d[np.newaxis, :, :]
            return nib.Nifti1Image(data_3d, np.eye(4))

        # 保存NIfTI文件
        nib.save(create_nifti(mov_img), os.path.join(sample_dir, 'moving_image.nii.gz'))
        nib.save(create_nifti(fix_img), os.path.join(sample_dir, 'fixed_image.nii.gz'))
        nib.save(create_nifti(wrap_img), os.path.join(sample_dir, 'warped_image.nii.gz'))
        nib.save(create_nifti(mov_seg), os.path.join(sample_dir, 'moving_segmentation.nii.gz'))
        nib.save(create_nifti(fix_seg), os.path.join(sample_dir, 'fixed_segmentation.nii.gz'))
        nib.save(create_nifti(wrap_seg), os.path.join(sample_dir, 'warped_segmentation.nii.gz'))
        nib.save(create_nifti(flow), os.path.join(sample_dir, 'flow_field.nii.gz'))
        nib.save(create_nifti(jacobian), os.path.join(sample_dir, 'jacobian_determinant.nii.gz'))
        nib.save(create_nifti(deformed_grid),os.path.join(sample_dir, 'deformed_grid.nii.gz'))

        # 保存指标为文本文件
        with open(os.path.join(sample_dir, 'metrics.txt'), 'w') as f:
            f.write(f"样本: {sample_name}\n")
            f.write("=" * 40 + "\n")
            f.write(f"配准后Dice: {sample_metrics['dice_def']:.4f}\n")
            f.write(f"原始Dice: {sample_metrics['dice_raw']:.4f}\n")
            f.write(f"Dice改进: {sample_metrics['dice_def'] - sample_metrics['dice_raw']:.4f}\n")
            f.write(f"配准后GNCC: {sample_metrics['gncc_def']:.4f}\n")
            f.write(f"原始GNCC: {sample_metrics['gncc_raw']:.4f}\n")
            f.write(f"配准后SSIM: {sample_metrics['ssim_def']:.4f}\n")
            f.write(f"原始SSIM: {sample_metrics['ssim_raw']:.4f}\n")
            f.write(f"雅可比负值比例: {sample_metrics['jacobian_neg']:.4f}\n")
            f.write(f"推理时间: {sample_metrics['inference_time']:.4f}秒\n")

        # 创建可视化图像
        self._create_visualization(
            sample_dir=sample_dir,
            sample_name=sample_name,
            mov_img=mov_img,
            fix_img=fix_img,
            mov_seg=mov_seg,
            fix_seg=fix_seg,
            wrap_img=wrap_img,
            wrap_seg=wrap_seg,
            deformed_grid=deformed_grid
        )

    def _create_visualization(self, sample_dir, sample_name, mov_img, fix_img, mov_seg, fix_seg,
                              wrap_img, wrap_seg, deformed_grid):
        """创建可视化图像"""
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))

        # 第一行：图像
        axes[0, 0].imshow(mov_img, cmap='gray')
        axes[0, 0].set_title('Moving Image')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(fix_img, cmap='gray')
        axes[0, 1].set_title('Fixed Image')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(wrap_img, cmap='gray')
        axes[0, 2].set_title('Warped Image')
        axes[0, 2].axis('off')

        diff_img = np.abs(fix_img - wrap_img)
        axes[0, 3].imshow(diff_img, cmap='hot', vmax=0.5)
        axes[0, 3].set_title('Difference')
        axes[0, 3].axis('off')

        axes[0, 4].imshow(deformed_grid, cmap='gray')
        axes[0, 4].set_title('Deformation Grid')
        axes[0, 4].axis('off')

        # 第二行：分割
        axes[1, 0].imshow(mov_seg, cmap='gray')
        axes[1, 0].set_title('Moving Seg')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(fix_seg, cmap='gray')
        axes[1, 1].set_title('Fixed Seg')
        axes[1, 1].axis('off')

        axes[1, 2].imshow(wrap_seg, cmap='gray')
        axes[1, 2].set_title('Warped Seg')
        axes[1, 2].axis('off')

        diff_seg = np.abs(fix_seg - wrap_seg)
        axes[1, 3].imshow(diff_seg, cmap='hot', vmax=1.0)
        axes[1, 3].set_title('Seg Difference')
        axes[1, 3].axis('off')

        # 最后一个子图可以展示雅可比行列式的直方图
        axes[1, 4].axis('off')

        plt.suptitle(f'Sample: {sample_name}', fontsize=16)
        plt.tight_layout()

        # 保存图像
        plt.savefig(os.path.join(sample_dir, 'visualization.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _compute_overall_statistics(self, metrics):
        """计算总体统计"""
        overall = {}

        # 为每个指标计算统计量
        metric_keys = ['dice_def', 'dice_raw', 'gncc_def', 'gncc_raw',
                       'ssim_def', 'ssim_raw', 'jacobian_neg', 'inference_time']

        for key in metric_keys:
            values = metrics.get(key, [])
            if values:
                overall[f'{key}_mean'] = np.mean(values)
                overall[f'{key}_std'] = np.std(values)
                overall[f'{key}_median'] = np.median(values)
                overall[f'{key}_min'] = np.min(values)
                overall[f'{key}_max'] = np.max(values)
            else:
                overall[f'{key}_mean'] = 0.0
                overall[f'{key}_std'] = 0.0
                overall[f'{key}_median'] = 0.0
                overall[f'{key}_min'] = 0.0
                overall[f'{key}_max'] = 0.0

        # 计算改进
        if metrics.get('dice_def') and metrics.get('dice_raw'):
            dice_improvement = np.array(metrics['dice_def']) - np.array(metrics['dice_raw'])
            overall['dice_improvement_mean'] = np.mean(dice_improvement)
            overall['dice_improvement_std'] = np.std(dice_improvement)
            overall['dice_improvement_min'] = np.min(dice_improvement)
            overall['dice_improvement_max'] = np.max(dice_improvement)

        return overall

    def _save_evaluation_results(self, metrics, overall_stats, output_dir):
        """保存评估结果到CSV文件"""
        csv_path = os.path.join(output_dir, 'evaluation_results.csv')

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # 写入详细结果
            writer.writerow(['详细结果'])
            writer.writerow(['样本', 'Dice(配准后)', 'Dice(原始)', 'GNCC(配准后)', 'GNCC(原始)',
                             'SSIM(配准后)', 'SSIM(原始)', '雅可比负值比例', '推理时间(秒)'])

            for i, sample in enumerate(metrics['samples']):
                writer.writerow([
                    sample,
                    f"{metrics['dice_def'][i]:.4f}" if i < len(metrics['dice_def']) else '',
                    f"{metrics['dice_raw'][i]:.4f}" if i < len(metrics['dice_raw']) else '',
                    f"{metrics['gncc_def'][i]:.4f}" if i < len(metrics['gncc_def']) else '',
                    f"{metrics['gncc_raw'][i]:.4f}" if i < len(metrics['gncc_raw']) else '',
                    f"{metrics['ssim_def'][i]:.4f}" if i < len(metrics['ssim_def']) else '',
                    f"{metrics['ssim_raw'][i]:.4f}" if i < len(metrics['ssim_raw']) else '',
                    f"{metrics['jacobian_neg'][i]:.4f}" if i < len(metrics['jacobian_neg']) else '',
                    f"{metrics['inference_time'][i]:.4f}" if i < len(metrics['inference_time']) else ''
                ])

            writer.writerow([])

            # 写入总体统计
            writer.writerow(['总体统计'])
            writer.writerow(['指标', '均值', '标准差', '中位数', '最小值', '最大值'])

            stat_keys = ['dice_def', 'dice_raw', 'gncc_def', 'gncc_raw',
                         'ssim_def', 'ssim_raw', 'jacobian_neg', 'inference_time']

            for key in stat_keys:
                writer.writerow([
                    key.replace('_', ' ').title(),
                    f"{overall_stats.get(f'{key}_mean', 0):.4f}",
                    f"{overall_stats.get(f'{key}_std', 0):.4f}",
                    f"{overall_stats.get(f'{key}_median', 0):.4f}",
                    f"{overall_stats.get(f'{key}_min', 0):.4f}",
                    f"{overall_stats.get(f'{key}_max', 0):.4f}"
                ])

        print(f"✅ 评估结果已保存到: {csv_path}")

        # 保存总体报告
        report_path = os.path.join(output_dir, 'summary_report.txt')
        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("        模型评估总结报告\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"测试样本数: {len(metrics['samples'])}\n")
            f.write(f"评估时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("-" * 60 + "\n")
            f.write("主要指标统计:\n")
            f.write("-" * 60 + "\n")
            f.write(
                f"Dice系数 (配准后): {overall_stats.get('dice_def_mean', 0):.4f} ± {overall_stats.get('dice_def_std', 0):.4f}\n")
            f.write(
                f"Dice系数 (原始):   {overall_stats.get('dice_raw_mean', 0):.4f} ± {overall_stats.get('dice_raw_std', 0):.4f}\n")

            if 'dice_improvement_mean' in overall_stats:
                f.write(
                    f"Dice改进:           {overall_stats['dice_improvement_mean']:.4f} ± {overall_stats['dice_improvement_std']:.4f}\n")

            f.write(
                f"SSIM (配准后):      {overall_stats.get('ssim_def_mean', 0):.4f} ± {overall_stats.get('ssim_def_std', 0):.4f}\n")
            f.write(
                f"SSIM (原始):        {overall_stats.get('ssim_raw_mean', 0):.4f} ± {overall_stats.get('ssim_raw_std', 0):.4f}\n")
            f.write(
                f"GNCC (配准后):      {overall_stats.get('gncc_def_mean', 0):.4f} ± {overall_stats.get('gncc_def_std', 0):.4f}\n")
            f.write(
                f"GNCC (原始):        {overall_stats.get('gncc_raw_mean', 0):.4f} ± {overall_stats.get('gncc_raw_std', 0):.4f}\n")
            f.write(
                f"雅可比负值比例:     {overall_stats.get('jacobian_neg_mean', 0):.4f} ± {overall_stats.get('jacobian_neg_std', 0):.4f}\n")
            f.write(
                f"平均推理时间:       {overall_stats.get('inference_time_mean', 0):.4f} ± {overall_stats.get('inference_time_std', 0):.4f}秒\n")
            f.write("=" * 60 + "\n")

        print(f"✅ 总结报告已保存到: {report_path}")

    def _print_evaluation_summary(self, overall_stats, num_samples):
        """打印评估摘要"""
        print("\n" + "=" * 60)
        print("评估结果总结")
        print("=" * 60)
        print(f"测试样本数: {num_samples}")
        print("\n主要指标:")
        print(
            f"  Dice系数 (配准后): {overall_stats.get('dice_def_mean', 0):.4f} ± {overall_stats.get('dice_def_std', 0):.4f}")
        print(
            f"  Dice系数 (原始):   {overall_stats.get('dice_raw_mean', 0):.4f} ± {overall_stats.get('dice_raw_std', 0):.4f}")

        if 'dice_improvement_mean' in overall_stats:
            print(
                f"  Dice改进:           {overall_stats['dice_improvement_mean']:.4f} ± {overall_stats['dice_improvement_std']:.4f}")

        print(
            f"  SSIM (配准后):      {overall_stats.get('ssim_def_mean', 0):.4f} ± {overall_stats.get('ssim_def_std', 0):.4f}")
        print(
            f"  SSIM (原始):        {overall_stats.get('ssim_raw_mean', 0):.4f} ± {overall_stats.get('ssim_raw_std', 0):.4f}")
        print(
            f"  GNCC (配准后):      {overall_stats.get('gncc_def_mean', 0):.4f} ± {overall_stats.get('gncc_def_std', 0):.4f}")
        print(
            f"  GNCC (原始):        {overall_stats.get('gncc_raw_mean', 0):.4f} ± {overall_stats.get('gncc_raw_std', 0):.4f}")
        print(
            f"  雅可比负值比例:     {overall_stats.get('jacobian_neg_mean', 0):.4f} ± {overall_stats.get('jacobian_neg_std', 0):.4f}")
        print(
            f"  平均推理时间:       {overall_stats.get('inference_time_mean', 0):.4f} ± {overall_stats.get('inference_time_std', 0):.4f}秒")
        print("=" * 60)


def load_config_from_experiment(experiment_dir, checkpoint_path=None):
    """从实验目录加载配置。

    优先读取 config.yaml；若不存在，则回退到从模型 checkpoint 内保存的 config 读取
    （训练时已将 config 一并写入 checkpoint），从而无需额外的 config.yaml 也能测试。
    """
    config_path = os.path.join(experiment_dir, "config.yaml")

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    # 回退：从 checkpoint 内嵌的 config 读取
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if isinstance(ckpt, dict) and 'config' in ckpt:
            print(f"⚠️ 未找到 config.yaml，改用 checkpoint 内保存的配置: {checkpoint_path}")
            return ckpt['config']

    print(f"❌ 错误: 既无 config.yaml，也无法从 checkpoint 读取配置: {experiment_dir}")
    return None


def find_best_model(experiment_dir):
    """在实验目录中查找最佳模型"""
    # 检查最佳模型文件
    best_model_path = os.path.join(experiment_dir, "best_model.pth")

    if os.path.exists(best_model_path):
        return best_model_path

    # 检查最新检查点
    latest_model_path = os.path.join(experiment_dir, "latest_checkpoint.pth")

    if os.path.exists(latest_model_path):
        print(f"⚠️ 未找到最佳模型，使用最新检查点: {latest_model_path}")
        return latest_model_path

    # 查找任何检查点文件
    checkpoint_files = glob.glob(os.path.join(experiment_dir, "*.pth"))

    if checkpoint_files:
        # 按修改时间排序，取最新的
        checkpoint_files.sort(key=os.path.getmtime, reverse=True)
        print(f"⚠️ 使用最新的检查点文件: {checkpoint_files[0]}")
        return checkpoint_files[0]

    return None


def main():
    parser = argparse.ArgumentParser(description='独立测试联合SAM分割和配准网络')

    # 模型相关参数
    parser.add_argument('--experiment_dir', type=str, default=None,
                        help='实验目录路径，包含训练好的模型和配置（必填）')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='模型检查点路径，如不指定则自动查找最佳模型')

    # 数据相关参数
    parser.add_argument('--test_data_dir', type=str, default=None,
                        help='测试数据目录路径（必填）')

    # 输出相关参数
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录路径；不指定时自动生成到 Test_results/ 下（按实验名命名），'
                             '与训练模型分开存放')
    parser.add_argument('--no_save_outputs', action='store_true',
                        help='不保存输出结果（仅计算指标）')
    parser.add_argument('--no_visualization', action='store_true',
                        help='不创建可视化图像')

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("联合SAM分割和配准模型测试")
    print("=" * 60)

    # 1. 检查实验目录
    if not os.path.exists(args.experiment_dir):
        print(f"❌ 错误: 实验目录不存在: {args.experiment_dir}")
        sys.exit(1)

    # 2. 查找或指定模型检查点
    if args.checkpoint:
        model_path = args.checkpoint
        if not os.path.exists(model_path):
            print(f"❌ 错误: 指定的检查点不存在: {model_path}")
            sys.exit(1)
    else:
        print("\n🔍 查找最佳模型...")
        model_path = find_best_model(args.experiment_dir)
        if model_path is None:
            print(f"❌ 错误: 在实验目录中找不到模型文件: {args.experiment_dir}")
            sys.exit(1)

    print(f"✅ 使用模型: {model_path}")

    # 3. 加载配置（缺 config.yaml 时回退到 checkpoint 内保存的配置）
    print("\n🔧 加载配置...")
    config = load_config_from_experiment(args.experiment_dir, checkpoint_path=model_path)
    if config is None:
        sys.exit(1)

    # 4. 检查测试数据
    if not os.path.exists(args.test_data_dir):
        print(f"❌ 错误: 测试数据目录不存在: {args.test_data_dir}")
        sys.exit(1)

    # 5. 设置输出目录：不指定时自动放到 Test_results/ 下，与训练模型分开存放
    if args.output_dir is None:
        parts = os.path.normpath(args.experiment_dir).split(os.sep)
        exp_tag = "_".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
        args.output_dir = os.path.join("Test_results", exp_tag)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"✅ 输出目录: {args.output_dir}")

    # 6. 创建测试器
    print("\n🚀 创建测试器...")
    try:
        tester = JointSAMRegistrationTester(config, model_path)
    except Exception as e:
        print(f"❌ 创建测试器失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 7. 运行测试
    print("\n" + "=" * 60)
    print("开始测试评估")
    print("=" * 60)

    try:
        results = tester.evaluate(
            test_data_dir=args.test_data_dir,
            save_outputs=not args.no_save_outputs,
            output_dir=args.output_dir
        )

        if results:
            print("\n🎉 测试评估完成！")
            print(f"📁 结果保存到: {results['output_dir']}")
        else:
            print("\n❌ 测试评估失败！")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⏹️ 测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n✨ 测试脚本完成！")


if __name__ == '__main__':
    main()