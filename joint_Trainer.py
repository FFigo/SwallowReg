import torch
import torch.optim as optim
from natsort import natsorted
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import glob


class JointSAMRegistrationTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"初始化联合训练器，设备: {self.device}")

        # 导入模型
        from joint_model import SwallowReg

        # 创建模型
        self.model = SwallowReg(
            sam_config=config['sam_config'],
            reg_config=config['reg_config'],
            beta=config.get('beta', 0.5),
            device=self.device,
            training_mode=config.get('training_mode', 'joint'),
            use_edge_iou_loss=config.get('use_edge_iou_loss', config.get('use_sam_label_loss', True)),
            use_feature=config.get('use_feature', True),
            fusion_mode=config.get('fusion_mode', None),
            reg_net=config.get('reg_net', 'swin_intern')
        )

        # 打印模型参数
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"模型总参数: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")

        # 优化器
        trainable_params = self.model.get_trainable_parameters()
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-5)
        )

        # 学习率调度器
        # 验证 Dice 停滞时按 factor 衰减学习率
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',  # 因为监控的是Dice（越大越好），模式设为max
            factor=0.5,  # 学习率衰减因子，连续不提升则乘以0.5
            patience=10,  # 连续10轮验证Dice不提升则衰减
            min_lr=config.get('min_lr', 1e-6)  # 学习率下限
        )

        # 创建数据加载器
        from Utils.utils import Dataset_epoch_with_name

        train_paths = glob.glob(config['train_data_dir'] + '/*.npz')
        val_paths = glob.glob(config['val_data_dir'] + '/*.npz')

        print(f"训练样本数: {len(train_paths)}")
        print(f"验证样本数: {len(val_paths)}")

        train_dataset = Dataset_epoch_with_name(
            train_paths,
            sam_image_size=config['sam_config']['image_size'],
            reg_image_size=config.get('reg_image_size', 256),
            augment=True,
            debug=True
        )

        val_dataset = Dataset_epoch_with_name(
            val_paths,
            sam_image_size=config['sam_config']['image_size'],
            reg_image_size=config.get('reg_image_size', 256),
            augment=False,
            debug=False
        )

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config.get('num_workers', 4),
            pin_memory=True
        )
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=config.get('num_workers', 2),
            pin_memory=True
        )

        # 创建保存目录
        os.makedirs(config['save_dir'], exist_ok=True)
        os.makedirs(config['log_dir'], exist_ok=True)

        # TensorBoard
        self.writer = SummaryWriter(config['log_dir'])

        # 训练历史
        self.train_history = {
            'epoch': [],
            'total_loss': [],
            'seg_loss': [],
            'reg_loss': [],
            'val_dice': [],
            'val_org_dice': [],
            'learning_rate': []
        }

        # CSV日志文件
        self.csv_path = os.path.join(config['save_dir'], 'training_log.csv')
        self._init_csv_log()

        # 最佳模型
        self.best_val_dice = 0.0
        self.best_epoch = 0
        self.start_epoch = 0  # 添加这行！

        print(f"联合训练器初始化完成！")
        print(f"保存目录: {config['save_dir']}")
        print(f"日志目录: {config['log_dir']}")

    def _init_csv_log(self):
        """初始化CSV日志文件"""
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'train_loss', 'seg_loss', 'reg_loss',
                'val_dice', 'val_org_dice', 'learning_rate'
            ])

    def _log_to_csv(self, epoch, train_loss, seg_loss, reg_loss, val_dice, val_org_dice):
        """记录到CSV文件"""
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, train_loss, seg_loss, reg_loss,
                val_dice, val_org_dice, self.optimizer.param_groups[0]['lr']
            ])

    def dice_coefficient(self, predictions, targets, threshold=0.5):
        """计算Dice系数"""
        predictions = (predictions > threshold).float()
        targets = targets.float()

        intersection = (predictions * targets).sum()
        dice = (2. * intersection) / (predictions.sum() + targets.sum() + 1e-8)
        return dice.item()

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_seg_loss = 0.0
        total_reg_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{self.config["epochs"]} [Train]')

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



            # 前向传播
            self.optimizer.zero_grad()

            # 使用正确的标签
            # SAM分割使用224×224的标签
            # 配准使用256×256的标签
            loss, results = self.model(
                mov_img_sam=mov_sam,
                movlab_sam=movlab_sam,
                mov_img_reg=mov_reg,
                fix_img_reg=fix_reg,
                movlab_reg = movlab_reg,
                fixlab_reg = fixlab_reg,
                training=True
            )

            # 反向传播
            loss.backward()

            # 梯度裁剪
            if self.config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.get('grad_clip', 1.0)
                )

            self.optimizer.step()

            # 记录损失
            total_loss += results['total_loss']
            total_seg_loss += results['seg_loss']
            total_reg_loss += results['reg_loss']

            # 更新进度条
            pbar.set_postfix({
                'Loss': f"{results['total_loss']:.4f}",
                'Seg': f"{results['seg_loss']:.4f}",
                'Reg': f"{results['reg_loss']:.4f}",
                'LR': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })

            # 记录到TensorBoard
            step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Train/Total_Loss', results['total_loss'], step)
            self.writer.add_scalar('Train/Seg_Loss', results['seg_loss'], step)
            self.writer.add_scalar('Train/Reg_Loss', results['reg_loss'], step)
            self.writer.add_scalar('Train/Learning_Rate', self.optimizer.param_groups[0]['lr'], step)

        avg_loss = total_loss / len(self.train_loader)
        avg_seg_loss = total_seg_loss / len(self.train_loader)
        avg_reg_loss = total_reg_loss / len(self.train_loader)

        return avg_loss, avg_seg_loss, avg_reg_loss

    def validate(self, epoch):
        """验证"""
        self.model.eval()
        dice_scores = []
        org_dice_scores = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch + 1}/{self.config["epochs"]} [Val]')

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

                # 前向传播（不计算分割损失）
                results = self.model(
                    mov_img_sam=mov_sam,
                    movlab_sam=movlab_sam,
                    mov_img_reg=mov_reg,
                    fix_img_reg=fix_reg,
                    movlab_reg=movlab_reg,
                    fixlab_reg=fixlab_reg,
                    training=False
                )

                # 变形分割标签
                wrap_seg = results.get('wrap_seg')

                # 计算Dice系数
                dice_val = self.dice_coefficient(wrap_seg, fixlab_reg)
                org_dice_val = self.dice_coefficient(movlab_reg, fixlab_reg)

                dice_scores.append(dice_val)
                org_dice_scores.append(org_dice_val)

                # 更新进度条
                pbar.set_postfix({
                    'Dice': f'{dice_val:.4f}',
                    'Org': f'{org_dice_val:.4f}'
                })
                # 可视化第一个样本
                if batch_idx == 1 and epoch % self.config.get('viz_interval', 5) == 0:
                    self._visualize_batch(
                        epoch=epoch,
                        mov_img=mov_reg[0].cpu().numpy(),
                        fix_img=fix_reg[0].cpu().numpy(),
                        mov_seg=movlab_reg[0].cpu().numpy(),
                        fix_seg=fixlab_reg[0].cpu().numpy(),
                        seg_out=results.get('seg_out')[0].cpu().numpy(),
                        warped_seg=wrap_seg[0].cpu().numpy()
                    )

        avg_dice = np.mean(dice_scores) if dice_scores else 0.0
        avg_org_dice = np.mean(org_dice_scores) if org_dice_scores else 0.0

        # 记录到TensorBoard
        self.writer.add_scalar('Val/Dice', avg_dice, epoch)
        self.writer.add_scalar('Val/Org_Dice', avg_org_dice, epoch)

        return avg_dice, avg_org_dice

    def _visualize_batch(self, epoch, mov_img, fix_img, mov_seg, fix_seg, seg_out, warped_seg):
        """可视化一个batch的结果"""
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))

        # 第一行：图像
        axes[0, 0].imshow(mov_img[0], cmap='gray')
        axes[0, 0].set_title('Moving Image')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(fix_img[0], cmap='gray')
        axes[0, 1].set_title('Fixed Image')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(seg_out[0], cmap='gray')
        axes[0, 2].set_title('seg_out')
        axes[0, 2].axis('off')

        axes[0, 3].imshow(fix_img[0] - mov_img[0], cmap='gray')
        axes[0, 3].set_title('Difference')
        axes[0, 3].axis('off')

        # 第二行：分割
        axes[1, 0].imshow(mov_seg[0], cmap='gray')
        axes[1, 0].set_title('Moving Segmentation')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(fix_seg[0], cmap='gray')
        axes[1, 1].set_title('Fixed Segmentation')
        axes[1, 1].axis('off')

        axes[1, 2].imshow(warped_seg[0], cmap='gray')
        axes[1, 2].set_title('Warped Segmentation')
        axes[1, 2].axis('off')

        axes[1, 3].imshow(fix_seg[0] - warped_seg[0], cmap='gray')
        axes[1, 3].set_title('Seg Difference')
        axes[1, 3].axis('off')

        plt.suptitle(f'Epoch {epoch + 1}', fontsize=16)
        plt.tight_layout()

        # 保存图像
        viz_dir = os.path.join(self.config['save_dir'], 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        plt.savefig(os.path.join(viz_dir, f'epoch_{epoch + 1:03d}.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)

        # 保存到TensorBoard
        self.writer.add_figure('Validation/Visualization', fig, epoch)

    def save_checkpoint(self, epoch, dice_score, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_dice': self.best_val_dice,
            'config': self.config
        }

        # 保存最新检查点
        latest_path = os.path.join(self.config['save_dir'], 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_path)

        # 保存epoch检查点
        if (epoch + 1) % self.config.get('save_interval', 10) == 0:
            epoch_path = os.path.join(self.config['save_dir'], f'checkpoint_epoch_{epoch + 1:03d}.pth')
            torch.save(checkpoint, epoch_path)

        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.config['save_dir'], 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"✅ 保存最佳模型，Dice: {dice_score:.4f}")

    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            if 'best_val_dice' in checkpoint:
                self.best_val_dice = checkpoint['best_val_dice']

            if 'epoch' in checkpoint:
                self.start_epoch = checkpoint['epoch'] + 1

            print(f"✅ 从检查点加载: {checkpoint_path}")
            print(f"   恢复训练，起始epoch: {self.start_epoch}")
            print(f"   最佳Dice: {self.best_val_dice:.4f}")

            return True
        else:
            print(f"⚠️ 检查点不存在: {checkpoint_path}")
            return False

    def _plot_training_curves(self, epoch):
        """绘制训练曲线（增强多任务损失权重监控）"""
        if len(self.train_history['total_loss']) < 2:
            return

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        beta = self.config.get('beta', 0.5)  # 获取当前beta权重

        # 总损失
        axes[0, 0].plot(self.train_history['total_loss'])
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)

        # 分割损失
        axes[0, 1].plot(self.train_history['seg_loss'])
        axes[0, 1].set_title('Segmentation Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)

        # 配准损失（添加参考线，方便观察）
        axes[0, 2].plot(self.train_history['reg_loss'])
        axes[0, 2].axhline(y=1e-2, color='r', linestyle='--', label='1e-2 Reference')
        axes[0, 2].set_title('Registration Loss')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # 验证 Dice
        if self.train_history['val_dice']:
            axes[1, 0].plot(self.train_history['val_dice'], label='Warped Dice')
            if self.train_history['val_org_dice']:
                axes[1, 0].plot(self.train_history['val_org_dice'], label='Original Dice', linestyle='--')
            axes[1, 0].set_title('Validation Dice')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Dice')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # 学习率
        axes[1, 1].plot(self.train_history['learning_rate'])
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('LR')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)

        # 损失比例（增强监控，添加配准损失比例）
        if len(self.train_history['seg_loss']) > 0 and len(self.train_history['reg_loss']) > 0:
            seg_ratio = []
            reg_ratio = []
            for seg, reg in zip(self.train_history['seg_loss'], self.train_history['reg_loss']):
                total = seg + reg
                if total > 0:
                    seg_ratio.append(seg / total)
                    reg_ratio.append(reg / total)
                else:
                    seg_ratio.append(beta)
                    reg_ratio.append(1 - beta)
            axes[1, 2].plot(seg_ratio, label=f'Seg Ratio (β={beta})')
            axes[1, 2].plot(reg_ratio, label='Reg Ratio', linestyle='--')
            axes[1, 2].axhline(y=beta, color='r', linestyle=':', label=f'Target β={beta}')
            axes[1, 2].set_title('Loss Ratio (Seg vs Reg)')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Ratio')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)

        plt.suptitle(f'Training Curves - Epoch {epoch + 1}', fontsize=16)
        plt.tight_layout()

        # 保存图像
        curves_path = os.path.join(self.config['save_dir'], 'training_curves.png')
        plt.savefig(curves_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def train(self, num_epochs=None):
        """训练主循环"""
        if num_epochs is None:
            num_epochs = self.config['epochs']

        print(f"\n{'=' * 60}")
        print(f"开始联合训练，共 {num_epochs} 个epoch")
        print(f"{'=' * 60}")

        start_epoch = self.start_epoch

        for epoch in range(start_epoch, num_epochs):
            print(f"\n{'=' * 60}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'=' * 60}")

            # 训练
            avg_loss, avg_seg_loss, avg_reg_loss = self.train_epoch(epoch)

            # 验证
            avg_dice, avg_org_dice = self.validate(epoch)

            # 学习率调度：修改处！ReduceLROnPlateau传入验证集的核心指标（avg_dice）
            self.scheduler.step(avg_dice)  # ReduceLROnPlateau 需传入监控指标

            # 记录历史
            self.train_history['epoch'].append(epoch)
            self.train_history['total_loss'].append(avg_loss)
            self.train_history['seg_loss'].append(avg_seg_loss)
            self.train_history['reg_loss'].append(avg_reg_loss)
            self.train_history['val_dice'].append(avg_dice)
            self.train_history['val_org_dice'].append(avg_org_dice)
            self.train_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])

            # 记录到CSV
            self._log_to_csv(epoch, avg_loss, avg_seg_loss, avg_reg_loss, avg_dice, avg_org_dice)

            # 检查是否是最佳模型
            is_best = avg_dice > self.best_val_dice
            if is_best:
                self.best_val_dice = avg_dice
                self.best_epoch = epoch

            # 保存检查点
            if (epoch + 1) % self.config.get('save_interval', 20) == 0 or is_best:
                self.save_checkpoint(epoch, avg_dice, is_best)

            # 绘制训练曲线
            if (epoch + 1) % self.config.get('plot_interval', 10) == 0:
                self._plot_training_curves(epoch)

            # 打印结果
            print(f"\n训练结果:")
            print(f"  总损失: {avg_loss:.4f}")
            print(f"  分割损失: {avg_seg_loss:.4f}")
            print(f"  配准损失: {avg_reg_loss:.4f}")
            print(f"  验证Dice: {avg_dice:.4f} (原始: {avg_org_dice:.4f})")
            print(f"  最佳Dice: {self.best_val_dice:.4f} (epoch {self.best_epoch + 1})")
            print(f"  学习率: {self.optimizer.param_groups[0]['lr']:.2e}")

            # 早停检查
            if self.config.get('patience', 0) > 0:
                if epoch - self.best_epoch > self.config['patience']:
                    print(f"\n早停！{self.config['patience']}个epoch没有改进")
                    break

        print(f"\n{'=' * 60}")
        print(f"训练完成！")
        print(f"最佳验证Dice: {self.best_val_dice:.4f} (epoch {self.best_epoch + 1})")
        print(f"{'=' * 60}")

        # 保存最终模型
        self.save_checkpoint(num_epochs - 1, self.best_val_dice, False)

        # 绘制最终训练曲线
        self._plot_training_curves(num_epochs - 1)

        # 关闭TensorBoard写入器
        self.writer.close()

        return self.best_val_dice