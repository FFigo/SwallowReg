import argparse
import os
import sys
import glob
import yaml
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings

# 设置环境变量，禁用oneDNN和TensorFlow日志
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=所有, 1=INFO, 2=WARNING, 3=ERROR

warnings.filterwarnings('ignore')

# 添加当前目录到路径
sys.path.append(str(Path(__file__).parent))

from joint_Trainer import JointSAMRegistrationTrainer


def setup_experiment_dirs(args):
    """设置实验目录"""
    # 创建时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 构建实验名称：
    # 显式指定 --experiment_name 时优先使用；否则按消融配置自动命名，
    # 避免不同消融实验（fusion_mode / 边缘损失开关）输出到同一目录互相覆盖。
    if args.experiment_name:
        experiment_name = args.experiment_name
    else:
        edge_tag = "edge" if args.use_edge_iou_loss else "noedge"
        experiment_name = f"{args.reg_net}_{args.fusion_mode}_{edge_tag}_{timestamp}"

    # 设置保存目录
    save_dir = os.path.join(args.save_dir, experiment_name)
    log_dir = os.path.join(save_dir, "logs")

    # 创建目录
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    return save_dir, log_dir


def check_data_paths(data_path):
    """检查数据路径结构"""
    train_path = os.path.join(data_path, "train")
    val_path = os.path.join(data_path, "val")
    test_path = os.path.join(data_path, "test")

    data_info = {
        'train': train_path if os.path.exists(train_path) else None,
        'val': val_path if os.path.exists(val_path) else None,
        'test': test_path if os.path.exists(test_path) else None
    }

    # 打印检查结果
    for split, path in data_info.items():
        if path is None:
            print(f"⚠️ 警告: {split} 目录不存在: {os.path.join(data_path, split)}")
        else:
            npz_files = glob.glob(os.path.join(path, "*.npz"))
            print(f"✅ {split} 目录: {path} ({len(npz_files)} 个文件)")

    return data_info


def print_training_info(args, data_info, save_dir):
    """打印训练信息"""
    print("\n" + "=" * 60)
    print("联合SAM分割和配准训练配置")
    print("=" * 60)

    print(f"\n🔧 模型配置:")
    print(f"  训练轮数: {args.epochs}")
    print(f"  批次大小: {args.batch_size}")
    print(f"  学习率: {args.learning_rate}")
    print(f"  分割损失权重(β): {args.beta}")
    print(f"  配准平滑性损失权重: {args.reg_smooth}")

    print(f"\n📁 数据配置:")
    print(f"  数据根目录: {args.data_path}")
    for split, path in data_info.items():
        if path:
            file_count = len(glob.glob(os.path.join(path, "*.npz")))
            print(f"  {split} 数据: {path} ({file_count} 个文件)")

    print(f"\n💾 输出配置:")
    print(f"  保存目录: {save_dir}")
    print(f"  日志目录: {os.path.join(save_dir, 'logs')}")

    print(f"\n⚙️ 硬件配置:")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  设备: {device}")
    if torch.cuda.is_available():
        print(f"  GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"  GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print("=" * 60)


def save_config(args, data_info, save_dir):
    """保存配置到文件"""
    config = {
        'sam_config': {
            'model_type': 'vit_b',
            'checkpoint_path': args.sam_checkpoint,
            'image_size': 224
        },
        'reg_config': {
            'image_size': 256
        },
        'train_data_dir': data_info['train'],
        'val_data_dir': data_info['val'],
        'test_data_dir': data_info['test'],
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'min_lr': 1e-6,
        'weight_decay': 1e-5,
        'beta': args.beta,
        'reg_smooth': args.reg_smooth,
        'save_dir': save_dir,
        'log_dir': os.path.join(save_dir, "logs"),
        'save_interval': 5,
        'num_workers': args.num_workers,
        'grad_clip': 1.0,
        'training_mode': 'joint',
        'use_edge_iou_loss': args.use_edge_iou_loss,  # 边缘加权 IoU 损失开关
        'reg_net': args.reg_net,  # swin_intern / voxelmorph / transmorph
        'fusion_mode': args.fusion_mode,  # full / concat / sam_only / enc_only（论文 Table 1 消融）
        'use_feature': True,  # 旧参数，已由 fusion_mode 取代，仅为向后兼容保留
        'viz_interval': 5,
        'plot_interval': 10
    }

    # 保存YAML配置
    config_path = os.path.join(save_dir, "config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"✅ 配置已保存到: {config_path}")

    return config


def set_random_seed(seed=42):
    """设置随机种子"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 确保可重复性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"✅ 随机种子设置为: {seed}")


def main():
    parser = argparse.ArgumentParser(description='联合训练SAM分割和配准网络')

    # 必需参数
    parser.add_argument('--data_path', type=str, default=None,
                        help='数据根目录，应包含 train/, val/ 子目录（必填）')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=61,
                        help='训练轮数 (默认: 50)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='批次大小 (默认: 4)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='学习率 (默认: 1e-4)')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='分割损失权重 β (默认: 0.5)')
    parser.add_argument('--reg_smooth', type=float, default=0.01,
                        help='配准平滑性损失权重 (默认: 0.01)')
    parser.add_argument('--sam_checkpoint', type=str,
                        default='./SAMWeights/sam_vit_b_01ec64.pth',
                        help='SAM ViT-B 预训练权重路径 (默认: ./SAMWeights/sam_vit_b_01ec64.pth)；'
                             '可用 SAMWeights/down_weights.py 下载')
    parser.add_argument('--reg_net', type=str, default='swin_intern',
                        choices=['swin_intern', 'voxelmorph', 'transmorph'],
                        help="配准网络: swin_intern(本文模型) / voxelmorph / transmorph "
                             "(后两者为 baseline, 仅支持 fusion_mode 的 full/enc_only) (默认: swin_intern)")
    parser.add_argument('--fusion_mode', type=str, default='full',
                        choices=['full', 'concat', 'sam_only', 'enc_only'],
                        help="特征融合模式，对应论文 Table 1 消融: "
                             "full(BCAF完整模型) / concat(直接拼接,无BCAF) / "
                             "sam_only(仅SAM特征) / enc_only(仅编码器特征) (默认: full)")
    # 边缘加权 IoU 损失开关（与 fusion_mode 正交，对应 Table 1 中 Row1↔Row3、Row2↔Row4 的差异）
    parser.add_argument('--use_edge_iou_loss', dest='use_edge_iou_loss',
                        action='store_true', default=True,
                        help='启用边缘加权 IoU 损失 (默认: 启用)')
    parser.add_argument('--no_edge_iou_loss', dest='use_edge_iou_loss',
                        action='store_false',
                        help='关闭边缘加权 IoU 损失（如复现 Table 1 的 Row 1 / Row 2）')

    # 输出参数
    parser.add_argument('--save_dir', type=str, default=None,
                        help='保存模型/日志的根目录（必填）')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='实验名称；不指定时按 "{fusion_mode}_{edge|noedge}_{时间戳}" 自动命名')

    # 其他参数
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载工作进程数 (默认: 4)')
    parser.add_argument('--resume', type=str, default=None,
                        help='从检查点恢复训练')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    parser.add_argument('--no_cuda', action='store_true',
                        help='禁用CUDA/GPU')

    args = parser.parse_args()

    # 1. 检查数据路径
    print("\n🔍 检查数据路径...")
    data_info = check_data_paths(args.data_path)

    if not data_info['train']:
        print("❌ 错误: 训练数据目录不存在!")
        sys.exit(1)

    if not data_info['val']:
        print("⚠️ 警告: 验证数据目录不存在，将使用训练数据进行验证")
        data_info['val'] = data_info['train']

    # 2. 设置实验目录
    print("\n📁 设置实验目录...")
    save_dir, log_dir = setup_experiment_dirs(args)

    # 3. 打印训练信息
    print_training_info(args, data_info, save_dir)

    # 4. 设置随机种子
    set_random_seed(args.seed)

    # 5. 保存配置
    print("\n💾 保存配置...")
    config = save_config(args, data_info, save_dir)

    # 6. 创建训练器
    print("\n🚀 创建训练器...")
    print(f"  配准网络 (reg_net): {config['reg_net']}")
    print(f"  特征融合模式 (fusion_mode): {config['fusion_mode']}")
    print(f"  边缘加权 IoU 损失: {config['use_edge_iou_loss']}")

    try:
        trainer = JointSAMRegistrationTrainer(config)
    except Exception as e:
        print(f"❌ 创建训练器失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 7. 恢复训练
    if args.resume:
        print(f"\n🔄 恢复训练: {args.resume}")
        if os.path.exists(args.resume):
            trainer.load_checkpoint(args.resume)
        else:
            print(f"⚠️ 检查点不存在: {args.resume}")

    # 8. 开始训练
    print("\n" + "=" * 60)
    print("开始训练！")
    print("=" * 60)

    try:
        best_dice = trainer.train(config['epochs'])
        print(f"\n🎉 训练完成！最佳验证 Dice: {best_dice:.4f}")

    except KeyboardInterrupt:
        print("\n\n⏹️ 训练被用户中断")

        # 保存中断时的检查点
        trainer.save_checkpoint(trainer.start_epoch, trainer.best_val_dice, is_best=False)
        print(f"✅ 中断时的检查点已保存到: {save_dir}")

    except Exception as e:
        print(f"\n❌ 训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

        # 保存错误时的检查点
        trainer.save_checkpoint(trainer.start_epoch, trainer.best_val_dice, is_best=False)
        print(f"✅ 错误时的检查点已保存到: {save_dir}")

    finally:
        print(f"\n📁 实验目录: {save_dir}")
        print("✨ 训练脚本完成！")


if __name__ == '__main__':
    main()