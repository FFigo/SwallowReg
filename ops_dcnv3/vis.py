import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 添加DCNv3模块路径
from modules.dcnv3 import DCNv3


# 创建StemLayer
class StemLayer(nn.Module):
    def __init__(self, in_chans=1, out_chans=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans // 2, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(out_chans // 2)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(out_chans // 2, out_chans, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(out_chans)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return x


# 创建可视化模型
class FeatureVisualizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = StemLayer(in_chans=1, out_chans=16)
        self.dcnv3 = DCNv3(
            channels=16, kernel_size=3, dw_kernel_size=3, stride=1, pad=1,
            dilation=1, group=4, offset_scale=1.0, act_layer='GELU',
            norm_layer='LN', center_feature_scale=False, remove_center=False
        )
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, 16),
            nn.GELU()
        )

    def forward(self, x):
        x_stem = self.stem(x)
        x_dcn = x_stem.permute(0, 2, 3, 1)
        dcn_features = self.dcnv3(x_dcn)
        dcn_features = dcn_features.permute(0, 3, 1, 2)
        cnn_features = self.cnn(x)
        return dcn_features, cnn_features


# 加载舌体图像
def load_tongue_image(image_path, size=256):
    img = Image.open(image_path).convert('L').resize((size, size))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
    return img_tensor, img_array


# 简洁的可视化函数
def visualize_features_simple(image, dcn_features, cnn_features):
    """简洁明了的特征可视化"""
    # 创建图像网格
    fig, axs = plt.subplots(3, 5, figsize=(20, 12))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    # 原始图像
    axs[0, 0].imshow(image, cmap='gray')
    axs[0, 0].set_title('Original Image', fontsize=12)
    axs[0, 0].axis('off')

    # DCNv3特征标题
    axs[0, 1].text(0.5, 0.5, 'DCNv3 Features',
                   ha='center', va='center', fontsize=14, fontweight='bold')
    axs[0, 1].axis('off')

    # DCNv3特征图
    for i in range(4):
        row = i // 2 + 1
        col = i % 2 + 1
        feature_map = dcn_features[0, i].detach().numpy()
        axs[row, col].imshow(feature_map, cmap='viridis')
        axs[row, col].set_title(f'DCNv3 Ch {i + 1}', fontsize=10)
        axs[row, col].axis('off')

    # CNN特征标题
    axs[0, 3].text(0.5, 0.5, 'CNN Features',
                   ha='center', va='center', fontsize=14, fontweight='bold')
    axs[0, 3].axis('off')

    # CNN特征图
    for i in range(4):
        row = i // 2 + 1
        col = i % 2 + 3
        feature_map = cnn_features[0, i].detach().numpy()
        axs[row, col].imshow(feature_map, cmap='viridis')
        axs[row, col].set_title(f'CNN Ch {i + 1}', fontsize=10)
        axs[row, col].axis('off')

    # 差异分析
    dcn_mean = dcn_features.mean(dim=1)[0].detach().numpy()
    cnn_mean = cnn_features.mean(dim=1)[0].detach().numpy()
    diff = np.abs(dcn_mean - cnn_mean)

    axs[2, 0].imshow(diff, cmap='hot')
    axs[2, 0].set_title('Feature Difference\n(DCNv3 vs CNN)', fontsize=12)
    axs[2, 0].axis('off')

    # 隐藏未使用的子图
    for i in range(1, 5):
        axs[2, i].axis('off')

    plt.tight_layout()
    plt.savefig('feature_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


# 主函数
def main():
    # 加载舌体图像
    image_path = r'C:\MrRegNet\DATA\normal\1_original.jpg'
    image_tensor, image_array = load_tongue_image(image_path)

    # 创建模型
    model = FeatureVisualizer()

    # 前向传播获取特征
    with torch.no_grad():
        dcn_features, cnn_features = model(image_tensor)

    # 可视化特征比较
    visualize_features_simple(image_array, dcn_features, cnn_features)

    print("可视化完成！结果已保存为 feature_comparison.png")


if __name__ == '__main__':
    main()