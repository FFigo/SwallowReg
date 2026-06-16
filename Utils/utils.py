'''
Jiong Wu 
University of Florida
jiongwu.application@ufl.edu

Thanks to 
Junyu Chen
Johns Hopkins Unversity
jchen245@jhmi.edu
'''

import math
import numpy as np
import torch.nn.functional as F
import torch, sys
from scipy import ndimage
from torch import nn
import torch.utils.data as Data
import pystrum.pynd.ndutils as nd
from scipy.ndimage import gaussian_filter
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import os

import numpy as np
from scipy import ndimage
class BoundaryLoss(nn.Module):
    def __init__(self, theta=10.0, reduction='mean'):
        super(BoundaryLoss, self).__init__()
        self.theta = theta
        self.reduction = reduction

    def compute_distance_map(self, binary_mask):
        if binary_mask.dim() == 3:
            # 批量处理 [B, H, W]
            batch_size = binary_mask.shape[0]
            distance_maps = []
            for i in range(batch_size):
                mask_np = binary_mask[i].detach().cpu().numpy()
                dist_map = self._single_distance_map(mask_np)
                distance_maps.append(torch.from_numpy(dist_map))
            return torch.stack(distance_maps).to(binary_mask.device)
        else:
            # 单张图像 [H, W]
            mask_np = binary_mask.detach().cpu().numpy()
            dist_map = self._single_distance_map(mask_np)
            return torch.from_numpy(dist_map).to(binary_mask.device)
    def _single_distance_map(self, binary_mask):
        """计算单张2D图像的距离变换"""
        # 确保是二值图
        binary_mask = binary_mask > 0.5

        # 计算边界（形态学梯度）
        structure = ndimage.generate_binary_structure(2, 2)  # 2x2结构元素
        eroded = ndimage.binary_erosion(binary_mask, structure)
        boundary = binary_mask & (~eroded)

        # 计算距离变换
        distance_map = ndimage.distance_transform_edt(~boundary)

        # 应用指数衰减：exp(-distance/theta)
        distance_map = np.exp(-distance_map / self.theta)

        return distance_map.astype(np.float32)

    def forward(self, warped_seg, fixed_seg):
        # 统一输入形状为 [B, H, W]
        if warped_seg.dim() == 4:
            warped_seg = warped_seg.squeeze(1)  # [B, 1, H, W] -> [B, H, W]
        if fixed_seg.dim() == 4:
            fixed_seg = fixed_seg.squeeze(1)  # [B, 1, H, W] -> [B, H, W]

        # 确保输入形状一致
        assert warped_seg.shape == fixed_seg.shape

        batch_size = warped_seg.size(0)
        total_loss = 0.0

        for i in range(batch_size):
            # 获取当前样本
            pred = warped_seg[i]  # [H, W] - 预测概率图
            target = fixed_seg[i]  # [H, W] - 真实标签

            # 二值化处理
            pred_binary = (pred > 0.5).float()
            target_binary = (target > 0.5).float()

            # 计算固定标签的距离变换图
            dist_map = self.compute_distance_map(target_binary.unsqueeze(0))
            dist_map = dist_map[0] if dist_map.dim() == 3 else dist_map  # [H, W]

            # 纯边界损失公式: ∑(距离图 × 预测概率)
            # 这会使预测向固定标签的边界对齐
            loss = torch.sum(dist_map * pred)
            total_loss += loss

        # 根据reduction参数返回损失
        if self.reduction == 'mean':
            return total_loss / batch_size
        elif self.reduction == 'sum':
            return total_loss
        else:
            raise ValueError("reduction参数必须是'mean'或'sum'")


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals = []
        self.std = 0
  
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.vals.append(val)
        self.std = np.std(self.vals)

class SpatialTransform(nn.Module):
    def __init__(self):
        super(SpatialTransform, self).__init__()
    def forward(self, mov_image, flow, mod = 'bilinear'):
        h2, w2 = mov_image.shape[-2:]
        grid_h, grid_w = torch.meshgrid([torch.linspace(-1, 1, h2), torch.linspace(-1, 1, w2)])
        grid_h = grid_h.to(flow.device).float()
        grid_w = grid_w.to(flow.device).float()
        grid_w = nn.Parameter(grid_w, requires_grad=False)
        grid_h = nn.Parameter(grid_h, requires_grad=False)
        flow_h = flow[:,:,:,0]
        flow_w = flow[:,:,:,1]

        disp_h = (grid_h + (flow_h)).squeeze(1)
        disp_w = (grid_w + (flow_w)).squeeze(1)
        sample_grid = torch.stack((disp_w, disp_h), 3)  # shape (N, D, H, W, 3)
        warped = torch.nn.functional.grid_sample(mov_image, sample_grid, mode = mod, align_corners = True,padding_mode="border")
        
        return sample_grid, warped


def smoothloss(y_pred):
    h2, w2 = y_pred.shape[-2:]
    dx = torch.abs(y_pred[:,:, 1:, :] - y_pred[:, :, :-1, :]) / 2 * h2
    dz = torch.abs(y_pred[:,:, :, 1:] - y_pred[:, :, :, :-1]) / 2 * w2
    return (torch.mean(dx * dx) + torch.mean(dz*dz))/2.0


def magnitude_loss(flow_1, flow_2):
    num_ele = torch.numel(flow_1)
    flow_1_mag = torch.sum(torch.abs(flow_1))
    flow_2_mag = torch.sum(torch.abs(flow_2))

    diff = (torch.abs(flow_1_mag - flow_2_mag))/num_ele

    return diff

def gncc(self, I, J, eps=1e-5):
    """计算归一化互相关"""
    I = I - I.mean()
    J = J - J.mean()
    cross = torch.sum(I * J)
    I_var = torch.sum(I ** 2)
    J_var = torch.sum(J ** 2)
    return cross / (torch.sqrt(I_var) * torch.sqrt(J_var) + eps)






class MSE:
    """
    Mean squared error loss.
    """
 
    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)


def jacobian_determinant_vxm(disp):
    """
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.
    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
              where vol_shape is of len nb_dims
    Returns:
        jacobian determinant (scalar)
    """

    # check inputs
    disp = disp.transpose(1, 2, 0)
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    J = np.gradient(disp + grid)

    # 3D glow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    else:  # must be 2

        dfdx = J[0]
        dfdy = J[1]

        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]


def crop_center(img, cropx, cropy, cropz):
    x, y, z = img.shape
    startx = x//2 - cropx//2
    starty = y//2 - cropy//2
    startz = z//2 - cropz//2
    return img[startx:startx+cropx, starty:starty+cropy, startz:startz+cropz]


def imgnorm(img):
    i_max = np.max(img)
    i_min = np.min(img)
    norm = (img - i_min)/(i_max - i_min)
    return norm

def loadnpz(npzpath):
    features=np.load(npzpath, allow_pickle=True)
    f_all = features['arr_0'].item()
    imglist = f_all['imglist']
    movimg = imglist[0,:,:]
    movlab = imglist[1,:,:]
    tarimg = imglist[2,:,:]
    tarlab = imglist[3,:,:]
    return movimg, movlab, tarimg, tarlab

def dice(pred, target, smooth=1e-6):
    """计算Dice系数"""
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    return dice

class dice_loss:
    def dice(self,pred, target, smooth=1e-6):
        """计算Dice系数"""
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        return dice

class Dataset_epoch_with_name(Dataset):
    """支持联合训练的数据集"""

    def __init__(self, file_paths, sam_image_size=224, reg_image_size=256, augment=True, debug=False):
        self.file_paths = file_paths
        self.sam_image_size = sam_image_size
        self.reg_image_size = reg_image_size
        self.augment = augment
        self.debug = debug

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = np.load(self.file_paths[idx])

        # 获取原始数据
        mov_img = data['mov']  # [256, 256, 3]
        fix_img = data['fix']  # [256, 256, 3]
        mov_seg = data['mov_seg']  # [256, 256]
        fix_seg = data['fix_seg']  # [256, 256]

        # 添加通道维度
        mov_seg = mov_seg[..., np.newaxis]  # [256, 256, 1]
        fix_seg = fix_seg[..., np.newaxis]  # [256, 256, 1]

        # 将RGB转换为灰度
        def rgb_to_grayscale(rgb_image):
            if rgb_image.shape[-1] == 3:
                gray = 0.299 * rgb_image[:, :, 0] + \
                       0.587 * rgb_image[:, :, 1] + \
                       0.114 * rgb_image[:, :, 2]
                return gray[..., np.newaxis]
            return rgb_image

        mov_img_gray = rgb_to_grayscale(mov_img)  # [256, 256, 1]
        fix_img_gray = rgb_to_grayscale(fix_img)  # [256, 256, 1]

        # 数据增强...

        # 调整SAM输入尺寸 (224×224)
        mov_sam = self.resize_image(mov_img_gray, self.sam_image_size, is_label=False)
        fix_sam = self.resize_image(fix_img_gray, self.sam_image_size, is_label=False)

        # 调整配准输入尺寸 (256×256)
        mov_reg = self.resize_image(mov_img_gray, self.reg_image_size, is_label=False)
        fix_reg = self.resize_image(fix_img_gray, self.reg_image_size, is_label=False)

        # 调整分割标签尺寸 - 为SAM提供224×224标签
        mov_lab_sam = self.resize_image(mov_seg, self.sam_image_size, is_label=True)
        fix_lab_sam = self.resize_image(fix_seg, self.sam_image_size, is_label=True)

        # 调整分割标签尺寸 - 为配准提供256×256标签
        mov_lab_reg = self.resize_image(mov_seg, self.reg_image_size, is_label=True)
        fix_lab_reg = self.resize_image(fix_seg, self.reg_image_size, is_label=True)

        # 归一化...

        # 二值化标签,数据Guan和Shi是127，2CH是0.5
        mov_lab_sam = (mov_lab_sam >127).astype(np.float32)
        fix_lab_sam = (fix_lab_sam > 127).astype(np.float32)
        mov_lab_reg = (mov_lab_reg > 127).astype(np.float32)
        fix_lab_reg = (fix_lab_reg > 127).astype(np.float32)

        # 转换为 [C, H, W] 格式
        def to_chw(img):
            if len(img.shape) == 3:  # H, W, C
                return np.transpose(img, (2, 0, 1))
            elif len(img.shape) == 2:  # H, W
                return np.expand_dims(img, axis=0)
            return img

        sample = {
            'movimg_sam': torch.from_numpy(to_chw(mov_sam)).float(),
            'tarimg_sam': torch.from_numpy(to_chw(fix_sam)).float(),
            'movimg_reg': torch.from_numpy(to_chw(mov_reg)).float(),
            'tarimg_reg': torch.from_numpy(to_chw(fix_reg)).float(),
            'movlab_sam': torch.from_numpy(to_chw(mov_lab_sam)).float(),  # 224×224
            'tarlab_sam': torch.from_numpy(to_chw(fix_lab_sam)).float(),  # 224×224
            'movlab_reg': torch.from_numpy(to_chw(mov_lab_reg)).float(),  # 256×256
            'tarlab_reg': torch.from_numpy(to_chw(fix_lab_reg)).float(),  # 256×256
            'filename': os.path.basename(self.file_paths[idx])
        }

        return sample

    def resize_image(self, image, target_size, is_label=False):
        """最简单的尺寸调整"""
        # image可能是: (256, 256, 1) 或 (1, 256, 256)

        # 1. 统一转换为(H, W, C)格式
        if len(image.shape) == 3 and image.shape[0] == 1:
            # 当前是(1, H, W)，转换为(H, W, 1)
            image = image.transpose(1, 2, 0)
        elif len(image.shape) == 2:
            # (H, W)，转换为(H, W, 1)
            image = image[..., np.newaxis]

        # 现在image肯定是(H, W, C)格式
        h, w, c = image.shape

        # 2. 如果已经是目标尺寸，直接返回
        if h == target_size and w == target_size:
            return image

        # 3. 使用scipy的zoom函数
        from scipy.ndimage import zoom

        # 计算缩放因子
        zoom_factor_h = target_size / h
        zoom_factor_w = target_size / w

        if is_label:
            # 标签：最近邻插值
            resized = zoom(
                image,
                (zoom_factor_h, zoom_factor_w, 1),  # 保持通道不变
                order=0,
                mode='nearest'
            )
        else:
            # 图像：双三次插值
            resized = zoom(
                image,
                (zoom_factor_h, zoom_factor_w, 1),
                order=3,
                mode='constant',
                cval=image.min()
            )

        return resized

    def normalize(self, image):
        """归一化图像"""
        image_min = image.min()
        image_max = image.max()

        if image_max - image_min > 1e-6:
            image = (image - image_min) / (image_max - image_min)

        return image


class Dataset_epoch(Data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, names):
        'Initialization'
        super(Dataset_epoch, self).__init__()
        self.names = names

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.names)

  def __getitem__(self, index):
        'Generates one sample of data'
        npzpath = self.names[index]
        movimg, movlab, tarimg, tarlab = loadnpz(npzpath)

        movimg = torch.from_numpy(movimg).float()
        tarimg = torch.from_numpy(tarimg).float()
        movlab = torch.from_numpy(movlab).float()
        tarlab = torch.from_numpy(tarlab).float()

        return movimg.unsqueeze(0), tarimg.unsqueeze(0), movlab.unsqueeze(0), tarlab.unsqueeze(0)

