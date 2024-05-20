import torch
import torch.nn as nn
import numpy as np
import copy
import random


def drop(points, ratio=0.875):
    """
    Randomly drop point cloud per point.
    Input:
        pc: Bx3xN array of original point clouds
    Return:
        points: Bx3x(N-N*ratio) array of point clouds after drop
        index: index of the drop point
    """
    point = torch.split(points, 1, dim=0)
    point = [tensor.squeeze(0) for tensor in point]
    
    for i in range(len(point)):
        point1 = point[i]
        enable = np.random.choice([False, True], replace=True, p=[0, 1])
        if enable:
            save_k = int(point1.shape[1] * ratio)
            index = np.random.choice(point1.shape[1], save_k, False, None).astype(np.int64)
            new_point1 = torch.zeros_like(point1)
            # points = torch.index_select(points, dim=2, index=torch.tensor(index).cuda())
            new_point1[:, index] = point1[:, torch.tensor(index).cuda()]
            point1 = new_point1
        point[i] = point1
    points = torch.stack(point, dim=0)
    return points


def flip(points):
    """
    Flip the point clouds randomly.
    Input:
        pc: Bx3xN array of original point clouds
    Return:
        pc: Bx3xN array of point clouds after flip
    """
    point = torch.split(points, 1, dim=0)
    point = [tensor.squeeze(0) for tensor in point]

    # 循环遍历每个tensor进行处理
    for i in range(len(point)):
        point1 = point[i]
        enable = np.random.choice([False, True], replace=True, p=[0, 1])
        if enable:
            point1_clone = point1.clone()
            axis = np.random.randint(0,3)
            point1_clone[axis, :] = -point1[axis, :]
            point1 = point1_clone
        # 更新修改后的tensor回到列表中
        point[i] = point1
    points = torch.stack(point, dim=0)
    
    return points


def scaling(points, scale_range=[0.5,1.5]):
    """
    Scale the point clouds randomly.
    Input:
        pc: Bx3xN array of original point clouds
    Return:
        pc: Bx3xN array of point clouds after scaling
    """
    # 如果缩放的尺度过小，则直接返回原来的box和点云
    if scale_range[1] - scale_range[0] < 1e-3:
        return points
    
    point = torch.split(points, 1, dim=0)
    point = [tensor.squeeze(0) for tensor in point]
    
    for i in range(len(point)):
        point1 = point[i]
        enable = np.random.choice([False, True], replace=True, p=[0, 1])
        if enable:
            point1_clone = point1.clone()
            # 在缩放范围内随机产生缩放尺度
            noise_scale = np.random.uniform(scale_range[0], scale_range[1])
            # 将点云乘以缩放尺度
            axis = np.random.randint(0,3)
            point1_clone[axis, :] = point1[axis, :]*noise_scale
            point1 = point1_clone
        point[i] = point1
    points = torch.stack(point, dim=0)
 
    return points


def shear(points):
    """
    Shear the point clouds randomly.
    Input:
        pc: Bx3xN array of original point clouds
    Return:
        pc: Bx3xN array of point clouds after shear
    """
    point = torch.split(points, 1, dim=0)
    point = [tensor.squeeze(0) for tensor in point]
    
    for i in range(len(point)):
        point1 = point[i]
        enable = np.random.choice([False, True], replace=True, p=[0, 1])
        if enable:
            point1_clone = point1.clone()
            c = torch.tensor([np.random.uniform(0, 0.15)])  # 剪切变形程度范围在[0, 0.15]之间
            # shear
            b = np.random.uniform(-c, c) * np.random.choice([-1, 1])
            d = np.random.uniform(-c, c) * np.random.choice([-1, 1])
            e = np.random.uniform(-c, c) * np.random.choice([-1, 1])
            f = np.random.uniform(-c, c) * np.random.choice([-1, 1])
            matrix = torch.tensor([[1, 0, b],
                                [d, 1, e],
                                [f, 0, 1]]).float().cuda()
            point1_clone = torch.matmul(point1.transpose(0, 1), matrix)
            point1 = point1_clone.transpose(0, 1).contiguous()
        point[i] = point1
    points = torch.stack(point, dim=0)
 
    return points


def translation(points, offset_range=[0, 0.1]):
    """
    Translation the point clouds randomly.
    Input:
        pc: Bx3xN array of original point clouds
    Return:
        pc: Bx3xN array of point clouds after translation
    """
    point = torch.split(points, 1, dim=0)
    point = [tensor.squeeze(0) for tensor in point]
    for i in range(len(point)):
        point1 = point[i]
        enable = np.random.choice([False, True], replace=True, p=[0, 1])
        if enable:
            point1_clone = point1.clone()
            offset = np.random.uniform(offset_range[0], offset_range[1]) * np.random.choice([-1,1])
            axis = np.random.randint(0,3)
            point1_clone[axis, :] += offset
            point1 = point1_clone
        point[i] = point1
    points = torch.stack(point, dim=0)
    
    return points


def check_numpy_to_torch(x):
    # 检测输入数据是否是numpy格式，如果是，则转换为torch格式
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def rotate_points_along_x(point, angle):
    """
    Args:
        points: (3, N)
        angle: (B), angle along x-axis
    Returns:
    """
    angle, _ = check_numpy_to_torch(angle)
    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    one = torch.ones([])
    zero = torch.zeros([])
    rotation_matrix = torch.stack([one, zero, zero,
                                   zero, cosa, -sina,
                                   zero, sina, cosa]).view(3, 3).float().cuda()
    point = point.transpose(0, 1).contiguous()
    rotated_pc = torch.matmul(point.cuda(), rotation_matrix) #[1024,3]*[3,3]

    return rotated_pc.transpose(0, 1)


def rotate_points_along_y(point, angle):
    
    # 首先利用torch.from_numpy().float将numpy转化为torch
    angle, _ = check_numpy_to_torch(angle)
    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    one = torch.ones([])
    zero = torch.zeros([])
    rotation_matrix = torch.stack([cosa, zero, sina,
                                   zero, one, zero,
                                   -sina, zero, cosa]).view(3, 3).float().cuda()
    # 对点云坐标进行旋转
    point = point.transpose(0, 1).contiguous()
    rotated_pc = torch.matmul(point.cuda(), rotation_matrix)

    return rotated_pc.transpose(0, 1)


def rotate_points_along_z(point, angle):
    
    angle, _ = check_numpy_to_torch(angle)
    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    one = torch.ones([])
    zero = torch.zeros([])
    rotation_matrix = torch.stack([cosa,  sina, zero,
                                   -sina, cosa, zero,
                                   zero, zero, one]).view(3, 3).float().cuda()
    # 对点云坐标进行旋转
    point = point.transpose(0, 1).contiguous()
    rotated_pc = torch.matmul(point.cuda(), rotation_matrix)

    return rotated_pc.transpose(0, 1)


def rotation(points, rot_range = [0, np.pi * 2]):
    """
    Rotation the point clouds randomly.
    Input:
        pc: Bx3xN array of original point clouds
    Return:
        pc: Bx3xN array of point clouds after rotation
    """
    point = torch.split(points, 1, dim=0)
    point = [tensor.squeeze(0) for tensor in point]
    for i in range(len(point)):
        point1 = point[i]
        enable = np.random.choice([False, True], replace=True, p=[0, 1])
        if enable:
            noise_rotation = np.random.uniform(rot_range[0], rot_range[1])
            r = np.random.randint(0,3)
            if r == 0:
                point1 = rotate_points_along_x(point1, np.array(noise_rotation))
            elif r == 1:
                point1 = rotate_points_along_y(point1, np.array(noise_rotation))
            else:
                point1 = rotate_points_along_z(point1, np.array(noise_rotation))
        point[i] = point1
    points = torch.stack(point, dim=0)
    return points






















def jitter(points, sigma=0.01, clip=0.05):
    """
    Randomly jitter point cloud per point.
    Input:
        pc: BxNx3 array of original point clouds
    Return:
        jittered_pc: BxNx3 array of point clouds after jitter
    """
    point = torch.split(points, 1, dim=0)
    point = [tensor.squeeze(0) for tensor in point]

    # 循环遍历每个tensor进行处理
    for i in range(len(point)):
        point1 = point[i]
        enable = np.random.choice([False, True], replace=True, p=[0.7, 0.3])
        if enable:
            C, N = point1.shape
            assert clip > 0
            jittered_pc = np.clip(sigma * np.random.randn(C, N), -1 * clip, clip)
            point1 += jittered_pc
        point[i] = point1
    points = torch.stack(point, dim=0)
    
    return points


def random_drop(points, k_num):
    # points_filtered, filter_index = filter_gt_points(points, gt_boxes_3d)
    # filter_idx = torch.where(filter_index)[0] #原始的index
    # points_ = copy.deepcopy(points_filtered)
    points_1 = copy.deepcopy(points)
    filter_idx = range(1024) #1024
    # Normal distribution function
    def normfun(x, mu, sigma):
        pdf = np.exp(-((x - mu)**2)/(2*sigma**2)) / (sigma * np.sqrt(2*np.pi))
        return pdf

    # p_index = np.linspace(0, points_.shape[0]-1, points_.shape[0])
    y = normfun(filter_idx.cpu().numpy(), filter_idx.cpu().numpy().mean(), filter_idx.cpu().numpy().std())
    y = y / y.sum()
    drop_k = int(points_1.shape[0] * k_num)
    output = np.random.choice(filter_idx.cpu().numpy(), drop_k, False, y) #随机采样函数
    out_index = output.astype(np.int64)
    
    pc_np = points_1.cpu().detach().numpy()
    points_np = np.delete(pc_np, out_index, axis=1)
    points_ = torch.from_numpy(points_np).to(points.device)
    return points_,out_index


def global_rotation(points, rot_range = [0, np.pi * 2]):
    """
    对点云进行整体旋转
    Args:
        points: (B, 3, N),
        rot_range: [min, max]
    Returns:
    """  
    point = torch.split(points, 1, dim=0)
    point = [tensor.squeeze(0) for tensor in point]
    for i in range(len(point)):
        point1 = point[i]
        enable = np.random.choice([False, True], replace=True, p=[0.7, 0.3])
        if enable:
            noise_rotation = np.random.uniform(rot_range[0], rot_range[1])
            point1= rotate_points_along_z(point1, np.array(noise_rotation))
        point[i] = point1
    points = torch.stack(point, dim=0)
    return points




def random_translation_along_x(points, offset_range=[-0.5, 0.5]):
    """
    Args:
        points: (B, 3, N),
        offset_range: [min max]]
    Returns:
    """
    point = torch.split(points, 1, dim=0)
    point = [tensor.squeeze(0) for tensor in point]
    for i in range(len(point)):
        point1 = point[i]
        offset = np.random.uniform(offset_range[0], offset_range[1])
        point1_clone = point1.clone()
        point1_clone[0, :] += offset
        point[i] = point1_clone
    points = torch.stack(point, dim=0)
    return points
 
 
def random_translation_along_y(points, offset_range=[0, 0.1]):

    point = torch.split(points, 1, dim=0)
    point = [tensor.squeeze(0) for tensor in point]
    for i in range(len(point)):
        point1 = point[i]
        offset = np.random.uniform(offset_range[0], offset_range[1])
        point1[1, :] += offset
        point[i] = point1
    points = torch.stack(point, dim=0)
    return points
 
 
def random_translation_along_z(points, offset_range=[0, 0.1]):

    point = torch.split(points, 1, dim=0)
    point = [tensor.squeeze(0) for tensor in point]
    for i in range(len(point)):
        point1 = point[i]
        offset = np.random.uniform(offset_range[0], offset_range[1])
        point1[2, :] += offset
        point[i] = point1
    points = torch.stack(point, dim=0)
    return points


#xz坐标不变，y取反，全局沿着xoz平面随机翻转
def random_flip_along_xoz(points):
    """
    沿着x轴随机翻转
    Args:
        points: (B, 3, N),
    Returns:
    """
    point = torch.split(points, 1, dim=0)
    point = [tensor.squeeze(0) for tensor in point]

    # 循环遍历每个tensor进行处理
    for i in range(len(point)):
        point1 = point[i]
        enable = np.random.choice([False, True], replace=True, p=[0.7, 0.3])
        if enable:
            point1_clone = point1.clone()
            point1_clone[1, :] = -point1[1, :]  # 点云z坐标取反
            point1 = point1_clone
        # 更新修改后的tensor回到列表中
        point[i] = point1

    # 合并为[32,3,1024]的tensor
    points = torch.stack(point, dim=0)
    return points


#yz坐标不变，x取反，全局沿着yoz平面随机翻转
def random_flip_along_yoz(points):

    point = torch.split(points, 1, dim=0)
    point = [tensor.squeeze(0) for tensor in point]

    # 循环遍历每个tensor进行处理
    for i in range(len(point)):
        point1 = point[i]
        enable = np.random.choice([False, True], replace=True, p=[0.7, 0.3])
        if enable:
            point1_clone = point1.clone()
            point1_clone[0, :] = -point1[0, :]  # 点云z坐标取反
            point1 = point1_clone
        # 更新修改后的tensor回到列表中
        point[i] = point1

    # 合并为[32,3,1024]的tensor
    points = torch.stack(point, dim=0)
    return points



#xy坐标不变，z取反，全局沿着xoy平面随机翻转
def random_flip_along_xoy(points):
    # points1 = points.clone()
    # points1[:, 2, :] = -points[:, 2, :] 
    # 进行维度变换，得到32个[3,1024]的tensor列表
    point = torch.split(points, 1, dim=0)
    point = [tensor.squeeze(0) for tensor in point]

    # 循环遍历每个tensor进行处理
    for i in range(len(point)):
        point1 = point[i]
        enable = np.random.choice([False, True], replace=True, p=[0.7, 0.3])
        if enable:
            point1_clone = point1.clone()
            point1_clone[2, :] = -point1[2, :]  # 点云z坐标取反
            point1 = point1_clone
        # 更新修改后的tensor回到列表中
        point[i] = point1

    # 合并为[32,3,1024]的tensor
    points = torch.stack(point, dim=0)
    return points


def global_scaling_x(points, scale_range=[0.5,1.5]):
    """
    随机缩放
    Args:
        points: (B, 3, N),s
        scale_range: [min, max]
    Returns:
    """
    # 如果缩放的尺度过小，则直接返回原来的box和点云
    if scale_range[1] - scale_range[0] < 1e-7:
        return points
    point = torch.split(points, 1, dim=0)
    point = [tensor.squeeze(0) for tensor in point]
    
    for i in range(len(point)):
        point1 = point[i]
        enable = np.random.choice([False, True], replace=True, p=[0.7, 0.3])
        if enable:
        # 在缩放范围内随机产生缩放尺度
            noise_scale = np.random.uniform(scale_range[0], scale_range[1])
            # 将点云乘以缩放尺度
            point1_clone = point1.clone()
            point1_clone[0, :] = point1[0, :] * noise_scale
            point1 = point1_clone
        point[i] = point1
    points = torch.stack(point, dim=0)
 
    return points


def global_scaling_y(points, scale_range=[0.5,1.5]):
    """
    随机缩放
    Args:
        points: (B, 3, N),s
        scale_range: [min, max]
    Returns:
    """
    # 如果缩放的尺度过小，则直接返回原来的box和点云
    if scale_range[1] - scale_range[0] < 1e-7:
        return points
    point = torch.split(points, 1, dim=0)
    point = [tensor.squeeze(0) for tensor in point]
    
    for i in range(len(point)):
        point1 = point[i]
        enable = np.random.choice([False, True], replace=True, p=[0.7, 0.3])
        if enable:
        # 在缩放范围内随机产生缩放尺度
            noise_scale = np.random.uniform(scale_range[0], scale_range[1])
            # 将点云乘以缩放尺度
            point1_clone = point1.clone()
            point1_clone[1, :] = point1[1, :] * noise_scale
            point1 = point1_clone
        point[i] = point1
    points = torch.stack(point, dim=0)
 
    return points


def global_scaling_z(points, scale_range=[0.5,1.5]):
    """
    随机缩放
    Args:
        points: (B, 3, N),s
        scale_range: [min, max]
    Returns:
    """
    # 如果缩放的尺度过小，则直接返回原来的box和点云
    if scale_range[1] - scale_range[0] < 1e-7:
        return points
    point = torch.split(points, 1, dim=0)
    point = [tensor.squeeze(0) for tensor in point]
    
    for i in range(len(point)):
        point1 = point[i]
        enable = np.random.choice([False, True], replace=True, p=[0.7, 0.3])
        if enable:
        # 在缩放范围内随机产生缩放尺度
            noise_scale = np.random.uniform(scale_range[0], scale_range[1])
            # 将点云乘以缩放尺度
            point1_clone = point1.clone()
            point1_clone[2, :]  = point1[2, :] * noise_scale
            point1 = point1_clone
        point[i] = point1
    points = torch.stack(point, dim=0)
 
    return points


