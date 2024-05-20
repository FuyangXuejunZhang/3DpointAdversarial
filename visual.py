import argparse
import os
from torch.utils.data import DataLoader
from dataset import ModelNet40Transfer
import numpy as np
from plyfile import PlyData,PlyElement


def save_ply(filename, point_cloud):
    """
    将点云数据保存为.ply文件

    参数:
        filename (str): 要保存的文件路径，如'/path/to/file.ply'
        point_cloud (numpy.ndarray): 点云数据，形状为 [B, 1024, 3]，其中B是Batchsize，1024是每个点云样本的点数，3是点的xyz坐标

    返回:
        无返回值
    """
    num_points = point_cloud.shape[1]

    indices_to_save = [0, 100, 150, 250, 270, 370, 470, 490, 590, 690, 710, 730, 750, 836, 856, 942, 962, 1062, 1162, 1182, 1202, \
        1222, 1322, 1422, 1508, 1528, 1628, 1728, 1748, 1848, 1868, 1968, 1988, 2008, 2108, 2128, 2228, 2328, 2428, 2448]  # 想要保存的特定索引

    for i in indices_to_save:
        points = []
        for j in range(num_points):
            point = (point_cloud[i, j, 0], point_cloud[i, j, 1], point_cloud[i, j, 2])
            points.append(point)

        vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

        el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
        plydata = PlyData([el])

        plydata.write(f"{filename}_{i}.ply")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visual Point Clouds')
    parser.add_argument('--file_root', type=str, default='')
    parser.add_argument('--dataset', type=str, default='mn40',
                        choices=['mn40', 'aug_mn40'])
    parser.add_argument('--batch_size', type=int, default=-1, metavar='BS',
                        help='Size of batch')
    parser.add_argument('--num_point', type=int, default=1024,
                        help='num of points to use')
    
    args = parser.parse_args()
    print(args)
    test_set = ModelNet40Transfer(args.file_root, num_points=args.num_point)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False, num_workers=4,
                             pin_memory=True, drop_last=False)
    
    for ori_data, adv_data, label in test_loader:
        ori_ply_path = './ply/ori'.format(args.model, args.temperature)
        if not os.path.exists(ori_ply_path):
            os.makedirs(ori_ply_path)
        save_ply("./ply/ori/{}".format(args.model, args.temperature, args.budget), ori_data)