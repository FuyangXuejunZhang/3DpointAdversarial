import os
import pdb
import time
from tqdm import tqdm
import copy
import argparse

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR


import sys
sys.path.append('../')
sys.path.append('./')

from dataset import ModelNetDataLoader, CustomModelNet40, ModelNet40Attack
from model import DGCNN, PointNetCls, feature_transform_reguliarzer, PointNet2ClsSsg, PointConvDensityClsSsg
from util.utils import cal_loss, AverageMeter, get_lr, str2bool, set_seed
from util import ClipPointsLinf, ChamferkNNDist, simCLRLoss, simCLRLoss_ori, CurvLoss, L2Dist, ChamferDist, HausdorffDist

from config import BEST_WEIGHTS
from config import MAX_BATCH as BATCH_SIZE
from Contrastive import CWContra

from plyfile import PlyData,PlyElement

    

def attack():
    model.eval()
    all_ori_pc = []
    all_adv_pc = []
    all_real_lbl = []
    
    ori_class_acc_num = 0 # 受害者模型成功分类数
    succ_num = 0 # 受害者模型攻击成功数
    
    for pc, label in tqdm(test_loader):
        with torch.no_grad():
            pc, label = pc.float().cuda(non_blocking=True), \
                label.long().cuda(non_blocking=True)

        # attack!
        _, adv_pc, ori_class_accuracy_num, success_num = attack_method.attack(pc, label, args)

        # results
        ori_class_acc_num += ori_class_accuracy_num
        succ_num += success_num
        
        all_ori_pc.append(pc.detach().cpu().numpy())
        all_adv_pc.append(adv_pc)
        all_real_lbl.append(label.detach().cpu().numpy())

    # accumulate results
    all_ori_pc=np.concatenate(all_ori_pc, axis=0)  # [num_data, K, 3]
    all_adv_pc = np.concatenate(all_adv_pc, axis=0)  # [num_data, K, 3]
    all_real_lbl = np.concatenate(all_real_lbl, axis=0)  # [num_data]
    


if __name__ == "__main__":
    # Training settings
    torch.autograd.set_detect_anomaly(True)
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--data_root', type=str,
                        default='') # MN40
    parser.add_argument('--model', type=str, default='pointnet', metavar='N',
                        choices=['pointnet', 'pointnet2',
                                 'dgcnn', 'pointconv'],
                        help='Model to use, [pointnet, pointnet++, dgcnn, pointconv]')
    parser.add_argument('--feature_transform', type=str2bool, default=True,
                        help='whether to use STN on features in PointNet')
    parser.add_argument('--dataset', type=str, default='mn40',
                        choices=['mn40', 'aug_mn40'])
    parser.add_argument('--batch_size', type=int, default=-1, metavar='BS',
                        help='Size of batch')
    parser.add_argument('--num_iter', type=int, default=200, metavar='N',
                        help='number of epochs to train ')
    parser.add_argument('--attack_lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate for the optimizer')
    parser.add_argument('--is_use_lr_scheduler', type=bool, default=True,
                        help='learning rate for the optimizer')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings in DGCNN')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use in DGCNN')
    parser.add_argument('--low_pass', type=int, default=100,
                        help='low_pass number')
    parser.add_argument('--budget', type=float, default=0.18, # 0.45
                        help='FGM attack budget')
    parser.add_argument('--GAMMA', type=float, default=0.25,
                        help='hyperparameter gamma')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='hyperparameter temperature')
    parser.add_argument('--binary_step', type=int, default=2, metavar='N',
                        help='Number of binary search step')
    parser.add_argument('--num_point', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--use_normals', action='store_true', default=True, help='use normals')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')

    parser.add_argument('--L2_loss_weight', type=float, default=1.0, help='')
    parser.add_argument('--Chamfer_loss_weight', type=float, default=1.0, help='')
    parser.add_argument('--Hausdorff_loss_weight', type=float, default=0.1, help='')
    parser.add_argument('--curv_loss_weight', type=float, default=1.0, help='')
    parser.add_argument('--curv_loss_knn', type=int, default=16, help='')
    args = parser.parse_args()

    BATCH_SIZE = BATCH_SIZE[args.num_point]
    BEST_WEIGHTS = BEST_WEIGHTS[args.dataset][args.num_point]
    if args.batch_size == -1:
        args.batch_size = BATCH_SIZE[args.model]
    set_seed(1)
    print(args)

    # build victim model
    if args.model.lower() == 'dgcnn':
        model = DGCNN(args.emb_dims, args.k, output_channels=40)
    elif args.model.lower() == 'pointnet':
        model = PointNetCls(k=40, feature_transform=args.feature_transform)
    elif args.model.lower() == 'pointnet2':
        model = PointNet2ClsSsg(num_classes=40)
    elif args.model.lower() == 'pointconv':
        model = PointConvDensityClsSsg(num_classes=40)
    else:
        print('Model not recognized')
        exit(-1)

    state_dict = torch.load(
        BEST_WEIGHTS[args.model], map_location='cpu')
    print('Loading weight {}'.format(BEST_WEIGHTS[args.model]))
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # eliminate 'module.' in keys
        state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

    model = model.cuda()

    # prepare dataset
    test_set = ModelNetDataLoader(root=args.data_root, args=args, split='test', process_data=args.process_data)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, 
                shuffle=False, num_workers=4, 
                drop_last=False)

    clip_func = ClipPointsLinf(budget=args.budget)
    dist_func1 = L2Dist()
    dist_func2 = ChamferDist()
    dist_func3 = HausdorffDist()
    contra_func = simCLRLoss(temperature = args.temperature)
    contra_func_ori = simCLRLoss_ori(temperature = args.temperature)
    curv_func = CurvLoss(curv_loss_knn = args.curv_loss_knn, curv_loss_weight = args.curv_loss_weight)
    attack_method = CWContra(model, dist_func1, dist_func2, dist_func3, contra_func, contra_func_ori, curv_func,
                        clip_func=clip_func)

    print(len(test_set))
    # run attack
    ori_data, attacked_data, real_label, ori_class_acc_num, succ_num = attack()

    # accumulate results
    data_num = len(test_set)
    ori_class_accuracy = float(ori_class_acc_num) / float(data_num)
    success_rate = float(succ_num) / float(data_num)

    print("受害者模型成功分类数ori_class_acc_num={}, 受害者模型攻击成功数succ_num={}".format(ori_class_acc_num, succ_num))
    print("在{}下的分类准确率为：{:.4f}, 攻击后攻击成功率为：{:.4f}".format(args.model, ori_class_accuracy, success_rate))

    save_path = './{}'.format(args.model)
    if not os.path.exists(save_path):
         os.makedirs(save_path)
    save_name = 'budget-{}_t-{}_success-{:.4f}.npz'.format(args.budget, args.temperature, success_rate)
    np.savez(os.path.join(save_path, save_name),
             ori_pc=ori_data.astype(np.float32),
             test_pc=attacked_data.astype(np.float32),
             test_label=real_label.astype(np.uint8))