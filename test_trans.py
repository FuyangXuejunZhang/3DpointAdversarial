from tqdm import tqdm
import argparse
import os
import numpy as np

import torch

import sys
sys.path.append('../')
sys.path.append('./')

from model import DGCNN, PointNetCls, feature_transform_reguliarzer, PointNet2ClsSsg, PointConvDensityClsSsg
from model.Hengshuang.model import PointTransformerCls
from util.utils import cal_loss, AverageMeter, get_lr, str2bool, set_seed

from config import BEST_WEIGHTS
from config import MAX_BATCH as BATCH_SIZE

from dataset import ModelNet40Transfer
from torch.utils.data import DataLoader


def transferability():
    trans_ori_class_acc_num = 0
    trans_error_num = 0
    trans_success_num = 0
    for ori_data, attacked_data, real_label in test_loader:

        # print(args.file_root)
        ori_data = ori_data.transpose(1, 2).contiguous().cuda()
        attacked_data = attacked_data.transpose(1, 2).contiguous().cuda()
        real_label = real_label.cuda()
                   
        trans_ori_logits = trans_model(ori_data)
        if args.trans_model.lower() == 'pointnet':
            trans_ori_logits = trans_ori_logits[0]
        
        trans_ori_preds = torch.argmax(trans_ori_logits, dim=-1)
        trans_ori_num = (trans_ori_preds == real_label).sum().item()
        
        # 错误率
        trans_err_logits = trans_model(attacked_data)
        if args.trans_model.lower() == 'pointnet':
            trans_err_logits = trans_err_logits[0]               
        trans_err_preds = torch.argmax(trans_err_logits, dim=-1)
        trans_err_num = (trans_err_preds != real_label).sum().item() # 生成对抗样本送入迁移模型，得到的错误分类个数
        
        # 可迁移性 advpc需要修改为 由在迁移模型中能够正确分类的样本生成的对抗样本
        diff = trans_ori_preds - real_label
        index = torch.where(diff == 0)[0]
        new_attacked_data = attacked_data[index]
        new_real_label = real_label[index]
        
        if new_attacked_data.shape[0] != 0: 
            trans_logits = trans_model(new_attacked_data)
            if args.trans_model.lower() == 'pointnet':
                trans_logits = trans_logits[0]
            trans_preds = torch.argmax(trans_logits, dim=-1)
        
            # 将由迁移模型正确分类的样本生成的对抗样本送入分类器，得到不能正确分类个数，即迁移模型攻击成功个数
            trans_num = (trans_preds != new_real_label).sum().item()
        else:
            trans_num = 0
            

        trans_ori_class_acc_num += trans_ori_num
        trans_error_num += trans_err_num
        trans_success_num += trans_num
    
    return trans_ori_class_acc_num, trans_error_num, trans_success_num


def test_normal(model_name):
    """Normal test mode.
    Test on all data.
    """
    trans_model.eval()
    at_num, at_denom = 0, 0  #

    num, denom = 0, 0 #
    num_error = 0
    with torch.no_grad():
        for ori_data, adv_data, label in test_loader:
            ori_data, adv_data, label = \
                ori_data.float().cuda(), adv_data.float().cuda(), label.long().cuda()
            # to [B, 3, N] point cloud
            ori_data = ori_data.transpose(1, 2).contiguous()
            adv_data = adv_data.transpose(1, 2).contiguous()
            batch_size = label.size(0)
            # batch in
            if model_name.lower() == 'pointnet':
                logits, _, _ = trans_model(ori_data)
                adv_logits, _, _ = trans_model(adv_data)
            elif model_name.lower() == 'pointtransformer':
                ori_data = ori_data.transpose(1, 2).contiguous() #[B,N,3]
                logits,_ = trans_model(ori_data)
                adv_data = adv_data.transpose(1, 2).contiguous() #[B,N,3]
                adv_logits,_ = trans_model(adv_data)
            else:
                logits = trans_model(ori_data)
                adv_logits = trans_model(adv_data)
            ori_preds = torch.argmax(logits, dim=-1)
            adv_preds = torch.argmax(adv_logits, dim=-1)
            mask_ori = (ori_preds == label)
            mask_adv = (adv_preds == label)
            err_num = (adv_preds != label)
            at_denom += mask_ori.sum().float().item() # 分类成功
            at_num += mask_ori.sum().float().item() - (mask_ori * mask_adv).sum().float().item() # 分类成功的样本生成的对抗样本分类不成功
            denom += float(batch_size)
            num_error += err_num.sum().float().item()
            num += mask_adv.sum().float()

    print('Overall attack success rate: {:.4f}'.format(at_num / (at_denom + 1e-9)))
    # ASR = at_num / (at_denom + 1e-9)
    print('Overall accuracy: {:.4f}'.format(at_denom / (denom + 1e-9))) # 模型本身的分类准确率
    print('top-1 error:{:.4f}'.format(num_error/(denom + 1e-9)))
    # print(ASR)
 

if __name__ == "__main__":
    # Training settings
    torch.autograd.set_detect_anomaly(True)
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--gpu', type=int, default=3)
    parser.add_argument('--file_root', type=str,
                        default='/home/chenhai-fwxz/ZYJ/Contra-master/results/contra+0.1fea/pointnet/budget-0.18_t-0.1_success-0.9887.npz')
    parser.add_argument('--trans_model', type=str, default='pointnet2',
                        choices=['pointnet', 'pointnet2', 'dgcnn', 'pointconv'],
                        help='Model to use, [pointnet, pointnet++, dgcnn, pointconv]')
    parser.add_argument('--feature_transform', type=str2bool, default=True,
                        help='whether to use STN on features in PointNet')
    parser.add_argument('--dataset', type=str, default='mn40',
                        choices=['mn40', 'aug_mn40'])
    parser.add_argument('--batch_size', type=int, default=-1, metavar='BS',
                        help='Size of batch')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings in DGCNN')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use in DGCNN')
    parser.add_argument('--num_point', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    args = parser.parse_args()

    BATCH_SIZE = BATCH_SIZE[args.num_point]
    BEST_WEIGHTS = BEST_WEIGHTS[args.dataset][args.num_point]
    if args.batch_size == -1:
        args.batch_size = BATCH_SIZE[args.trans_model]
    set_seed(1)
    print(args)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # build trans model
    if args.trans_model.lower() == 'dgcnn':
        trans_model = DGCNN(args.emb_dims, args.k, output_channels=40)
    elif args.trans_model.lower() == 'pointnet':
        trans_model = PointNetCls(k=40, feature_transform=args.feature_transform)
    elif args.trans_model.lower() == 'pointnet2':
        trans_model = PointNet2ClsSsg(num_classes=40)
    elif args.trans_model.lower() == 'pointconv':
        trans_model = PointConvDensityClsSsg(num_classes=40)
    elif args.trans_model.lower() == 'pointtransformer':
        trans_model = PointTransformerCls(args)
    else:
        print('Model not recognized')
        exit(-1)
        
    trans_state_dict = torch.load(
        BEST_WEIGHTS[args.trans_model], map_location='cpu')
    print('Loading weight {}'.format(BEST_WEIGHTS[args.trans_model]))
    if args.trans_model.lower() == 'pointtransformer':
        trans_model.load_state_dict(trans_state_dict['model_state_dict'])
    else:
        try:
            trans_model.load_state_dict(trans_state_dict)
        except RuntimeError:
            # eliminate 'module.' in keys
            trans_state_dict = {k[7:]: v for k, v in trans_state_dict.items()}
            trans_model.load_state_dict(trans_state_dict)
    trans_model.eval()
    trans_model.cuda()
    
    test_set = ModelNet40Transfer(args.file_root, num_points=args.num_point)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False, num_workers=4,
                             pin_memory=True, drop_last=False)
    data_num = len(test_set)

    test_normal(args.trans_model)

    # trans_ori_class_acc_num = 0 # 迁移模型原始成功分类数
    # trans_error_num = 0 # 将生成对抗样本送人迁移模型，得到的错误分类个数
    # trans_success_num = 0 # 将由迁移模型正确分类的样本生成的对抗样本送入分类器，得到不能正确分类个数，即迁移模型攻击成功个数

    # trans_ori_class_acc_num, trans_error_num, trans_success_num = transferability()

    # # accumulate results
    # trans_ori_class_accuracy = float(trans_ori_class_acc_num) / float(data_num)
    # error_rate = float(trans_error_num) / float(data_num)
    # transferablity = float(trans_success_num) / float(trans_ori_class_acc_num)

    # print("迁移模型{}原始成功分类数trans_ori_class_acc_num={}, 将生成对抗样本送入迁移模型,得到的错误分类个数trans_error_num={}, 将由迁移模型正确分类的样本生成的对抗样本送入分类器,得到不能正确分类个数,即迁移模型攻击成功个数trans_success_num={}"
    #       .format(args.trans_model, trans_ori_class_acc_num, trans_error_num, trans_success_num))
    # print("{}下的分类准确率为：{:.4f}, 错误率为：{:.4f}, 可迁移率为：{:.4f}"
    #       .format(args.trans_model, trans_ori_class_accuracy, error_rate, transferablity))
