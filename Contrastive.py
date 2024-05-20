"""Implementation of Untargeted AdvPC CW Attack for point perturbation.
"""

import pdb
import time
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from util.dist_utils import L2Dist, ChamferDist, HausdorffDist
from util.augment import drop, flip, rotation, scaling, shear, translation
from torchvision.transforms import Compose

from plyfile import PlyData,PlyElement
import matplotlib.pyplot as plt


def get_lr_scheduler(optim, scheduler, total_step):
    '''
    get lr values
    '''
    lrs = []
    for step in range(total_step):
        lr_current = optim.param_groups[0]['lr']
        lrs.append(lr_current)
        if scheduler is not None:
            scheduler.step()
    return lrs
# global
model = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3))
initial_lr = 1.
total_step = 200


def grad_scores(logits, feature):

    grads = torch.autograd.grad(logits, feature, grad_outputs=torch.ones_like(logits), create_graph=True)
    feature_gradients = grads[0]
    feature_gradients_abs = torch.abs(feature_gradients)
    feature = feature_gradients_abs * feature


    return feature


class CWContra:
    """Class for CW Contra attack.
    """
    def __init__(self, model, dist_func1, dist_func2, dist_func3, contra_func, contra_func_ori, curv_func, clip_func=None):
        """CW attack by perturbing points.

        Args:
            model (torch.nn.Module): victim model
            adv_func (function): adversarial loss function
            dist_func (function): distance metric
            attack_lr (float, optional): lr for optimization. Defaults to 1e-2.
            binary_step (int, optional): binary search step. Defaults to 10.
            num_iter (int, optional): max iter num in every search step. Defaults to 500.
        """

        self.model = model.cuda()
        self.model.eval()

        self.dist_func1 = dist_func1
        self.dist_func2 = dist_func2
        self.dist_func3 = dist_func3
        self.contra_func = contra_func
        self.contra_func_ori = contra_func_ori
        self.curv_func = curv_func
        self.clip_func = clip_func

    def attack(self, pc, target, args):
        """Attack on given data to target.

        Args:
            data (torch.FloatTensor): victim data, [B, num_points, 3]
            target (torch.LongTensor): target output, [B]
        """
        B, K = pc.shape[:2]
        data = pc[:,:,:3].float().cuda().detach()
        normal = pc[:,:,-3:].float().cuda()
        data = data.transpose(1, 2).contiguous() # torch.Size([B, 3, 1024])
        normal = normal.transpose(1, 2).contiguous()
        ori_data = data.clone().detach()
        ori_data.requires_grad = False
        target = target.long().cuda().detach()
        label_val = target.detach().cpu().numpy()  # [B]

        # record best results in binary search
        o_bestdist = np.array([1e10] * B)
        o_bestscore = np.array([-1] * B)
        o_bestattack = np.zeros((B, 3, K))

        functions = [drop, flip, rotation, scaling, shear, translation]
        selected_function1 = random.choice(functions)
        aug_pc1 = selected_function1(ori_data)
        aug_pc1.requires_grad = False
        selected_function2 = random.choice(functions)
        aug_pc2 = selected_function2(ori_data)
        aug_pc2.requires_grad = False



        for binary_step in range(args.binary_step):
            offset = torch.zeros(B, 3, K).cuda()
            nn.init.normal_(offset, mean=0, std=1e-3) # 正态分布初始化

            adv_data = ori_data.clone() + offset
            adv_data.requires_grad_()
            opt = optim.Adam([adv_data], lr=args.attack_lr)
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9990)


            dist_loss1 = torch.tensor(0.).cuda()
            dist_loss2 = torch.tensor(0.).cuda()
            dist_loss3 = torch.tensor(0.).cuda()
            contra_loss = torch.tensor(0.).cuda()
            contra_loss_ori = torch.tensor(0.).cuda()

            total_time = 0.
            optimize_time = 0.
            clip_time = 0.
            update_time = 0.

            # one step in binary search
            for iteration in range(args.num_iter):
                t1 = time.time()

                ori_logits= self.model(ori_data)
                if isinstance(ori_logits, tuple):  # PointNet
                    features_256 = ori_logits[4]
                    ori_logits = ori_logits[0]
                features_ori = grad_scores(ori_logits, features_256)

                aug_pc1_logits = self.model(aug_pc1)
                if isinstance(aug_pc1_logits, tuple):  # PointNet
                    features_aug1_256 = aug_pc1_logits[4]
                    aug_pc1_logits = aug_pc1_logits[0]
                features_aug1 = grad_scores(aug_pc1_logits, features_aug1_256)    

                aug_pc2_logits = self.model(aug_pc2)
                if isinstance(aug_pc2_logits, tuple):  # PointNet
                    features_aug2_256 = aug_pc2_logits[4]
                    aug_pc2_logits = aug_pc2_logits[0]
                features_aug2 = grad_scores(aug_pc2_logits, features_aug2_256)  
                
                #original adversarial loss
                adv_pc_logits = self.model(adv_data)  # [B, num_classes]
                if isinstance(adv_pc_logits, tuple):  # PointNet
                    features_adv_256 = adv_pc_logits[4]
                    adv_pc_logits = adv_pc_logits[0]
                features_adv = grad_scores(adv_pc_logits, features_adv_256) 

                contra_loss = self.contra_func(adv_pc_logits, aug_pc1_logits, aug_pc2_logits).mean()
                mse = nn.MSELoss()
                fea_loss = -mse(features_ori, features_adv).mean()
                curv_loss = self.curv_func(ori_data, adv_data, normal).mean()

                dist_loss1 = self.dist_func1(adv_data, ori_data)
                dist_loss2 = self.dist_func2(adv_data, ori_data)
                dist_loss3 = self.dist_func3(adv_data, ori_data)
                dist_loss = args.L2_loss_weight * dist_loss1 + args.Chamfer_loss_weight * dist_loss2 + args.Hausdorff_loss_weight * dist_loss3
                
                loss = args.GAMMA * contra_loss + (1-args.GAMMA) * dist_loss + args.curv_loss_weight * curv_loss + 0.1 * fea_loss
                opt.zero_grad()
                loss.backward()
                opt.step()
                if args.is_use_lr_scheduler:
                    lr_scheduler.step()

                t2 = time.time()
                optimize_time += t2 - t1

                # 裁剪，将对抗点云裁剪到扰动范围内
                adv_data.data = self.clip_func(adv_data.detach().clone(), ori_data)

                t3 = time.time()
                clip_time += t3 - t2

                # print
                pred = torch.argmax(adv_pc_logits, dim=1)  # [B]
                success_num = (pred != target).sum().item()

                print('Step {}, iteration {}, success {}/{}\n'
                        'contra_loss: {:.4f}, dist_loss: {:.4f}, curv_loss: {:.4f}, fea_loss: {:.4f}'.
                        format(binary_step, iteration, success_num, B, contra_loss.item(), 
                                 dist_loss.item(), curv_loss.item(), fea_loss.item()))

                # record values!
                dist_val = torch.sqrt(torch.sum(
                    (adv_data - ori_data) ** 2, dim=[1, 2])).\
                    detach().cpu().numpy()  # [B]
                pred_val = pred.detach().cpu().numpy()  # [B]
                input_val = adv_data.detach().cpu().numpy()  # [B, 3, K]

                # update
                for e, (dist, pred, label, ii) in \
                        enumerate(zip(dist_val, pred_val, label_val, input_val)):
                    if dist < o_bestdist[e] and pred != label and args.GAMMA < 0.001:
                        o_bestdist[e] = dist
                        o_bestscore[e] = pred
                        o_bestattack[e] = ii

                t4 = time.time()
                update_time += t4 - t3

                total_time += t4 - t1

                if iteration % 10 == 0:
                    print('total time: {:.2f}, opt: {:.2f}, '
                            'clip: {:.2f}, update: {:.2f}'.
                            format(total_time, optimize_time,
                                    clip_time, update_time))
                    total_time = 0.
                    optimize_time = 0.
                    clip_time = 0.
                    update_time = 0.
                    torch.cuda.empty_cache()

            torch.cuda.empty_cache()

        # end of CW attack
        # fail to attack some examples
        fail_idx = (o_bestscore < 0)
        o_bestattack[fail_idx] = input_val[fail_idx]

        adv_pc = torch.tensor(o_bestattack).to(adv_data)
        adv_pc = self.clip_func(adv_pc, ori_data)

        logits = self.model(adv_pc)
        if isinstance(logits, tuple):  # PointNet
            logits = logits[0]
        preds = torch.argmax(logits, dim=-1)
        
        ori_logits = self.model(ori_data)
        if isinstance(ori_logits, tuple):  # PointNet
            ori_logits = ori_logits[0]
        ori_preds = torch.argmax(ori_logits, dim=-1)
        
        # return final results
        ori_class_accuracy_num = (ori_preds == target).sum().item()
        success_num = (preds != target).sum().item()
        print('Successfully attack {}/{}'.format(success_num, B))
        
        return o_bestdist, adv_pc.detach().cpu().numpy().transpose((0, 2, 1)), ori_class_accuracy_num, success_num
