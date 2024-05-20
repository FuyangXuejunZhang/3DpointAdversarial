import torch
import torch.nn as nn
import torch.nn.functional as F


class simCLRLoss(nn.Module):

    def __init__(self, temperature=0.07):
        super(simCLRLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, adv_pc_logits, aug_pc1_logits, aug_pc2_logits): # 只考虑负例，生成对抗样本
        # 对特征进行归一化
        aug_pc1_logits = F.normalize(aug_pc1_logits, p=2, dim=1)
        aug_pc2_logits = F.normalize(aug_pc2_logits, p=2, dim=1)
        adv_pc_logits = F.normalize(adv_pc_logits, p=2, dim=1)

        # 正样本对的相似度
        q_k = torch.sum(aug_pc1_logits * aug_pc2_logits, dim=1) # torch.Size([B])
        
        # 负样本对的相似度
        q_i_1 = torch.mm(aug_pc1_logits, adv_pc_logits.t())
        q_i_2 = torch.mm(aug_pc2_logits, adv_pc_logits.t())
        q_i = (q_i_1 + q_i_2) / 2 # torch.Size([B, B])
        
        # # 计算对比损失
        logits = torch.cat([q_k.unsqueeze(1), q_i], dim=1) / self.temperature # torch.Size([B, B+1])
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device) # torch.Size([B])
        loss = nn.CrossEntropyLoss()(logits, labels) #首先对预测得分进行 softmax 操作，将得分转换为概率分布，然后计算预测概率分布与真实标签之间的交叉熵损失
        
        return loss