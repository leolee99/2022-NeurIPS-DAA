import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class DAA(nn.Module):
    r"""Creates an approach that can optimize object directly.
    """
    def __init__(self):
        super().__init__()
        pass
    
    def sigmoid(self, tensor, temp):
        """ temperature controlled sigmoid

        takes as input a torch tensor (tensor) and passes it through a sigmoid, controlled by temperature: temp
        """
        exponent = -tensor / temp
        # clamp the input tensor for stability
        exponent = torch.clamp(exponent, min=-50, max=50)
        y = 1.0 / (1.0 + torch.exp(exponent))
        return y


    def coco_cider_compute(self, tfidf, eps=1e-6):
        """
        Shape
        -----
        Input1 : (torch.Tensor, torch.Tensor)
            :math:`((N, K, D), (N, D))` shape, `N` is the batch size, `K` is the number of the ground-truth captions and `D` is the length of the vocabulary.
        Output: torch.Tensor
            :math:`(N, N)`. The semantic score matrix computed by CIDEr.
        """
        (tfidf_GT, tfidf_single) = tfidf
        N = len(tfidf_single)
        tfidf_GT = tfidf_GT.view(-1, tfidf_GT.shape[-1]).to('cuda')
        tfidf_single = tfidf_single.to('cuda')

        cider_map_ = torch.mm(tfidf_GT, tfidf_single.t())

        cider_map = cider_map_.view(N , 5, -1)
        cider_map = cider_map.mean(1).squeeze(1)
        # cls_weight = cider_map * torch.eye(N).to('cuda')
        # cider_map = cider_map + cls_weight

        return cider_map

    def coco_DAA(self, input1, input2, cider_map):
        """
        Shape
        -----
        Input1 : torch.Tensor
            :math:`(N, K, D)` shape, `N` is the batch size, `K` is the number of samples and `D` is the size of a sample.
        Input2 : torch.Tensor
            :math:`(N, K, D)` shape, `N` is the batch size, `K` is the number of samples and `D` is the size of a sample.
        Input3 : torch.Tensor
            :math:`(N, N)` shape, `N` is the batch size, `K` is the number of samples and `D` is the size of a sample.
        Output: torch.Tensor
            :math:`(1)`.
        """
        sample_num = input1.shape[1]

        input1 = input1.view(-1, input1.shape[-1])
        input2 = input1.view(-1, input2.shape[-1]).t()
        N = len(input1)
        
        cider_map = cider_map.unsqueeze(-1).repeat(1, 1, sample_num).view(-1, N)
        cider_map = cider_map.unsqueeze(1).repeat(1, sample_num, 1).view(N, -1).to('cuda')

        score = torch.mm(input1, input2)

        score_broad = score.unsqueeze(-1)
        score_expand = score[None, :, :]
        score_diff = score_broad - score_expand

        cider_broad = cider_map.unsqueeze(-1)
        cider_expand = cider_map[None, :, :]
        cider_diff = cider_broad - cider_expand

        diag_mask = (1 - torch.eye(N)).unsqueeze(1).to('cuda')  #mask diag to 0

        cider_rank = diag_mask * self.sigmoid(tensor = cider_diff, temp = 0.01)
        cider_rank_all = cider_rank.sum(0) + 1
        score_rank = diag_mask * self.sigmoid(tensor = score_diff, temp = 0.01)
        score_rank_all = score_rank.sum(0) + 1
    
        min_rank = torch.min(cider_rank_all, score_rank_all)
        max_rank = torch.max(cider_rank_all, score_rank_all)
    
        ASP = (min_rank / max_rank).sum() / (N**2)

        return 1 - ASP


    def Smooth_AP(self, input1, input2):
        sample_num = input1.shape[1]
        input1 = input1.view(-1, input1.shape[-1])
        input2 = input2.view(-1, input2.shape[-1]).t()
        N = len(input1)

        matched = torch.eye(N).to('cuda')  #label_mask
        for i in range(0, N, sample_num):
            matched[i:i + sample_num, i:i + sample_num] = 1
        score = torch.mm(input1, input2)

        score_broad = score.unsqueeze(-1)
        score_expand = score[None, :, :]
        score_diff = score_broad - score_expand

        diag_mask = (1 - torch.eye(N)).unsqueeze(1).to('cuda')  #mask diag to 0
        pos_mask_label = matched.repeat(N, 1, 1) #mask for every tensor
        pos_mask_loc = matched.unsqueeze(-1)

        score_rank = diag_mask * self.sigmoid(tensor = score_diff, temp = 0.01)
        score_rank_all = score_rank.sum(0) + 1
        score_rank_pos = ((score_rank * pos_mask_label * pos_mask_loc).sum(0) + 1) * matched
        AP = (score_rank_pos / score_rank_all).sum() / (sample_num * N)

        return 1 - AP