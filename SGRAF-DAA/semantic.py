import os
import string
import torch
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge

def cider_compute_one(tfidf, eps=1e-6):
    """
    Shape
    -----
    Input1 : (torch.Tensor, torch.Tensor)
        :math:`((N, K, D), (N, D))` shape, `N` is the batch size, `K` is the number of the ground-truth captions and `D` is the length of the vocabulary.
    Output: torch.Tensor
        :math:`(N, N)`. The semantic score matrix computed by CIDEr.
    """
    (tfidf_GT, tfidf_single) = tfidf
    sample_N = tfidf_GT.shape[1]
    N = len(tfidf_single)
    tfidf_GT = tfidf_GT.view(-1, tfidf_GT.shape[-1]).to('cuda')
    tfidf_single = tfidf_single.to('cuda')

    cider_map_ = torch.mm(tfidf_GT, tfidf_single.t())

    cider_map = cider_map_.view(N , sample_N, -1)
    cider_map = cider_map.mean(1).squeeze(1)  #i2t: GT 2 single
    
    # cls_weight = cider_map * torch.eye(N).to('cuda')
    # cider_map = cider_map + cls_weight
    #cider_map = cider_map.view(-1, 5, N).mean(1).squeeze(1)  #[5000, 25000]

    return cider_map


def cider_compute(tfidf, eps=1e-6):
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
    cider_map = cider_map.mean(1).squeeze(1)  #i2t: GT 2 single
    
    # cls_weight = cider_map * torch.eye(N).to('cuda')
    # cider_map = cider_map + cls_weight
    #cider_map = cider_map.view(-1, 5, N).mean(1).squeeze(1)  #[5000, 25000]

    return cider_map


def cider_compute_eval(tfidf, split='dev', eps=1e-6):
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
    cider_map = cider_map.mean(1).squeeze(1)  #i2t: GT 2 single
    
    # cls_weight = cider_map * torch.eye(N).to('cuda')
    # cider_map = cider_map + cls_weight

    return cider_map


def bleu_compute(corpus, eps=1e-6):  # train
    (pre_corpus_GT, pre_corpus) = corpus
    
    score = torch.zeros(len(pre_corpus_GT), len(pre_corpus), dtype=torch.float32)
    smooth = SmoothingFunction()
    for idx, reference in enumerate(pre_corpus_GT):
        for idy, candidate in enumerate(pre_corpus):
            score_ = sentence_bleu(reference, candidate, smoothing_function=smooth.method1)
            score[idx][idy] = score_

    return score.to('cuda')


def bleu_compute_eval(corpus, split='dev', eps=1e-6):   # valid and test
    (pre_corpus_GT, pre_corpus) = corpus
    
    score = torch.zeros(len(pre_corpus_GT), len(pre_corpus), dtype=torch.float32)
    score_line = np.zeros(len(pre_corpus), dtype=np.float32)
    smooth = SmoothingFunction()
    N = len(pre_corpus_GT)
    for i in range(0, N):
        idx = i
        reference = pre_corpus_GT[idx]
    # for idx, reference in enumerate(pre_corpus_GT):
        save_path = 'data/semantic/BLEU/{}/{}.npy'.format(split, idx)
        if os.path.exists(save_path):
            score[idx] = torch.Tensor(np.load(save_path))
            continue

        for idy, candidate in enumerate(pre_corpus):
            score_ = sentence_bleu(reference, candidate, smoothing_function=smooth.method1)
            score_line[idy] = score_
            #print('(%d, %d, %f)' % (idx, idy, score_))
        
        print('(%d)' % (idx))
        score[idx] = torch.Tensor(score_line)

        if not os.path.exists(save_path):
            np.save(save_path, score_line)
            #score[idx] = torch.Tensor(np.load(save_path))

    return score.to('cuda')


def meteor_compute(corpus, eps=1e-6):
    (pre_corpus_GT, pre_corpus) = corpus
    
    score = torch.zeros(len(pre_corpus_GT), len(pre_corpus), dtype=torch.float32)

    for idx, reference in enumerate(pre_corpus_GT):
        for idy, candidate in enumerate(pre_corpus):
            score_ = meteor_score(reference, candidate)
            score[idx][idy] = score_
            #score_line.append(score_)
        #score.append(score_line)

    return score.to('cuda')


def meteor_compute_eval(corpus, split='dev', eps=1e-6):
    (pre_corpus_GT, pre_corpus) = corpus

    N = len(pre_corpus_GT)
    score = torch.zeros(len(pre_corpus_GT), len(pre_corpus), dtype=torch.float32)
    score_line = np.zeros(len(pre_corpus), dtype=np.float32)

    for i in range(0, N):

        idx = i
        reference = pre_corpus_GT[idx]
        save_path = 'data/semantic/Meteor/{}/{}.npy'.format(split, idx)

        if os.path.exists(save_path):
            score[idx] = torch.Tensor(np.load(save_path))
            continue

        for idy, candidate in enumerate(pre_corpus):
            score_ = meteor_score(reference, candidate)
            score_line[idy] = score_
            print('(%d, %d, %f)' % (idx, idy, score_))
        
        print('(%d)' % (idx))
        score[idx] = torch.Tensor(score_line)

        if not os.path.exists(save_path):
            np.save(save_path, score_line)
            #score[idx] = torch.Tensor(np.load(save_path))

    return score.to('cuda')


def rouge_compute(corpus, eps=1e-6):
    (pre_corpus_GT, pre_corpus) = corpus
    
    rouge = Rouge()
    score = torch.zeros(len(pre_corpus_GT), len(pre_corpus), dtype=torch.float32)
    for idx, reference in enumerate(pre_corpus_GT):
        for idy, candidate in enumerate(pre_corpus):
            #score_ = rouge.get_scores(candidate, ".".join(reference))
            score_ = rouge.get_scores(candidate, reference, avg=True)
            score[idx][idy] = (score_['rouge-1']['f'] + score_['rouge-2']['f'] + score_['rouge-l']['f'])/3
            #score[idx][idy] = score_['rouge-1']['f']
            #score_line.append(score_)
        #score.append(score_line)

    return score.to('cuda')


def rouge_compute_eval(corpus, split='dev', eps=1e-6):
    (pre_corpus_GT, pre_corpus) = corpus
    
    rouge = Rouge()
    
    score = torch.zeros(len(pre_corpus_GT), len(pre_corpus), dtype=torch.float32)
    score_line = np.zeros(len(pre_corpus), dtype=np.float32)
    N = len(pre_corpus_GT)
    for i in range(0, N):
        idx = i
        reference = pre_corpus_GT[idx]
        save_path = 'data/semantic/Rouge/{}/{}.npy'.format(split, idx)
        if os.path.exists(save_path):
            score[idx] = torch.Tensor(np.load(save_path))
            continue

        for idy, candidate in enumerate(pre_corpus):
            score_ = rouge.get_scores(candidate, reference, avg=True)
            score_line[idy] = (score_['rouge-1']['f'] + score_['rouge-2']['f'] + score_['rouge-l']['f'])/3
            #print('(%d, %d, %f)' % (idx, idy, score_line[idy]))

        score[idx] = torch.Tensor(score_line)
        print('(%d)' % (idx))


        if not os.path.exists(save_path):
            np.save(save_path, score_line)
            #score[idx] = torch.Tensor(np.load(save_path))

    return score.to('cuda')


def all_compute_eval(tfidf, split='dev', eps=1e-6):
    
    (tfidf_GT, tfidf_single) = tfidf
    sample_N = tfidf_GT.shape[1]
    N = len(tfidf_single)
    tfidf_GT = tfidf_GT.view(-1, tfidf_GT.shape[-1]).to('cuda')
    tfidf_single = tfidf_single.to('cuda')

    cider_map_ = torch.mm(tfidf_GT, tfidf_single.t())

    cider_map = cider_map_.view(N , sample_N, -1)
    cider_map = cider_map.mean(1).squeeze(1)  #i2t: GT 2 single
    
    bleu_map = bleu_compute_eval(tfidf, split)
    rouge_map = rouge_compute_eval(tfidf, split)

    semantic_map = (cider_map + bleu_map + rouge_map) / 3

    return semantic_map