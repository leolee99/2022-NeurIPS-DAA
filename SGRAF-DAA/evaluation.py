"""Evaluation"""

from __future__ import print_function
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import h5py

from data import get_test_loader
from vocab import Vocabulary, deserialize_vocab
from model import SGRAF
from collections import OrderedDict
from semantic import cider_compute_eval, bleu_compute_eval, meteor_compute_eval, rouge_compute_eval, all_compute_eval

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def _generalConfig(rank: int, worldSize: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "51337" #任意一个没被占用的端口号
    torch.autograd.set_detect_anomaly(False)
    torch.backends.cudnn.benchmark = True
    # random.seed(1234)
    # torch.manual_seed(1234)
    # np.random.seed(1234)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", world_size=worldSize, rank=rank)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.items():
            tb_logger.log_value(prefix + k, v.val, step=step)


def encode_data(model, data_loader, log_step=10, logging=print, split='dev'):
    """Encode all images and captions loadable by `data_loader`
    """
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    # np array to keep all the embeddings
    img_embs = None
    cap_embs = None

    max_n_word = 0

    # CIDEr 
    #tfidf_shape = h5py.File(data_loader.dataset.CIDEr_path,'r')
    tfidf_GT_map = torch.zeros((data_loader.dataset.tfidf_shape[0], 5, data_loader.dataset.tfidf_shape[1]))
    tfidf_map = torch.zeros((data_loader.dataset.tfidf_shape[0], data_loader.dataset.tfidf_shape[1]))
    #####tfidf_GT_map = torch.zeros((len(data_loader.dataset.tfidf_map['TFIDF']), 5, len(data_loader.dataset.tfidf_map['TFIDF'][0])))
    #####tfidf_map = torch.zeros((len(data_loader.dataset.tfidf_map['TFIDF']), len(data_loader.dataset.tfidf_map['TFIDF'][0])))

    # BLEU Meteor Rouge
    # caption_GT = []
    # caption_single = []
    # sortindex = []#np.zeros(len(data_loader.dataset.tfidf_map))

    for i, (images, captions, lengths, ids, _, _) in enumerate(data_loader):
        max_n_word = max(max_n_word, max(lengths))

    for i, (images, captions, lengths, ids, tfidf_GT, tfidf_single) in enumerate(data_loader):
        # make sure val logger is used
        model.logger = val_logger

        # compute the embeddings
        with torch.no_grad():
            img_emb, cap_emb, cap_len = model.forward_emb(images, captions, lengths)
        if img_embs is None:
            img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1), img_emb.size(2)))
            cap_embs = np.zeros((len(data_loader.dataset), max_n_word, cap_emb.size(2)))
            cap_lens = [0] * len(data_loader.dataset)

        # cache embeddings
        img_embs[ids] = img_emb.data.cpu().numpy().copy()
        cap_embs[ids, :max(lengths), :] = cap_emb.data.cpu().numpy().copy()

        # cache CIDEr tfidf
        tfidf_GT_map[ids] = tfidf_GT
        tfidf_map[ids] = tfidf_single
        
        # cache BLEU Meteor Rouge
        # for i in range(len(ids)):
        #     sortindex.append(ids[i])
        #     caption_GT.append(tfidf_GT[i])
        #     caption_single.append(tfidf_single[i])

        for j, nid in enumerate(ids):
            cap_lens[nid] = cap_len[j]

        del images, captions

    # compute CIDEr matrix
    tfidf = (tfidf_GT_map, tfidf_map)

    # compute BLEU Meteor Rouge matrix
    # zipped = zip(sortindex, caption_GT, caption_single)
    # sorted_zip = sorted(zipped, key = lambda x:x[0])
    # sorted_tuple = zip(*sorted_zip)
    # _, tfidf_GT_map, tfidf_map = sorted_tuple
    # tfidf = (tfidf_GT_map, tfidf_map)
    
    # CIDEr
    semantic_map = cider_compute_eval(tfidf, split=split)
    #semantic_map = bleu_compute_eval(tfidf, split=split)
    #semantic_map = meteor_compute_eval(tfidf, split=split)
    #semantic_map = rouge_compute_eval(tfidf, split=split)
    #semantic_map = all_compute_eval(tfidf, split=split)
    semantic_map = semantic_map.view(-1, 5, len(semantic_map)).mean(1).squeeze(1)  #[5000, 25000]
    return img_embs, cap_embs, cap_lens, semantic_map


def evalrank(rank, worldSize, model_path, data_path=None, split='dev', fold5=False):
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """
    _generalConfig(rank, worldSize)
    # load model and options
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']
    save_epoch = checkpoint['epoch']
    print(opt)
    if data_path is not None:
        opt.data_path = data_path

    # load vocabulary used by the model
    vocab = deserialize_vocab('./vocab/%s_vocab.json' % opt.data_name)
    opt.vocab_size = len(vocab)

    # construct model
    model = SGRAF(opt)

    # load model state
    model.load_state_dict(checkpoint['model'])

    print('Loading dataset')
    data_loader = get_test_loader(split, opt.data_name, vocab,
                                  opt.batch_size, opt.workers, opt)
    print("=> loaded checkpoint_epoch {}".format(save_epoch))

    print('Computing results...')
    img_embs, cap_embs, cap_lens, sems = encode_data(model, data_loader, split=split)
    print('Images: %d, Captions: %d' %
          (img_embs.shape[0] / 5, cap_embs.shape[0]))

    if not fold5:
        # no cross-validation, full evaluation
        img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), 5)])

        # record computation time of validation
        start = time.time()
        sims = shard_attn_scores(model, img_embs, cap_embs, cap_lens, opt, shard_size=100)
        #saf_sims = np.load('/home/SGRAF' + '/runs/f30k/checkpoint/ex6/best_5k.npy')
        #sims = (sims + saf_sims) / 2
        #np.save('/home/SGRAF' + '/runs/f30k/checkpoint/ex27/best_5k.npy', sims)
        end = time.time()
        print("calculate similarity time:", end-start)

        # bi-directional retrieval
        r, rt = i2t(img_embs, cap_embs, cap_lens, sims, sems, return_ranks=True)
        ri, rti = t2i(img_embs, cap_embs, cap_lens, sims, sems, return_ranks=True)
        ar = (r[0] + r[1] + r[2]) / 3
        ari = (ri[0] + ri[1] + ri[2]) / 3
        rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
        print("rsum: %.1f" % rsum)
        print("Average i2t Recall: %.1f" % ar)
        print("Image to text: %.1f %.1f %.1f %.1f %.1f %.1f" % r)
        print("Average t2i Recall: %.1f" % ari)
        print("Text to image: %.1f %.1f %.1f %.1f %.1f %.1f" % ri)
    else:
        # 5fold cross-validation, only for MSCOCO
        results = []
        for i in range(5):
            img_embs_shard = img_embs[i * 5000:(i + 1) * 5000:5]
            cap_embs_shard = cap_embs[i * 5000:(i + 1) * 5000]
            cap_lens_shard = cap_lens[i * 5000:(i + 1) * 5000]
            sems_shared = sems[i * 1000:(i + 1) * 1000, i * 5000:(i + 1) * 5000]

            start = time.time()
            sims = shard_attn_scores(model, img_embs_shard, cap_embs_shard, cap_lens_shard, opt, shard_size=100)
            # saf_sims = np.load(data_path + '/runs/COCO/checkpoint/ex48/best_1k_{}.npy'.format(i))
            # sims = (sims + saf_sims) / 2
            #np.save(data_path + '/runs/COCO/checkpoint/ex51/best_1k_{}.npy'.format(i), sims)

            end = time.time()
            print("calculate similarity time:", end-start)

            r, rt0 = i2t(img_embs_shard, cap_embs_shard, cap_lens_shard, sims, sems_shared, return_ranks=True)
            print("Image to text: %.1f %.1f %.1f %.1f %.1f %.1f" % r)
            ri, rti0 = t2i(img_embs_shard, cap_embs_shard, cap_lens_shard, sims, sems_shared, return_ranks=True)
            print("Text to image: %.1f %.1f %.1f %.1f %.1f %.1f" % ri)

            if i == 0:
                rt, rti = rt0, rti0
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            print("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
            results += [list(r) + list(ri) + [ar, ari, rsum]]

        print("-----------------------------------")
        print("Mean metrics: ")
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        print("rsum: %.1f" % (mean_metrics[12] * 6))
        print("Average i2t Recall: %.1f" % mean_metrics[13])
        print("Image to text: %.1f %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[:6])
        print("Average t2i Recall: %.1f" % mean_metrics[14])
        print("Text to image: %.1f %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[6:12])


def shard_attn_scores(model, img_embs, cap_embs, cap_lens, opt, shard_size=100, rank=0):
    n_im_shard = (len(img_embs) - 1) // shard_size + 1
    n_cap_shard = (len(cap_embs) - 1) // shard_size + 1

    sims = np.zeros((len(img_embs), len(cap_embs)))
    for i in range(n_im_shard):
        im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(img_embs))
        for j in range(n_cap_shard):
            if rank==0:
                sys.stdout.write('\r>> shard_attn_scores batch (%d,%d)' % (i, j))
            ca_start, ca_end = shard_size * j, min(shard_size * (j + 1), len(cap_embs))

            with torch.no_grad():
                im = torch.from_numpy(img_embs[im_start:im_end]).float().cuda()
                ca = torch.from_numpy(cap_embs[ca_start:ca_end]).float().cuda()
                l = cap_lens[ca_start:ca_end]
                sim = model.forward_sim(im, ca, l)

            sims[im_start:im_end, ca_start:ca_end] = sim.data.cpu().numpy()
    sys.stdout.write('\n')
    #np.save('visualize/ex11.npy', sims)
    #print(sims)
    return sims


def i2t(images, captions, caplens, sims, sems, npts=None, return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    ASP = ASP_compute(sims, sems)

    npts = images.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)

    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]

        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr, ASP), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr, ASP)


def t2i(images, captions, caplens, sims, sems, npts=None, return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """

    npts = images.shape[0]
    ranks = np.zeros(5 * npts)
    top1 = np.zeros(5 * npts)

    # --> (5N(caption), N(image))
    sims = sims.T
    sems = sems.T
    ASP = ASP_compute(sims, sems)

    for index in range(npts):
        for i in range(5):
            inds = np.argsort(sims[5 * index + i])[::-1]
            ranks[5 * index + i] = np.where(inds == index)[0][0]
            top1[5 * index + i] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr, ASP), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr, ASP)


def ASP_compute(sims, sems):
    """
    sims: (N, 5N) matrix of similarity im-cap
    sems: (N, 5N) matrix of semantic similarity im-cap
    """
    ASP = 0
    N = len(sims)
    sims = torch.from_numpy(sims).cuda()
    for idx in range(N):
        sim_rank = sims[idx].argsort(descending=True).argsort() + 1
        sem_rank = sems[idx].argsort(descending=True).argsort() + 1

        min_rank = torch.min(sim_rank, sem_rank)
        max_rank = torch.max(sim_rank, sem_rank)
        ASP += (min_rank/max_rank).mean()

    ASP = ASP / N

    return 100 * ASP.item()


if __name__ == '__main__':

    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

    worldSize = 1
    # worldsize, model_path, data_path, split, fold5
    mp.spawn(evalrank, (worldSize, "SGRAF/model_best.pth.tar", 'data', "testall", False), worldSize)



