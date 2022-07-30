"""CUB Caption image-to-caption retrieval dataset code

PCME
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import torch


class CUBCaption(Dataset):
    """CUB Captions Dataset.

    Args:
        image_root (string): Root directory where images are downloaded to.
        caption_root (string): Root directory where captions are downloaded to.
        target_classes (str or list): target class ids
            - if str, it is the name of the file with target classes (line by line)
            - if list, it is directly used to get classes
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        omit_ids (str, optional): Path of file with the list of image ids to omit,
            if not specified, use all images in the target classes.
        ids (str, optional): Path of file with the list of target image ids,
            if not specified, use all images in the target classes.
    """
    def __init__(self, image_root, caption_root,
                 target_classes,
                 transform=None, target_transform=None,
                 omit_ids=None, ids=None):
        if omit_ids and ids:
            raise ValueError('omit ids and ids cannot be defined at the same time.')
        if omit_ids:
            with open(omit_ids) as fin:
                omit_ids = set([line.strip() for line in fin])
        else:
            omit_ids = set()
        if ids:
            with open(ids) as fin:
                ids = set([line.strip() for line in fin])

        self.image_root = os.path.expanduser(image_root)
        self.caption_root = os.path.expanduser(caption_root)

        if isinstance(target_classes, str):
            with open(target_classes) as fin:
                _classes = [int(line.strip().split('.')[0]) - 1 for line in fin]
            target_classes = _classes

        target_classes = set(list(target_classes))
        if (target_classes - set(range(200))):
            raise ValueError(f'target classes should be an integer array between 0-199, but {target_classes}')
        print(f'prepare cub dataset with {len(target_classes)} classes')

        targets = []
        txts = []
        index_to_class = {}
        class_to_indices = {}
        class_to_img_indices = {}
        idx = 0
        n_images = 0
        for bird_name in os.listdir(image_root):
            cls_num = int(bird_name.split('.')[0]) - 1
            if cls_num in target_classes:
                _target = []
                txt = []
                for fname in os.listdir(os.path.join(image_root, bird_name)):
                    if os.path.join(bird_name, fname) in omit_ids:
                        continue

                    if ids and os.path.join(bird_name, fname) not in ids:
                        continue

                    txt_fname = os.path.join(caption_root, bird_name, fname.replace('jpg', 'txt'))
                    with open(txt_fname) as fin:
                        captions = [line.strip() for line in fin]

                    n_images += 1
                    class_to_img_indices.setdefault(cls_num, []).append(n_images)
                    for caption in captions:
                        _target.append(
                            (os.path.join(image_root, bird_name, fname), caption)
                        )
                        txt.append(caption)
                        index_to_class[idx] = cls_num
                        class_to_indices.setdefault(cls_num, []).append(idx)
                        idx += 1
                targets.extend(_target)
                txts.extend(txt)

        #tfidf = np.sum(tfidf_mat.toarray(),axis=1)

        self.target_classes_dict = {}
        target_classes_list = list(target_classes)
        for i in range(len(target_classes_list)):
            self.target_classes_dict[target_classes_list[i]] = i

        self.targets = targets
        self.txts = txts
        self.target_classes = target_classes
        self.index_to_class = index_to_class
        self.class_to_indices = class_to_indices
        self.class_to_img_indices = class_to_img_indices

        self.n_images = n_images

        self.transform = transform
        self.target_transform = target_transform

        self.cider_map = self.cider_compute(txts).cpu()
        torch.cuda.empty_cache()

    def __getitem__(self, index):
        img_path, txt = self.targets[index]

        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            caption = self.target_transform(txt)
        
        trans_index = self.target_classes_dict[self.index_to_class[index]]

        return img, caption, self.index_to_class[index], index, trans_index

    def __len__(self):
        return len(self.targets)

    # def cider_compute(self, corpus):
    #     vectorizer = CountVectorizer()
    #     X = vectorizer.fit_transform(corpus)
    #     transformer = TfidfTransformer()
    #     tfidf_mat = transformer.fit_transform(X).toarray()
    #     cider_map = []
    #     for cls in self.target_classes:
    #         print("formatting class {} ...".format(cls))
    #         cider_cls = []
    #         groundtruths = tfidf_mat[np.array(self.class_to_indices[cls])]
    #         length = len(groundtruths)
    #         for idx, candidate in enumerate(tfidf_mat):
    #             if cls == self.index_to_class[idx]:
    #                 print('same')
    #             cider = 0
    #             for groundtruth in groundtruths:
    #                 cider_tmp = np.dot(np.transpose(groundtruth), candidate) / np.sqrt(np.power(candidate, 2).sum() * np.power(groundtruth, 2).sum())
    #                 cider += cider_tmp
    #             cider = cider / length
    #             cider_cls.append(cider)
    #             if idx % 500 == 0:
    #                 print("{}/{} has completed in {}th class!".format(idx, len(tfidf_mat), cls))
    #         cider_map.append(cider_cls)
    #     return cider_map

    def cider_compute(self, corpus, eps=1e-6):
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(corpus)
        transformer = TfidfTransformer()
        tfidf_mat = transformer.fit_transform(X).toarray()
        cat_flag = False

        for cls in self.target_classes:
            #print("formatting class {} ...".format(cls))
            
            cls_indices = torch.LongTensor(self.class_to_indices[cls]).cuda()
            groundtruths = torch.from_numpy(tfidf_mat[np.array(self.class_to_indices[cls])]).cuda() # M x L
            candidate = torch.from_numpy(tfidf_mat).cuda()  # A x L
            cider_tmp = torch.mm(groundtruths, candidate.t()) / torch.sqrt((candidate**2).sum(1).unsqueeze(0) * (groundtruths**2).sum(1).unsqueeze(1))
            cider_score = -torch.log((1 - cider_tmp) + eps)
            cider = cider_score.mean(0).t().unsqueeze(-1)
            cls_mean = cider.index_select(0, cls_indices).mean()
            cls_weight = torch.zeros(cider.shape[0], 1).cuda()
            cls_weight = cls_weight.index_fill(0, cls_indices, cls_mean)
            cider = cider + cls_weight

            if cat_flag:
                cider_map = torch.cat((cider_map, cider), 1)
            else: 
                cider_map = cider
                cat_flag = True
        print("Cider_map have been initialized!")
        return cider_map
                


class CUBSampler(Sampler):
    """ Sampler for CUB Captions training.

    Args:
        dataset (CUBCaption object): dataset object to apply the sampler.
        batch_size (int): batch size.
        adjust_epoch (bool): if true, the iterations for one epoch is re-calculated.
    """
    def __init__(self, dataset, batch_size, adjust_epoch=True):
        self.dataset = dataset
        self.batch_size = batch_size

        self.target_classes = dataset.target_classes
        if batch_size != len(self.target_classes):
            raise ValueError(f'{batch_size} != {len(self.target_classes)}')
        self.index_to_class = dataset.index_to_class
        self.class_to_indices = dataset.class_to_indices
        self.n_items = len(self.index_to_class)

        if adjust_epoch:
            self.n_iters = int(self.n_items / len(self.target_classes))
        else:
            self.n_iters = self.n_items

    def __iter__(self):
        batch = []
        indices = list(range(self.n_items))

        np.random.shuffle(indices)
        for cur_iter, idx in enumerate(indices):
            batch = [idx]
            pos_cls = self.index_to_class[idx]  # bird cls
            for cls_num, _indices in self.class_to_indices.items():
                if cls_num == pos_cls:
                    continue
                else:
                    batch.append(np.random.choice(_indices))
            np.random.shuffle(batch)
            yield batch
            if cur_iter > self.n_iters:
                raise StopIteration

    def __len__(self):
        return self.n_iters
