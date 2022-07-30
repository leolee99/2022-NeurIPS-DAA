"""MS-COCO image-to-caption retrieval dataset code

reference codes:
https://github.com/pytorch/vision/blob/v0.2.2_branch/torchvision/datasets/coco.py
https://github.com/yalesong/pvse/blob/master/data.py
"""

import os
try:
    import ujson as json
except ImportError:
    import json

from PIL import Image
from pycocotools.coco import COCO

import torch
from torch.utils.data import Dataset
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import numpy as np
import random
import os.path as osp


class CocoCaptionsCap(Dataset):
    """`MS Coco Captions <http://mscoco.org/dataset/#captions-challenge2015>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        ids (list, optional): list of target caption ids
        extra_annFile (string, optional): Path to extra json annotation file (for training)
        extra_ids (list, optional): list of extra target caption ids (for training)
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        instance_annFile (str, optional): Path to instance annotation json (for PMRP computation)

    Example:
        .. code:: python
            import torchvision.datasets as dset
            import torchvision.transforms as transforms
            cap = dset.CocoCaptions(root='dir where images are',
                                    annFile='json annotation file',
                                    transform=transforms.ToTensor())
            print('Number of samples: ', len(cap))
            img, target = cap[3] # load 4th sample
            print("Image Size: ", img.size())
            print(target)
        Output: ::
            Number of samples: 82783
            Image Size: (3L, 427L, 640L)
            [u'A plane emitting smoke stream flying over a mountain.',
            u'A plane darts across a bright blue sky behind a mountain covered in snow',
            u'A plane leaves a contrail above the snowy mountain top.',
            u'A mountain that has a plane flying overheard in the distance.',
            u'A mountain view with a plume of smoke in the background']
    """
    def __init__(self, root, annFile, ids=None,
                 train=False, extra_annFile=None, extra_ids=None,
                 transform=None, instance_annFile=None, tokenizer=None):
        self.root = os.path.expanduser(root)

        self.train=train
        self.tokenizer=tokenizer

        if extra_annFile:
            self.coco = COCO()
            with open(annFile, 'r') as fin1, open(extra_annFile, 'r') as fin2:
                dataset = json.load(fin1)
                extra_dataset = json.load(fin2)
                if not isinstance(dataset, dict) or not isinstance(extra_dataset, dict):
                    raise TypeError('invalid type {} {}'.format(type(dataset),
                                                                type(extra_dataset)))
                if set(dataset.keys()) != set(extra_dataset.keys()):
                    raise KeyError('key mismatch {} != {}'.format(list(dataset.keys()),
                                                                  list(extra_dataset.keys())))
                for key in ['images', 'annotations']:
                    dataset[key].extend(extra_dataset[key])
            self.coco.dataset = dataset
            self.coco.createIndex()
        else:
            self.coco = COCO(annFile)
        self.ids = list(self.coco.anns.keys()) if ids is None else list(ids)
        if extra_ids is not None:
            self.ids += list(extra_ids)
        self.ids = [int(id_) for id_ in self.ids]
        self.transform = transform

        self.all_image_ids = set([self.coco.loadAnns(annotation_id)[0]['image_id'] for annotation_id in self.ids])
        txts, txts_ids = [], {}
        #txts_i2c = {}
        for idx, annotation_id in enumerate(self.ids):
            txts.append(self.coco.loadAnns(annotation_id)[0]['caption'])
            txts_ids[annotation_id] = idx
            #txts_i2c[annotation_id] = self.coco.loadAnns(annotation_id)[0]['caption']
        self.tfidf_map = self.tfidf_compute(txts)
        self.txts_ids = txts_ids
        

        self.image2tfidf = {}
        self.i2t_id, self.t2i_id = {}, {}
        for image_ids in self.all_image_ids:
            anns, mapped_anns = [], []
            annotations = self.coco.imgToAnns[image_ids]
            for idx, ann in enumerate(annotations):
                if idx < 5:
                    anns.append(ann['id'])
                    mapped_anns.append(self.txts_ids[ann['id']])
            while len(anns) < 5:
                mapped_anns.append(mapped_anns[0])
            self.image2tfidf[image_ids] = np.array(mapped_anns)
            self.i2t_id[image_ids] = anns
            for aid in anns:
                self.t2i_id[aid] = [image_ids]

        iid_to_cls = {}
        if instance_annFile:
            with open(instance_annFile) as fin:
                instance_ann = json.load(fin)
            for ann in instance_ann['annotations']:
                image_id = int(ann['image_id'])
                code = iid_to_cls.get(image_id, [0] * 90)
                code[int(ann['category_id']) - 1] = 1
                iid_to_cls[image_id] = code

            seen_classes = {}
            new_iid_to_cls = {}
            idx = 0
            for k, v in iid_to_cls.items():
                v = ''.join([str(s) for s in v])
                if v in seen_classes:
                    new_iid_to_cls[k] = seen_classes[v]
                else:
                    new_iid_to_cls[k] = idx
                    seen_classes[v] = idx
                    idx += 1
            iid_to_cls = new_iid_to_cls

            if self.all_image_ids - set(iid_to_cls.keys()):
                print(f'Found mismatched! {self.all_image_ids - set(iid_to_cls.keys())}')

        self.iid_to_cls = iid_to_cls
        self.n_images = len(self.all_image_ids)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is a caption for the annotation.
        """
        coco = self.coco
        annotation_id = self.ids[index]
        annotation = coco.loadAnns(annotation_id)[0]
        image_id = annotation['image_id']

        caption = annotation['caption']
        caption_tokens = self.tokenizer.basic_tokenizer.tokenize(caption)

        # Convert caption (string) to word ids (with Size Augmentation at training time)
        target = self.process_caption(self.tokenizer, caption_tokens, self.train)
        tfidf_GT = self.tfidf_map[self.image2tfidf[image_id]]

        tfidf_single = self.tfidf_map[self.txts_ids[annotation_id]]

        path = coco.loadImgs(image_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if len(target) > 80:
            target = target[:80]

        return img, target, annotation_id, image_id, tfidf_GT, tfidf_single

    def __len__(self):
        return len(self.ids)


    def tfidf_compute(self, corpus, eps=1e-6):
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(corpus)
        transformer = TfidfTransformer()
        tfidf_mat = transformer.fit_transform(X).toarray()

        return tfidf_mat

    def process_caption(self, tokenizer, tokens, train=True):
        output_tokens = []
        deleted_idx = []

        for i, token in enumerate(tokens):
            sub_tokens = tokenizer.wordpiece_tokenizer.tokenize(token)
            prob = random.random()

            if prob < 0.20 and train:  # mask/remove the tokens only during training
                prob /= 0.20

                # 50% randomly change token to mask token
                if prob < 0.5:
                    for sub_token in sub_tokens:
                        output_tokens.append("[MASK]")
                # 10% randomly change token to random token
                elif prob < 0.6:
                    for sub_token in sub_tokens:
                        output_tokens.append(random.choice(list(tokenizer.vocab.keys())))
                        # -> rest 10% randomly keep current token
                else:
                    for sub_token in sub_tokens:
                        output_tokens.append(sub_token)
                        deleted_idx.append(len(output_tokens) - 1)
            else:
                for sub_token in sub_tokens:
                    # no masking token (will be ignored by loss function later)
                    output_tokens.append(sub_token)

        if len(deleted_idx) != 0:
            output_tokens = [output_tokens[i] for i in range(len(output_tokens)) if i not in deleted_idx]

        output_tokens = ['[CLS]'] + output_tokens + ['[SEP]']
        target = tokenizer.convert_tokens_to_ids(output_tokens)
        target = torch.Tensor(target)
        return target
    