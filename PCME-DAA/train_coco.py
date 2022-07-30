"""End-to-end training code for cross-modal retrieval tasks.

PCME
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import os
import fire

import torch.backends.cudnn as cudnn
import torch

from config import parse_config
from datasets import prepare_coco_dataloaders
from engine import TrainerEngine
from engine import COCOEvaluator
from logger import PythonLogger

from transformers import BertTokenizer

def pretrain(config, dataloaders, logger):
    logger.log('start pretrain')
    engine = TrainerEngine()
    engine.set_logger(logger)

    config.model.img_finetune = False
    config.model.txt_finetune = False

    _dataloaders = dataloaders.copy()

    val_epochs = config.train.get('pretrain_val_epochs', 1)
    warmup_epochs = config.train.get('pretrain_warmup_epochs', 0)
    evaluator = COCOEvaluator(eval_method=config.model.get('eval_method', 'matmul'),
                              verbose=False,
                              eval_device=torch.device('cuda'),
                              n_crossfolds=5)
    engine.create(config, evaluator)

    n_epoch = config.train.pretrain_epochs
    start_epoch = 0

    if config.train.restore_flag and os.path.exists(config.train.restore_pretrain_path):
        print("Restoring model from {}...".format(config.train.restore_pretrain_path))
        engine.load_models(config.train.restore_pretrain_path,
                          load_keys=None)
        start_epoch = config.train.restore_epoch

    engine.train(tr_loader=_dataloaders.pop('train'),
                 n_epochs=n_epoch,
                 start_epochs=start_epoch,
                 warmup_epochs = warmup_epochs,
                 val_loaders=_dataloaders,
                 val_epochs=val_epochs,
                 model_save_to=config.train.pretrain_save_path,
                 best_model_save_to=config.train.best_pretrain_save_path,
                 epoch_model_path=config.train.epoch_model_save_path)


def finetune(config, pretrain_path, dataloaders, logger):
    logger.log('start finetune')
    engine = TrainerEngine()
    engine.set_logger(logger)

    config.model.img_finetune = True
    config.model.txt_finetune = True
    config.optimizer.learning_rate *= config.train.get('finetune_lr_decay', 0.1)

    _dataloaders = dataloaders.copy()

    val_epochs = config.train.get('val_epochs', 1)
    warmup_epochs = config.train.get('finetune_warmup_epochs', 0)

    evaluator = COCOEvaluator(eval_method=config.model.get('eval_method', 'matmul'),
                              verbose=False,
                              eval_device='cuda',
                              n_crossfolds=5)
    engine.create(config, evaluator)
    start_epoch = 0

    if config.train.restore_flag and os.path.exists(config.train.restore_finetune_path):
        print("Restoring model from {}...".format(config.train.restore_finetune_path))
        engine.load_models(config.train.restore_finetune_path,
                          load_keys=None)
        start_epoch = config.train.restore_epoch

    elif os.path.exists(pretrain_path):
       engine.load_models(pretrain_path,
                          load_keys=['model', 'criterion'])

    engine.train(tr_loader=_dataloaders.pop('train'),
                 n_epochs=config.train.finetune_epochs,
                 start_epochs=start_epoch,
                 warmup_epochs=warmup_epochs,
                 val_loaders=_dataloaders,
                 val_epochs=val_epochs,
                 model_save_to=config.train.model_save_path,
                 best_model_save_to=config.train.best_model_save_path,
                 epoch_model_path=config.train.epoch_model_save_path)


def main(config_path,
         dataset_root,
         **kwargs):
    """Main interface for the training.

    Args:
        config_path: path to the configuration file
        dataset_root: root for the dataset
        vocab_path: vocab filename

        Other configurations:
            you can override any pcme configuration in the command line!
            try, --<depth1>__<depth2>. E.g., --dataloader__batch_size 32
    """
    logger = PythonLogger()

    config = parse_config(config_path,
                          strict_cast=False,
                          **kwargs)
    torch.cuda.set_device(config.train.gpus)
    cudnn.benchmark = True
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataloaders = prepare_coco_dataloaders(config.dataloader,
                                                  dataset_root, tokenizer)

    pretrain(config, dataloaders, logger)
    finetune(config, config.train.best_pretrain_save_path,
             dataloaders, logger)


if __name__ == '__main__':
    fire.Fire(main)
