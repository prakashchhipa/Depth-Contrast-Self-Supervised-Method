import argparse
import logging
import os, sys

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split

from self_supervised.core import ssl_loss, models, pretrain, utility, trainer_depth_contrast
sys.path.append(os.path.dirname(__file__))
from self_supervised.apply import datasets, config, transform, augmentation_strategy
sys.path.append(os.path.dirname(__file__))
import mbv_config



def pretrain_dc():

    parser = argparse.ArgumentParser(description='PyTorch MBV Training')
    parser.add_argument('--lr', default=0.00005, type=float, help='learning rate')
    parser.add_argument('--machine', default=7, type=int, help='define gpu no.')
    parser.add_argument('--patience', default=50, type=int, help='patience for learning rate change')
    parser.add_argument('--batch_size', default=252, type=int, help='batch size')
    parser.add_argument('--temperature', default=0.1, type=float, help='temparature parameter')
    parser.add_argument('--epochs', default=300, type=int, help='epochs')
    parser.add_argument('--description', default="fine_tune", type=str, help='experiment name | description')
    parser.add_argument('--data_path', default="fine_tune", type=str, help=' path for data of specifc fold - Fold 0|1|2|3|4 ')
    
    args = parser.parse_args()

    fold_root = args.data_path
    GPU = torch.device(f"cuda:{args.machine}")
    batch_size = args.batch_size
    epochs = args.epochs
    lr=args.lr
    patience = args.patience
    temperature = args.temperature
    experiment_description = args.description


    train_raw_image_file = fold_root + 'X_raw_train.npy'
    train_ref_image_file = fold_root + 'X_ref_train.npy'

    train_loader = datasets.get_MBV_trainset_loader(
        raw_train_file_path=train_raw_image_file, 
        reflectance_train_file_path= train_ref_image_file,
        batch_size = batch_size, 
        training_method=config.dc,
        transform = [transform.raw_train_transform, transform.ref_train_transform],
        augmentation_strategy = augmentation_strategy.custom_224_random_crop_augmentation,
        image_type = mbv_config.image_both_seprately)

    model = models.EfficientNet_MLP(features_dim=128, v='b2', mlp_dim=2048)
    model = model.cuda(GPU)

    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     'min',
                                                     factor=0.1,
                                                     patience=patience,
                                                     min_lr=5e-4)
    #simclr style                                                     
    criterion = ssl_loss.Depth_Contrast_loss(gpu=GPU, temperature=temperature)
    

    trainer = trainer_depth_contrast.Trainer_DepthContrast(
        experiment_description=experiment_description,
        dataloader=train_loader,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=epochs,
        batch_size=batch_size,
        gpu = GPU,
        criterion=criterion)
    trainer.train()



if __name__ == '__main__':

    pretrain_dc()