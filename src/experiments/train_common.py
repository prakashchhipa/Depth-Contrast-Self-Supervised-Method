from distutils.log import error
import errno
import numpy as np
import json
import argparse
import time
from tqdm import tqdm
import cv2
import logging
import sys, os

import torch
import torchvision
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms

from sklearn.metrics import f1_score

from sampler import BalancedBatchSampler
from models import Resnext_Model, Densenet_Model
from dataloader import MBV_Dataset
sys.path.append(os.path.dirname(__file__))
from train_util import Train_Util
sys.path.append(os.path.dirname(__file__))
from utils import *
from mbv_config import MBV_Config
import mbv_config
import argparse

def train():
    
    parser = argparse.ArgumentParser(description='PyTorch MBV Training')
    parser.add_argument('--lr', default=0.00001, type=float, help='learning rate')
    parser.add_argument('--wd', default=5e-3, type=float, help='weight decay')
    parser.add_argument('--architecture', default="resnext", type=str, help='architecture - resnext | densenet')
    parser.add_argument('--machine', default=7, type=int, help='define gpu no.')
    parser.add_argument('--patience', default=10, type=int, help='patience for learning rate change')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--input_size', default=225, type=int, help='input image')
    parser.add_argument('--epochs', default=100, type=int, help='epochs')
    parser.add_argument('--description', default="fine_tune", type=str, help='experiment name | description')
    parser.add_argument('--data_path', default="fine_tune", type=str, help=' path for data of specifc fold - Fold 0|1|2|3|4 ')
    
    args = parser.parse_args()    


    batch_size = args.batch_size
    image_size = args.input_size
    LR = args.lr
    patience = args.patience
    weight_decay = args.wd
    fold_root = args.data_path
    device = torch.device(f"cuda:{args.machine}")
    epochs = args.epochs
    experiment_description = args.description
    architecture = args.architecture

    raw_train_transform = transforms.Compose([
        transforms.RandomCrop((image_size,image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
        ])
    ref_train_transform = transforms.Compose([
        transforms.RandomCrop((image_size,image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
        ])
    val_transform = transforms.Compose([
        transforms.Resize((image_size*4,image_size*4)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
        ])
      
    

    train_raw_image_file = fold_root + 'X_raw_train.npy'
    train_ref_image_file = fold_root + 'X_ref_train.npy'
    train_label_file = fold_root + 'Y_train.npy'
    val_raw_image_file = fold_root + 'X_raw_val.npy'
    val_ref_image_file = fold_root + 'X_ref_val.npy'
    val_label_file = fold_root + 'Y_val.npy'

    train_dataset = MBV_Dataset(raw_train_file_path = train_raw_image_file , reflectance_train_file_path=train_ref_image_file, label_file_path=train_label_file, transform= [raw_train_transform, ref_train_transform])
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, sampler=None) #, sampler = BalancedBatchSampler(train_dataset)
    val_dataset = MBV_Dataset(raw_train_file_path = val_raw_image_file , reflectance_train_file_path=val_ref_image_file, label_file_path=val_label_file, transform= [val_transform])
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle=False, sampler=None)


    if architecture == "resnext":
        downstream_task_model = Resnext_Model( pretrained=True)
    elif architecture == "densenet":
        downstream_task_model = Densenet_Model(pretrained=True)
    else:
        raise error ("invalid architecture name")
    
    downstream_task_model = downstream_task_model.to(device)


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(downstream_task_model.parameters(), lr=LR, weight_decay= weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1 ,patience=patience, min_lr= 5e-3)


    writer = SummaryWriter(log_dir=mbv_config.tensorboard_path+experiment_description)
    train_util = Train_Util(experiment_description = experiment_description,image_type=mbv_config.image_both, epochs = epochs, model=downstream_task_model, device=device, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer, criterion=criterion, batch_size=batch_size,scheduler=scheduler, writer=writer)
    train_util.train_and_evaluate()




if __name__ == "__main__":
    train()
