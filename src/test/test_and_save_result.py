import numpy as np
import json
import argparse
import time
from tqdm import tqdm
import cv2
import logging
import sys, os, csv

import torch
import torchvision
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
#from torchsampler import ImbalancedDatasetSampler

#import albumentations as A
from sklearn.metrics import f1_score,matthews_corrcoef

#from sampler import BalancedBatchSampler
from models import Densenet_Model, EfficientNet_Model, Resnext_Model

#import train_util
from dataloader import MBV_Dataset
sys.path.append(os.path.dirname(__file__))

from utils import *
from mbv_config import MBV_Config
import mbv_config
from pathlib import Path


def get_metrics_from_confusion_matrix(confusion_matrix_epoch):

        #classwise precision
        epoch_classwise_precision_manual_cpu = np.array(confusion_matrix_epoch.diag().cpu())/np.array(confusion_matrix_epoch.cpu()).sum(axis=0)
        epoch_classwise_precision_manual_cpu = np.nan_to_num(epoch_classwise_precision_manual_cpu, nan=0, neginf=0, posinf=0)
        #classwise recall
        epoch_classwise_recall_manual_cpu = np.array(confusion_matrix_epoch.diag().cpu())/np.array(confusion_matrix_epoch.cpu()).sum(axis=1)
        epoch_classwise_recall_manual_cpu = np.nan_to_num(epoch_classwise_recall_manual_cpu, nan=0, neginf=0, posinf=0)
        #classwise f1
        epoch_classwise_f1_manual_cpu = (2*(epoch_classwise_precision_manual_cpu*epoch_classwise_recall_manual_cpu))/(epoch_classwise_precision_manual_cpu + epoch_classwise_recall_manual_cpu)
        epoch_classwise_f1_manual_cpu = np.nan_to_num(epoch_classwise_f1_manual_cpu, nan=0, neginf=0, posinf=0)
        #weighted average F1
        epoch_avg_f1_manual = np.sum(epoch_classwise_f1_manual_cpu*np.array(confusion_matrix_epoch.cpu()).sum(axis=1))/np.array(confusion_matrix_epoch.cpu()).sum(axis=1).sum()
        #accuracy
        epoch_acc_manual = 100*np.sum(np.array(confusion_matrix_epoch.diag().cpu()))/np.sum(np.array(confusion_matrix_epoch.cpu()))

        return epoch_avg_f1_manual, epoch_acc_manual, epoch_classwise_precision_manual_cpu, epoch_classwise_recall_manual_cpu, epoch_classwise_f1_manual_cpu

def save_results(eval_result_path,fold_root, dataset_portion, model_name, weighted_f1, classwise_precision, classwise_recall, classwise_f1, confusion_matrix):
    
    
    eval_path_until_fold = eval_result_path + fold_root.split("/")[-2]
    Path(eval_path_until_fold).mkdir(parents=True, exist_ok=True)
    eval_path_until_model = eval_path_until_fold + "/" + model_name
    Path(eval_path_until_model).mkdir(parents=True, exist_ok=True)
    eval_path_until_dataset_portion = eval_path_until_model + "/" + dataset_portion
    Path(eval_path_until_dataset_portion).mkdir(parents=True, exist_ok=True)
    
    #precision-recall-F1
    with open(eval_path_until_dataset_portion + "/precision_recall_f1.csv", 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_NONE)
        #labels = mbv_config.label_list
        #labels.insert(0,'')
        #wr.writerow(labels)
        classwise_precision = list(classwise_precision)
        classwise_precision.insert(0, 'classwise_precision')
        wr.writerow(classwise_precision)
        classwise_recall = list(classwise_recall)
        classwise_recall.insert(0, 'classwise_recall')
        wr.writerow(classwise_recall)
        classwise_f1 = list(classwise_f1)
        classwise_f1.insert(0, 'classwise_f1')
        wr.writerow(classwise_f1)
        #weightaed F1
        wr.writerow(['weighted_f1',weighted_f1])
    
    #confusion matrix
    with open(eval_path_until_dataset_portion + "/confusion_matrix.csv", 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_NONE)
        #labels = mbv_config.label_list
        #labels.insert(0,'')
        #print(labels)
        #wr.writerow(labels)
        confusion_matrix = confusion_matrix.cpu().detach().numpy()
        '''row = list(confusion_matrix[0])
        print('row0', row)
        row.insert(0, mbv_config.label_list[0])
        print('row0-aftter', row)
        wr.writerow(row)
        row = list(confusion_matrix[1])
        print('row1', row)
        row.insert(0, mbv_config.label_list[1])
        print('row1 - after', row)
        wr.writerow(row)'''


        for idx in range(0, 7):
            row = []
            row = list(confusion_matrix[idx])
            #row.insert(0, mbv_config.label_list[idx])
            print(row)
            wr.writerow(row)

def test(eval_result_path, fold_root, dataset_portion, model_name, model, test_loader, device, image_type):
        confusion_matrix_val = torch.zeros(len(mbv_config.label_list), len(mbv_config.label_list))
        mcc_sum = 0
        model.eval()
        
        with torch.no_grad():
            for raw_image, ref_image, label in tqdm(test_loader):
                raw_image = np.asarray(raw_image)
                ref_image = np.asarray(ref_image)
                #image = cv2.merge((raw_image, ref_image, raw_image))

                if image_type == mbv_config.image_both:
                    image = cv2.merge((raw_image, ref_image, raw_image))
                elif image_type == mbv_config.image_raw:
                    image = cv2.merge((raw_image, raw_image, raw_image))
                elif image_type == mbv_config.image_ref:
                    image = cv2.merge((ref_image, ref_image, ref_image))  
                
                inputs = torch.from_numpy(image).to(device)
                inputs = inputs.permute(0,1,4,2,3)

                   

                inputs = inputs.squeeze(1)
                target = label.to(device)
                
                outputs = model(inputs)

                #Accuracy
                _, predicted = torch.max(outputs.data, 1)
                predicted = predicted.to(device)
                
                #Update confusion matrix for each batch for current epoch
                for targetx, predictedx in zip(target.view(-1), predicted.view(-1)):
                    confusion_matrix_val[targetx.long(), predictedx.long()] += 1

                #MCC
                mcc_sum += matthews_corrcoef(target.cpu(), predicted.cpu())
                
        
        weighted_f1, accuracy, classwise_precision, classwise_recall, classwise_f1 = get_metrics_from_confusion_matrix(confusion_matrix_val)
        mcc = mcc_sum/len(test_loader)



        print(eval_result_path, fold_root, dataset_portion, model_name)
        print('Testset classwise precision', classwise_precision)
        print('Testset classwise recall', classwise_recall)
        print('Testset classwise f1', classwise_f1)

        print('Testset Weighted F1',weighted_f1)
        print('Testset Accuracy', accuracy)

        save_results(eval_result_path, fold_root, dataset_portion, model_name, weighted_f1, classwise_precision, classwise_recall, classwise_f1, confusion_matrix_val)

        return mcc, weighted_f1, accuracy,classwise_precision,classwise_recall,classwise_f1

        
def test_trained_model(image_type = mbv_config.image_both,eval_result_path = mbv_config.evaluation_path_supervised, fold_root=mbv_config.data_path_fold0, dataset_portion=mbv_config.test, model_name = mbv_config.EfficientNet_b2, pretrained_weights_file_path = None, device=mbv_config.gpu1):
    
    batch_size = 16
    image_size = 224
    fold_root = fold_root
    device = device
   
   
    val_transform = transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
        ])
    
    test_raw_image_file = None
    test_ref_image_file = None
    test_label_file = None

    if dataset_portion == mbv_config.train:
        test_raw_image_file = fold_root + 'X_raw_train.npy'
        test_ref_image_file = fold_root + 'X_ref_train.npy'
        test_label_file = fold_root + 'Y_train.npy'
    elif dataset_portion == mbv_config.test:
        test_raw_image_file = fold_root + 'X_raw_test.npy'
        test_ref_image_file = fold_root + 'X_ref_test.npy'
        test_label_file = fold_root + 'Y_test.npy'
    elif dataset_portion == mbv_config.val:
        test_raw_image_file = fold_root + 'X_raw_val.npy'
        test_ref_image_file = fold_root + 'X_ref_val.npy'
        test_label_file = fold_root + 'Y_val.npy'


    test_dataset = MBV_Dataset(raw_train_file_path = test_raw_image_file , reflectance_train_file_path=test_ref_image_file, label_file_path=test_label_file, transform= [val_transform])
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False, sampler=None)
    
    model = None
    if model_name == mbv_config.EfficientNet_b2:
        model = EfficientNet_Model()
    elif model_name == mbv_config.DenseNet_121:
        model = Densenet_Model(pretrained=True)
    elif model_name == mbv_config.ResNext_50_32x4d:
        model = Resnext_Model()
        
    model.load_state_dict(torch.load(pretrained_weights_file_path))
    model = model.to(device)
    image_type = image_type

    test(eval_result_path = eval_result_path, fold_root=fold_root,dataset_portion=dataset_portion,model_name = model_name, model=model, test_loader=test_loader, device=device, image_type=image_type)
    


if __name__ == "__main__":

    
    test_trained_model(
        eval_result_path = mbv_config.evaluation_path_supervised,
        fold_root=mbv_config.data_path_fold0, 
        dataset_portion=mbv_config.test, 
        model_name=mbv_config.DenseNet_121, 
        pretrained_weights_file_path='/home/prachh/mbv/src/results/MBV_<class \'models.Densenet_Model\'>_V3_224_224_1e-05_7_Classes_Combined_Images_Fold0/_82_0.49678096175193787.pth'
    )

    