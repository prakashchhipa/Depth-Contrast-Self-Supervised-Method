import numpy as np, json, argparse, time
from tqdm import tqdm
import cv2, logging
from pathlib import Path
import torch, torchvision
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from sklearn.metrics import f1_score, matthews_corrcoef, classification_report, confusion_matrix, accuracy_score
from sampler import BalancedBatchSampler
from dataloader import MBV_Dataset
from utils import *
from mbv_config import MBV_Config
import mbv_config

class Train_Util:

    def __init__(self, experiment_description, image_type, epochs, model, device, train_loader, val_loader, optimizer, criterion, batch_size, scheduler, writer):
        self.experiment_description = experiment_description
        self.image_type = image_type
        self.epochs = epochs
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.batch_size = batch_size
        self.scheduler = scheduler
        self.writer = writer
        #
        

    def train_epoch(self):
        self.model.train()
        loss_agg = Aggregator()
        confusion_matrix_epoch = torch.zeros(len(mbv_config.label_list), len(mbv_config.label_list))
        mcc_sum = 0
        with tqdm(total=(len(self.train_loader))) as (t):
            for raw_image, ref_image, label in tqdm(self.train_loader):
                raw_image = np.asarray(raw_image)
                ref_image = np.asarray(ref_image)

                    
                #print(raw_image.shape, ref_image.shape)
                if self.image_type == mbv_config.image_both:
                    image = cv2.merge((raw_image, ref_image, raw_image))
                else:
                    if self.image_type == mbv_config.image_raw:
                        image = cv2.merge((raw_image, raw_image, raw_image))
                    else:
                        if self.image_type == mbv_config.image_ref:
                            image = cv2.merge((ref_image, ref_image, ref_image))
                
                #print(image.shape)

                inputs = torch.from_numpy(image).to(self.device)
                inputs = inputs.permute(0, 1, 4, 2, 3)
                

                inputs = inputs.squeeze(1)
                #print('tensor shape', inputs.shape)

                '''for img in inputs:
                    Img = self.transform(image = np.array(img))
'''

                target = label.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                #print('target', target)
                #print('predicted', predicted)
                predicted = predicted.to(self.device)
                for targetx, predictedx in zip(target.view(-1), predicted.view(-1)):
                    confusion_matrix_epoch[(targetx.long(), predictedx.long())] += 1
                else:
                    mcc_sum += matthews_corrcoef(target.cpu(), predicted.cpu())
                    loss = self.criterion(outputs, target)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    loss_agg.update(loss.item())
                    t.set_postfix(loss=('{:05.3f}'.format(loss_agg())))
                    t.update()

        weighted_f1, accuracy, classwise_precision, classwise_recall, classwise_f1 = self.get_metrics_from_confusion_matrix(confusion_matrix_epoch)
        mcc = mcc_sum / len(self.train_loader)
        print(f'{self.experiment_description}:classwise precision', classwise_precision)
        print(f'{self.experiment_description}: classwise recall', classwise_recall)
        print(f'{self.experiment_description}: classwise f1', classwise_f1)
        print(f'{self.experiment_description}: MCC', mcc)
        print(f'{self.experiment_description}: Weighted F1', weighted_f1)
        print(f'{self.experiment_description}: Accuracy', accuracy)
        return (
         mcc, weighted_f1, accuracy, classwise_precision, classwise_recall, classwise_f1, loss_agg())
    

    def train_epoch_32bit(self):
        self.model.train()
        loss_agg = Aggregator()
        confusion_matrix_epoch = torch.zeros(len(mbv_config.label_list), len(mbv_config.label_list))
        mcc_sum = 0
        with tqdm(total=(len(self.train_loader))) as (t):
            for image, label in tqdm(self.train_loader):
                
                print('image shape', image.shape)
                
                inputs = torch.from_numpy(image).to(self.device)
                inputs = inputs.permute(0, 3, 1, 2)
                
                print('tensor shape', inputs.shape)

                
                target = label.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                predicted = predicted.to(self.device)
                for targetx, predictedx in zip(target.view(-1), predicted.view(-1)):
                    confusion_matrix_epoch[(targetx.long(), predictedx.long())] += 1
                else:
                    mcc_sum += matthews_corrcoef(target.cpu(), predicted.cpu())
                    loss = self.criterion(outputs, target)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    loss_agg.update(loss.item())
                    t.set_postfix(loss=('{:05.3f}'.format(loss_agg())))
                    t.update()

        weighted_f1, accuracy, classwise_precision, classwise_recall, classwise_f1 = self.get_metrics_from_confusion_matrix(confusion_matrix_epoch)
        mcc = mcc_sum / len(self.train_loader)
        print(f'{self.experiment_description}:classwise precision', classwise_precision)
        print(f'{self.experiment_description}: classwise recall', classwise_recall)
        print(f'{self.experiment_description}: classwise f1', classwise_f1)
        print(f'{self.experiment_description}: MCC', mcc)
        print(f'{self.experiment_description}: Weighted F1', weighted_f1)
        print(f'{self.experiment_description}: Accuracy', accuracy)
        return (
         mcc, weighted_f1, accuracy, classwise_precision, classwise_recall, classwise_f1, loss_agg())


    def evaluate_validation_set(self):
        confusion_matrix_val = torch.zeros(len(mbv_config.label_list), len(mbv_config.label_list))
        mcc_sum = 0
        self.model.eval()
        val_loss_avg = Aggregator()
        with torch.no_grad():
            for raw_image, ref_image, label in tqdm(self.val_loader):
                raw_image = np.asarray(raw_image)
                ref_image = np.asarray(ref_image)
                if self.image_type == mbv_config.image_both:
                    image = cv2.merge((raw_image, ref_image, raw_image))
                else:
                    if self.image_type == mbv_config.image_raw:
                        image = cv2.merge((raw_image, raw_image, raw_image))
                    else:
                        if self.image_type == mbv_config.image_ref:
                            image = cv2.merge((ref_image, ref_image, ref_image))
                inputs = torch.from_numpy(image).to(self.device)
                inputs = inputs.permute(0, 1, 4, 2, 3)
                inputs = inputs.squeeze(1)
                target = label.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                predicted = predicted.to(self.device)
                for targetx, predictedx in zip(target.view(-1), predicted.view(-1)):
                    confusion_matrix_val[(targetx.long(), predictedx.long())] += 1
                else:
                    mcc_sum += matthews_corrcoef(target.cpu(), predicted.cpu())
                    loss = self.criterion(outputs, target)
                    val_loss_avg.update(loss.item())

        weighted_f1, accuracy, classwise_precision, classwise_recall, classwise_f1 = self.get_metrics_from_confusion_matrix(confusion_matrix_val)
        mcc = mcc_sum / len(self.val_loader)
        
        print(f'{self.experiment_description}: Validation classwise precision', classwise_precision)
        print(f'{self.experiment_description}: Validation classwise recall', classwise_recall)
        print(f'{self.experiment_description}: Validation classwise f1', classwise_f1)
        print(f'{self.experiment_description}: Validation MCC', mcc)
        print(f'{self.experiment_description}: Validation Weighted F1', weighted_f1)
        print(f'{self.experiment_description}: Validation Accuracy', accuracy)
        return (
         mcc, weighted_f1, accuracy, classwise_precision, classwise_recall, classwise_f1, val_loss_avg())

    def test_class_probabilities(self, model, device, test_loader, which_class):
        model.eval()
        actuals = []
        probabilities = []
        with torch.no_grad():
            for image, label in test_loader:
                image, label = image.to(device), label.to(device)
                output = torch.sigmoid(model(image))
                prediction = output.argmax(dim=1, keepdim=True)
                actuals.extend(label.view_as(prediction) == which_class)
                output = output.cpu()
                probabilities.extend(np.exp(output[:, which_class]))

        return (
         [i.item() for i in actuals], [i.item() for i in probabilities])

    def train_and_evaluate(self):
        
        best_f1 = 0.0
        for epoch in range(self.epochs):
            #train epoch
            mcc, weighted_f1, accuracy,classwise_precision,classwise_recall,classwise_f1, loss = self.train_epoch()
            #evaluate on validation set
            val_mcc, val_weighted_f1, val_accuracy, val_classwise_precision,val_classwise_recall,val_classwise_f1, val_loss = self.evaluate_validation_set()
                        
            print("Epoch {}/{} Train Loss:{}, Val Loss: {}".format(epoch, self.epochs, loss, val_loss))

            if best_f1 < val_weighted_f1:
                best_f1 = val_weighted_f1
                result_path = f"/home/a_shared_data/MBV_data/results/{self.experiment_description}"
                Path(result_path).mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), f"{result_path}/_{epoch}_{val_weighted_f1}.pth")
            
            self.scheduler.step(val_loss)

            #Tensorboard
            self.writer.add_scalar('Loss/Validation_Set', val_loss, epoch)
            self.writer.add_scalar('Loss/Training_Set', loss, epoch)

            self.writer.add_scalar('Accuracy/Validation_Set', val_accuracy, epoch)
            self.writer.add_scalar('Accuracy/Training_Set', accuracy, epoch)
            
            self.writer.add_scalar('Weighted F1/Validation_Set', val_weighted_f1, epoch)
            self.writer.add_scalar('Weighted F1/Training_Set', weighted_f1, epoch)

            self.writer.add_scalar('Matthew Correlation Cofficient/Validation_Set', val_mcc, epoch)
            self.writer.add_scalar('Matthew Correlation Cofficient/Training_Set', mcc, epoch)
            
            self.writer.add_scalar('Learning Rate', self.optimizer.param_groups[0]['lr'], epoch)

            #Classwise metrics logging
            for index in range(0,len(mbv_config.label_list)):
                
                self.writer.add_scalar(f'F1/Validation_Set/{mbv_config.label_list[index]}', val_classwise_f1[index], epoch)
                self.writer.add_scalar(f'F1/Training_Set/{mbv_config.label_list[index]}', classwise_f1[index], epoch)

                self.writer.add_scalar(f'Precision/Validation_Set/{mbv_config.label_list[index]}', val_classwise_precision[index], epoch)
                self.writer.add_scalar(f'Precision/Training_Set/{mbv_config.label_list[index]}', classwise_precision[index], epoch)

                self.writer.add_scalar(f'Recall/Validation_Set/{mbv_config.label_list[index]}', val_classwise_recall[index], epoch)
                self.writer.add_scalar(f'Recall/Training_Set/{mbv_config.label_list[index]}', classwise_recall[index], epoch)

                
    def process_classification_report(self, report):
        report_data = []
        lines = report.split('\n')
        for line in lines[2:-3]:
            row = {}
            row_data = line.split(' ')
            row_data = list(filter(None, row_data))
            row['class'] = row_data[0]
            row['precision'] = float(row_data[1])
            row['recall'] = float(row_data[2])
            row['f1_score'] = float(row_data[3])
            report_data.append(row)
        else:
            return report_data

    def get_metrics_from_confusion_matrix(self, confusion_matrix_epoch):
        epoch_classwise_precision_manual_cpu = np.array(confusion_matrix_epoch.diag().cpu()) / np.array(confusion_matrix_epoch.cpu()).sum(axis=0)
        epoch_classwise_precision_manual_cpu = np.nan_to_num(epoch_classwise_precision_manual_cpu, nan=0, neginf=0, posinf=0)
        epoch_classwise_recall_manual_cpu = np.array(confusion_matrix_epoch.diag().cpu()) / np.array(confusion_matrix_epoch.cpu()).sum(axis=1)
        epoch_classwise_recall_manual_cpu = np.nan_to_num(epoch_classwise_recall_manual_cpu, nan=0, neginf=0, posinf=0)
        epoch_classwise_f1_manual_cpu = 2 * (epoch_classwise_precision_manual_cpu * epoch_classwise_recall_manual_cpu) / (epoch_classwise_precision_manual_cpu + epoch_classwise_recall_manual_cpu)
        epoch_classwise_f1_manual_cpu = np.nan_to_num(epoch_classwise_f1_manual_cpu, nan=0, neginf=0, posinf=0)
        epoch_avg_f1_manual = np.sum(epoch_classwise_f1_manual_cpu * np.array(confusion_matrix_epoch.cpu()).sum(axis=1)) / np.array(confusion_matrix_epoch.cpu()).sum(axis=1).sum()
        epoch_acc_manual = 100 * np.sum(np.array(confusion_matrix_epoch.diag().cpu())) / np.sum(np.array(confusion_matrix_epoch.cpu()))
        return (
         epoch_avg_f1_manual, epoch_acc_manual, epoch_classwise_precision_manual_cpu, epoch_classwise_recall_manual_cpu, epoch_classwise_f1_manual_cpu)
