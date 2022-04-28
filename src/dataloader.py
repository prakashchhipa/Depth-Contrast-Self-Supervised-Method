import torch.nn as nn
import os
import pandas as pd
from PIL import Image
import PIL
import csv
#import albumentations as A
import numpy as np
import os.path
import torch
from mbv_config import MBV_Config
import mbv_config

import cv2

class MBV_Dataset(nn.Module):

    def __init__(self, raw_train_file_path, reflectance_train_file_path, label_file_path, transform=None):
        
        self.raw_image_path_list = []
        self.reflectance_image_path_list = []
        self.transform = transform
        self.labels = []
        self.label_dict = mbv_config.label_dict
        self.class_count_list = [0,0,0,0,0,0,0]
        
        
        self.raw_image_path_list = np.load (raw_train_file_path)
        self.reflectance_image_path_list = np.load(reflectance_train_file_path)
        self.labels = np.load(label_file_path)
        

             
                        
    def __len__(self):
        return len(self.raw_image_path_list)

    def __getitem__(self, index):

        raw_image_path = self.raw_image_path_list[index]
        raw_image  = PIL.Image.open(raw_image_path)
        #raw_image = cv2.imread(raw_image_path, cv2.IMREAD_UNCHANGED)
        #raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

        reflectance_image_path = self.reflectance_image_path_list[index]
        reflectance_image  = PIL.Image.open(reflectance_image_path)

        label =  self.label_dict[self.labels[index]]
        

        if None != self.transform:
            if len(self.transform) == 1:
                raw_image = self.transform[0](raw_image)
                reflectance_image = self.transform[0](reflectance_image)
            elif len(self.transform) >= 2:

                reflectance_image = self.transform[1](reflectance_image)
        
        return (raw_image, reflectance_image, label)


class MBV_Dataset_SSL_Valset_from_Valset(nn.Module):

    def __init__(self, raw_train_file_path, reflectance_train_file_path, label_file_path, transform=None):
        
        self.raw_image_path_list = []
        self.reflectance_image_path_list = []
        self.transform = transform
        self.labels = []
        self.label_dict = mbv_config.label_dict
        self.class_count_list = [0,0,0,0,0,0,0]
        
        
        self.raw_image_path_list = np.load (raw_train_file_path)
        self.reflectance_image_path_list = np.load(reflectance_train_file_path)
        self.labels = np.load(label_file_path)

        self.raw_image_path_list_val = []
        self.reflectance_image_path_list_val = []
        self.labels_val = []
        for idx in range(0, len(self.raw_image_path_list)):
            if(idx%2 == 0):
                self.raw_image_path_list_val.append(self.raw_image_path_list[idx])
                self.reflectance_image_path_list_val.append(self.reflectance_image_path_list[idx])
                self.labels_val.append(self.labels[idx])

                        
    def __len__(self):
        return len(self.raw_image_path_list_val)

    def __getitem__(self, index):

        raw_image_path = self.raw_image_path_list_val[index]
        raw_image  = PIL.Image.open(raw_image_path)
        #raw_image = cv2.imread(raw_image_path, cv2.IMREAD_UNCHANGED)
        #raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

        reflectance_image_path = self.reflectance_image_path_list_val[index]
        reflectance_image  = PIL.Image.open(reflectance_image_path)

        label =  self.label_dict[self.labels_val[index]]
        

        if None != self.transform:
            if len(self.transform) == 1:
                raw_image = self.transform[0](raw_image)
                reflectance_image = self.transform[0](reflectance_image)
            elif len(self.transform) >= 2:
                reflectance_image = self.transform[1](reflectance_image)
        
        return (raw_image, reflectance_image, label)

class MBV_Dataset_SSL_Trainset_from_Valset(nn.Module):

    def __init__(self, raw_train_file_path, reflectance_train_file_path, label_file_path, transform=None):
        
        self.raw_image_path_list = []
        self.reflectance_image_path_list = []
        self.transform = transform
        self.labels = []
        self.label_dict = mbv_config.label_dict
        self.class_count_list = [0,0,0,0,0,0,0]
        
        
        self.raw_image_path_list = np.load (raw_train_file_path)
        self.reflectance_image_path_list = np.load(reflectance_train_file_path)
        self.labels = np.load(label_file_path)

        self.raw_image_path_list_val = []
        self.reflectance_image_path_list_val = []
        self.labels_val = []
        for idx in range(0, len(self.raw_image_path_list)):
            if(idx%2 != 0):
                self.raw_image_path_list_val.append(self.raw_image_path_list[idx])
                self.reflectance_image_path_list_val.append(self.reflectance_image_path_list[idx])
                self.labels_val.append(self.labels[idx])

                        
    def __len__(self):
        return len(self.raw_image_path_list_val)

    def __getitem__(self, index):

        raw_image_path = self.raw_image_path_list_val[index]
        raw_image  = PIL.Image.open(raw_image_path)


        reflectance_image_path = self.reflectance_image_path_list_val[index]
        reflectance_image  = PIL.Image.open(reflectance_image_path)

        label =  self.label_dict[self.labels_val[index]]
        

        if None != self.transform:
            if len(self.transform) == 1:
                raw_image = self.transform[0](raw_image)
                reflectance_image = self.transform[0](reflectance_image)
            elif len(self.transform) >= 2:
                reflectance_image = self.transform[1](reflectance_image)
        
        return (raw_image, reflectance_image, label)


class MBV_Dataset_8bit(nn.Module):

    def __init__(self, raw_train_file_path, reflectance_train_file_path, label_file_path, transform=None):
        
        self.raw_image_path_list = []
        self.reflectance_image_path_list = []
        self.transform = transform
        self.labels = []
        self.label_dict = mbv_config.label_dict
        self.class_count_list = [0,0,0,0,0,0,0]
        
        
        self.raw_image_path_list = np.load (raw_train_file_path)
        self.reflectance_image_path_list = np.load(reflectance_train_file_path)
        self.labels = np.load(label_file_path)
             
                        
    def __len__(self):
        return len(self.raw_image_path_list)

    def __getitem__(self, index):

        raw_image_path = self.raw_image_path_list[index]
        raw_image  = PIL.Image.open(raw_image_path)
        #Making it 8 bit
        raw_image = raw_image.convert('L')
       
        reflectance_image_path = self.reflectance_image_path_list[index]
        reflectance_image  = PIL.Image.open(reflectance_image_path)
        #Making it 8 bit
        raw_image = raw_image.convert('L')

        label =  self.label_dict[self.labels[index]]
       

        if None != self.transform:
            if len(self.transform) == 1:
                raw_image = self.transform[0](raw_image)
                reflectance_image = self.transform[0](reflectance_image)
            elif len(self.transform) >= 2:
                raw_image = self.transform[1](raw_image) # = np.array(raw_image))
                reflectance_image = self.transform[1](reflectance_image)
        
        return (raw_image, reflectance_image, label)




class MBV_Dataset_32bit(nn.Module):

    def __init__(self, raw_train_file_path, reflectance_train_file_path, label_file_path, transform=None):
        
        self.raw_image_path_list = []
        self.reflectance_image_path_list = []
        self.transform = transform
        self.labels = []
        self.label_dict = mbv_config.label_dict
        self.class_count_list = [0,0,0,0,0,0,0]
        
        
        self.raw_image_path_list = np.load (raw_train_file_path)
        self.reflectance_image_path_list = np.load(reflectance_train_file_path)
        self.labels = np.load(label_file_path)
        
       
             
                        
    def __len__(self):
        return len(self.raw_image_path_list)

    def __getitem__(self, index):

        raw_image_path = self.raw_image_path_list[index]
        raw_image  = PIL.Image.open(raw_image_path)
        #raw_image = cv2.imread(raw_image_path, cv2.IMREAD_UNCHANGED)
        #raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

        reflectance_image_path = self.reflectance_image_path_list[index]
        reflectance_image  = PIL.Image.open(reflectance_image_path)

        label =  self.label_dict[self.labels[index]]
        
        raw_image = np.asarray(raw_image)
        reflectance_image = np.asarray(reflectance_image)

        image = cv2.merge((raw_image, reflectance_image, raw_image))
        
        image = torch.from_numpy(image.astype(np.float32))
        print(image.dtype)

        image = np.array(image)
        image = self.transform[0](image= image)

        
        
        return (image, label)
