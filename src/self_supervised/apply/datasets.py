import os
import torch
import cv2
import torch.nn as nn
import torchvision.datasets as datasets
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
from glob import glob
from torch.utils.data import DataLoader, Dataset
import glob
import numpy as np
from skimage import io
from torch.utils.data import DataLoader, Dataset
import json
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import PIL
from PIL import Image
from torchvision.transforms import transforms
from torchvision.transforms import ToTensor, Resize

from self_supervised.apply import config

import mbv_config


class MBV_Dataset_SSL(nn.Module):

    def __init__(self, raw_train_file_path, reflectance_train_file_path, training_method=None, transform = None, augmentation_strategy = None, image_type = None):
        
        self.raw_image_path_list = []
        self.reflectance_image_path_list = []
        self.transform = transform
        
        self.label_dict = mbv_config.label_dict
        

        #SSL specific
        self.augmentation_strategy_1 = augmentation_strategy
        self.augmentation_strategy_2 = augmentation_strategy
        self.training_method = training_method
        self.image_type = image_type


        
        
        self.raw_image_path_list = np.load (raw_train_file_path)
        self.reflectance_image_path_list = np.load(reflectance_train_file_path)

                        
    def __len__(self):
        return len(self.raw_image_path_list)

    def __getitem__(self, index):

        raw_image_path = self.raw_image_path_list[index]
        raw_image  = PIL.Image.open(raw_image_path)


        reflectance_image_path = self.reflectance_image_path_list[index]
        reflectance_image  = PIL.Image.open(reflectance_image_path)

        if self.training_method == config.dc:
            
            if self.image_type == mbv_config.image_seprately:
                raw_image = np.asarray(raw_image)
                reflectance_image = np.asarray(reflectance_image)
                raw_ref_ref_image = cv2.merge((raw_image, reflectance_image, reflectance_image))
                view1_view2_view2 = self.augmentation_strategy_1(image = raw_ref_ref_image)

                transformed_view1, transformed_view2 = None, None
                if self.transform != None:
                    transformed_view1 = self.transform[0](image = view1_view2_view2)
                    #transformed_view2 = self.transform[1](image = view1_view2_view2[:,:,1])
                transformed_view1_3ch = np.vstack([transformed_view1, transformed_view1, transformed_view1])
                transformed_view2_3ch = np.vstack([transformed_view2, transformed_view2, transformed_view2])

                return transformed_view1_3ch, transformed_view2_3ch
            
            if self.image_type == mbv_config.image_both_seprately:
                '''using state for uniform random cropp on both ref and raw image'''
                state = torch.get_rng_state()
                view1 = self.augmentation_strategy_1(raw_image)
                torch.set_rng_state(state)
                view2 = self.augmentation_strategy_1(reflectance_image)
                transformed_view1, transformed_view2 = None, None
                if self.transform != None:
                    transformed_view1 = self.transform[0](view1)
                    transformed_view2 = self.transform[1](view2)
                transformed_view1_3ch = np.vstack([transformed_view1, transformed_view1, transformed_view1])
                transformed_view2_3ch = np.vstack([transformed_view2, transformed_view2, transformed_view2])

                return transformed_view1_3ch, transformed_view2_3ch
            
            elif self.image_type == mbv_config.image_both:
                transformed_view1, transformed_view2 = None, None
                if self.transform != None:
                    transformed_view1 = self.transform[0](raw_image)
                    transformed_view2 = self.transform[1](reflectance_image)
                
                raw_image = np.asarray(raw_image)
                reflectance_image = np.asarray(reflectance_image)
                combined_image = cv2.merge((raw_image, reflectance_image, raw_image))
                #combined_image = Image.fromarray(combined_image).astype('float32')
                view1 = self.augmentation_strategy_1(image = combined_image)
                view2 = self.augmentation_strategy_2(image = combined_image)
                return view1, view2
            elif self.image_type == mbv_config.image_raw:
                view1 = self.augmentation_strategy_1(raw_image)
                view2 = self.augmentation_strategy_2(raw_image)
                transformed_view1, transformed_view2 = None, None
                if self.transform != None:
                    transformed_view1 = self.transform[0](view1)
                    transformed_view2 = self.transform[0](view2)
                return transformed_view1, transformed_view2
            elif self.image_type == mbv_config.image_ref:
                view1 = self.augmentation_strategy_1(reflectance_image)
                view2 = self.augmentation_strategy_2(reflectance_image)
                transformed_view1, transformed_view2 = None, None
                if self.transform != None:
                    transformed_view1 = self.transform[0](view1)
                    transformed_view2 = self.transform[0](view2)
                return transformed_view1, transformed_view2

        elif self.training_method == config.BYOL:
            if self.image_type == mbv_config.image_both:
                transformed_view1, transformed_view2 = None, None
                if self.transform != None:
                    transformed_view1 = self.transform[0](raw_image)
                    transformed_view2 = self.transform[1](reflectance_image)
                
                combined_image = cv2.merge((raw_image, reflectance_image, raw_image))
                view1 = self.augmentation_strategy_1(combined_image)
                return view1
            elif self.image_type == mbv_config.image_raw:
                view1 = self.augmentation_strategy_1(raw_image)
                transformed_view1  = None
                if self.transform != None:
                    transformed_view1 = self.transform[0](view1)
                return transformed_view1
            elif self.image_type == mbv_config.image_ref:
                view1 = self.augmentation_strategy_1(reflectance_image)
                transformed_view1 = None
                if self.transform != None:
                    transformed_view1 = self.transform[0](view1)
                return transformed_view1


def get_MBV_trainset_loader(raw_train_file_path, reflectance_train_file_path, batch_size, training_method=None, transform = None, augmentation_strategy = None, image_type = None):

    dataset = MBV_Dataset_SSL(raw_train_file_path, reflectance_train_file_path, training_method, transform, augmentation_strategy, image_type)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    return train_loader