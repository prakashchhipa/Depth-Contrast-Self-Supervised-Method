from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

from self_supervised.apply import config

# Dataset input processing - trainset
raw_train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
        ])
ref_train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
        ])

A_common_train_transform = transforms.Compose([
        ToTensorV2(),
        A.Normalize(mean=[0.485], std=[0.229])
        ])