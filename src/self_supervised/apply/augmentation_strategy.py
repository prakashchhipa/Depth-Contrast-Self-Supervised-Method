from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from self_supervised.apply import config

custom_center_crop_augmentation = transforms.Compose([
    transforms.CenterCrop((config.image_size,config.image_size))
])

custom_224_random_crop_augmentation = transforms.Compose([
    transforms.RandomCrop((config.image_size,config.image_size))
])

custom_random_crop_augmentation = transforms.Compose([
    transforms.RandomCrop((config.image_size_big,config.image_size_big))
])

A_random_crop_augmentation = A.Compose([
    A.RandomCrop(config.image_size,config.image_size, always_apply=True),
    ToTensorV2()
])

A512_random_crop_augmentation = A.Compose([
    A.RandomCrop(config.image_size_big,config.image_size_big, always_apply=True),
    ToTensorV2()
])


A224_only_random_crop_augmentation = A.Compose([
    A.RandomCrop(config.image_size,config.image_size, always_apply=True),
])