import torchvision.models as models
import torch.nn as nn
import torch
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet


class EfficientNet_Model(torch.nn.Module):
    def __init__(self, pretrained=True):
        super(EfficientNet_Model, self).__init__()
        num_classes = 7
        self.model = EfficientNet.from_pretrained("efficientnet-b2")
        num_ftrs=self.model._fc.in_features
        self.model._fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(num_ftrs, num_classes))
        #self.model.fc=nn.Linear(512,num_classes)
    def forward(self, x):
        output = self.model(x)
        return output



class Resnext_Model(torch.nn.Module):
    def __init__(self, pretrained=True):
        super(Resnext_Model, self).__init__()
        #num_classes = 10
        #new setting
        num_classes = 7
        self.model = models.resnext50_32x4d(pretrained=True)
        #self.model.conv1=nn.Conv2d(2,64,kernel_size=(3,3),stride=(2,2),padding=(3,3),bias=False)
        #self.model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
        num_ftrs=self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, num_classes))
        #self.model.fc=nn.Linear(num_ftrs,512)
        #self.model.fc=nn.Linear(512,num_classes)

    def forward(self, x):
        output = self.model(x)
        return output


class Densenet_Model(torch.nn.Module):
    def __init__(self, pretrained=True):
        super(Densenet_Model, self).__init__()
        #num_classes = 10
        #new setting
        num_classes = 7
        self.model = models.densenet121(pretrained=True)
        #self.model.conv1=nn.Conv2d(2,64,kernel_size=(3,3),stride=(2,2),padding=(3,3),bias=False)
        #self.model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
        #num_ftrs=self.model.fc.in_features
        #self.model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(num_ftrs, num_classes))
        self.model.classifier = nn.Linear(1024, num_classes)
        #self.model.fc=nn.Linear(num_ftrs,512)
        #self.model.fc=nn.Linear(512,num_classes)

    def forward(self, x):
        output = self.model(x)
        return output