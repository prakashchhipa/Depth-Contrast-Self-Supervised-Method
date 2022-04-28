import torch

label_dict = {'Mixed1': 0, 'Aggl': 1, 'Mixed2': 2, 'Ore1': 3, 'Ore2': 4, 'Ore3': 5, 'Cylindrical': 6}
label_list = ['Mixed1', 'Aggl', 'Mixed2', 'Ore1', 'Ore2', 'Ore3', 'Cylindrical']

data_path_fold0 = '/home/a_shared_data/MBV_data/LTUdata/Fold_0/'
data_path_fold1 = '/home/a_shared_data/MBV_data/LTUdata/Fold_1/'
data_path_fold2 = '/home/a_shared_data/MBV_data/LTUdata/Fold_2/'
data_path_fold3 = '/home/a_shared_data/MBV_data/LTUdata/Fold_3/'
data_path_fold4 = '/home/a_shared_data/MBV_data/LTUdata/Fold_4/'

tensorboard_path = '/home/a_shared_data/MBV_data/tensorboard/'
result_path = '/home/a_shared_data/MBV_data/results/'
evaluation_path_supervised = '/home/vscode/evaluation/supervised_finetuned_80/'
evaluation_path_supervised_ref_images = '/home/vscode/evaluation/supervised_finetuned_80_ref_images/'
evaluation_path_supervised_raw_images = '/home/vscode/evaluation/supervised_finetuned_80_raw_images/'
evaluation_path_self_supervised = '/home/vscode/evaluation/self_supervised/'
evaluation_path_self_supervised_80 = '/home/vscode/evaluation/self_supervised_pretrained_finetuned_80/'
evaluation_path_transfer_learning = '/home/vscode/evaluation/supervised_finetuned_20/'

#GPU
gpu0 = torch.device("cuda:0")
gpu1 = torch.device("cuda:1")
gpu2 = torch.device("cuda:2")
gpu3 = torch.device("cuda:3")
gpu4 = torch.device("cuda:4")
gpu5 = torch.device("cuda:5")
gpu6 = torch.device("cuda:6")
gpu7 = torch.device("cuda:7")


#Data Input
image_both = "b"
image_raw = "r"
image_ref = "ref"
image_both_seprately = "ssl_"
image_seprately = "ssl_im"


#dataset portion
train = 'train'
test = 'test'
val = 'val'

#networks
EfficientNet_b2 = 'EfficientNet_b2'
DenseNet_121 = 'DenseNet_121'
ResNext_50_32x4d = 'ResNext_50_32x4d' 

#Model params
num_classes = 7



class MBV_Config():
	def __init__(self):
		
		self.label_dict = {'Mixed1': 0, 'Aggl': 1, 'Mixed2': 2, 'Ore1': 3, 'Ore2': 4, 'Ore3': 5, 'Cylindrical': 6}
		self.label_list = ['Mixed1', 'Aggl', 'Mixed2', 'Ore1', 'Ore2', 'Ore3', 'Cylindrical']
		
		self.data_path_fold0 = '/home/a_shared_data/MBV_data/LTUdata/Fold_0/'
		self.data_path_fold1 = '/home/a_shared_data/MBV_data/LTUdata/Fold_1/'
		self.data_path_fold2 = '/home/a_shared_data/MBV_data/LTUdata/Fold_2/'
		self.data_path_fold3 = '/home/a_shared_data/MBV_data/LTUdata/Fold_3/'
		self.data_path_fold4 = '/home/a_shared_data/MBV_data/LTUdata/Fold_4/'