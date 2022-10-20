# Depth-Contrast-Self-Supervised-Method
Source code for [Depth Contrast: Self-Supervised Pretraining on 3DPM Images for Mining Material Classification](https://arxiv.org/abs/2210.10633) - ***Accpeted in ECCV Workshop 2022, Tel Aviv***




# Commands
Code also includes supervised finutung and evaluation.
**Self-supervised pretraining (Assuming in directory 'src') - check all paramters in py file - all default paramters are set** 

```python -m self_supervised.experiments.main_pretrain_depth_contrast --data_path <'train_data_fold_path of 3DPM dataset'> --LR <learning_rate - 0.00005> --epochs <300> --description <'experiment_name'>```


**Fintuning/linear evaluation training for Efficient-net b2 on 3DPM dataset wit hdifferent data portion and pretrained model weights (Assuming in directory 'src')**

```python -m experiments.train_efficientnet --data_path <'data_fold_path of 3DPM dataset - includes train and validation part both'> --LR <learning_rate - 0.00001> --epochs <100> --description <'experiment_name'>  --pretrained_model_path <pretrained moodel path for depth contrast trained model> --data_portion <data portion - 20 or 60> --LE <True| False - give False for full fintnuing>```

**Fintuning using MPCS pretrained Efficient-net b2 on BreakHis (Assuming in directory 'src')**

```python -m experiments.train_common --data_path <'data_fold_path of 3DPM dataset - includes train and validation part both'> --LR <learning_rate - 0.00001> --epochs <100> --description <'experiment_name'>  --architecture <resnext | densenet>```

**Evaluation**

```python - m test.test_and_save_results --data_path <'data_fold_path of 3DPM dataset - includes train and validation part both'> --architecture <resnext | densenet | efficient> --split <train | test| val> --model_file_path <path for trained model's weights>```
