# Depth-Contrast-Self-Supervised-Method
Source code for [Depth Contrast: Self-Supervised Pretraining on 3DPM Images for Mining Material Classification](https://arxiv.org/abs/2210.10633) - ***Accpeted in ECCV Workshop 2022, Tel Aviv***

Chhipa, P. C., Upadhyay, R., Saini, R., Lindqvist, L., Nordenskjold, R., Uchida, S., & Liwicki, M. (2023, February). Depth Contrast: Self-supervised Pretraining on 3DPM Images for Mining Material Classification. In Computer Vision–ECCV 2022 Workshops: Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part VII (pp. 212-227). Cham: Springer Nature Switzerland.

# Abstract
This work presents a novel self-supervised representation learning method to learn efficient representations without labels on images from a 3DPM sensor (3-Dimensional Particle Measurement; estimates the particle size distribution of material) utilizing RGB images and depth maps of mining material on the conveyor belt. Human annotations for material categories on sensor-generated data are scarce and cost-intensive. Currently, representation learning without human annotations remains unexplored for mining materials and does not leverage on utilization of sensor-generated data. The proposed method, Depth Contrast, enables self-supervised learning of representations without labels on the 3DPM dataset by exploiting depth maps and inductive transfer. The proposed method outperforms material classification over ImageNet transfer learning performance in fully supervised learning settings and achieves an F1 score of 0.73. Further, The proposed method yields an F1 score of 0.65 with an 11% improvement over ImageNet transfer learning performance in a semi-supervised setting when only 20% of labels are used in fine-tuning. Finally, the Proposed method showcases improved performance generalization on linear evaluation.

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
