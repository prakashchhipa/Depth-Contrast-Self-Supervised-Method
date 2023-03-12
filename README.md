# Title

[Depth Contrast: Self-Supervised Pretraining on 3DPM Images for Mining Material Classification](https://arxiv.org/abs/2210.10633)

# Venue

Accpeted in ECCV Workshop 2022, Tel Aviv.

Chhipa, P. C., Upadhyay, R., Saini, R., Lindqvist, L., Nordenskjold, R., Uchida, S., & Liwicki, M. (2023, February). Depth Contrast: Self-supervised Pretraining on 3DPM Images for Mining Material Classification. In Computer Vision–ECCV 2022 Workshops: Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part VII (pp. 212-227). Cham: Springer Nature Switzerland.

# Abstract
This work presents a novel self-supervised representation learning method to learn efficient representations without labels on images from a 3DPM sensor (3-Dimensional Particle Measurement; estimates the particle size distribution of material) utilizing RGB images and depth maps of mining material on the conveyor belt. Human annotations for material categories on sensor-generated data are scarce and cost-intensive. Currently, representation learning without human annotations remains unexplored for mining materials and does not leverage on utilization of sensor-generated data. The proposed method, Depth Contrast, enables self-supervised learning of representations without labels on the 3DPM dataset by exploiting depth maps and inductive transfer. The proposed method outperforms material classification over ImageNet transfer learning performance in fully supervised learning settings and achieves an F1 score of 0.73. Further, The proposed method yields an F1 score of 0.65 with an 11% improvement over ImageNet transfer learning performance in a semi-supervised setting when only 20% of labels are used in fine-tuning. Finally, the Proposed method showcases improved performance generalization on linear evaluation.

# Method

Depth Contrast Method with downstream task - complete pipeline

<p align="center">
  <img src="https://github.com/prakashchhipa/Depth-Contrast-Self-Supervised-Method/blob/main/figures/method.PNG">
</p>

# Dataset

3DPM Dataset -  3-Dimensional Particle Measurement (3DPM®) is a system that estimates the particle size distribution of material passing on a conveyor belt. The 3DPM data can be visualized as images but are different from RGB/gray images. The 3DPM dataset has seven classes corresponding to the categories of mining material available on the conveyor belt. These classes are Cylindrical, Ore1, Ore2, Ore3, Mixed1, Mixed2, and Agglomerated (Aggl.). Two modalities of mining material data that are acquired using the 3DPM sensor. First are the reflectance images, which are 2D grayscale images of the material on the conveyor belt taken in red laser light. The second modality is the 3D depth map of the bulk material, i.e., the distance between the material and the sensor. Total 3008 reflectance images wit htheir coresponding depth maps are used. Material classes are shown below.

<p align="center">
  <img src="https://github.com/prakashchhipa/Depth-Contrast-Self-Supervised-Method/blob/main/figures/dataset.PNG">
</p>

# Results

Finetuned for classification tasks
<p align="center">
  <img src="https://github.com/prakashchhipa/Depth-Contrast-Self-Supervised-Method/blob/main/figures/results_ft.PNG">
</p>

Linear Evaluation for classification tasks
<p align="center">
  <img src="https://github.com/prakashchhipa/Depth-Contrast-Self-Supervised-Method/blob/main/figures/results_ft.PNG">
</p>

Performance Improvement
<p align="center">
  <img src="https://github.com/prakashchhipa/Depth-Contrast-Self-Supervised-Method/blob/main/figures/results.PNG">
</p>


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
