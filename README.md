# ALMRR: Anomaly Localization Mamba with Feature Reconstruction and refinement
## Paper
[Paper](https://arxiv.org/abs/2407.17705)

**Abstract.**  Unsupervised anomaly localization on industrial textured images has achieved remarkable results through reconstruction-based methods, yet existing approaches based on image reconstruction and feature reconstruction each have their own shortcomings. Firstly, image-based methods tend to reconstruct both normal and anomalous regions well, which  lead to over-generalization. Feature-based methods contain a large amount of distinguishable semantic information, however, its feature structure is redundant and lacks anomalous information, which leads to significant reconstruction errors. In this paper, we propose an Anomaly Localization method based on Mamba with Feature Reconstruction and Refinement(ALMRR) which reconstructs semantic features based on Mamba and then refines them through a feature refinement module. To equip the model with prior knowledge of anomalies, we enhance it by adding artificially simulated anomalies to the original images. Unlike image reconstruction or repair, the features of synthesized defects are  repaired along with those of normal areas. Finally, the aligned features containing rich semantic information are fed into the refinement module to obtain the anomaly map. Extensive experiments have been conducted on the MVTec-AD-Textured  dataset and other real-world industrial dataset, which has demonstrated superior performance compared to state-of-the-art (SOTA) methods.

![framework](https://github.com/qsc1103/ALMRR/blob/main/figures/ALMRR.png?raw=true)
**The overview of our method’s pipeline** which consists four stages: simulate anomaly process, dual-branch feature embedding, Mamba feature reconstruction module and feature refinement module.

![mvtec-ad](https://github.com/qsc1103/ALMRR/blob/main/figures/MVTec-AD-textured.png?raw=true)
![other-datasets](https://github.com/qsc1103/ALMRR/blob/main/figures/MT-defect&NanoTWICE.png?raw=true)
Qualitative results of anomaly localization for both MVTec-AD-textured dataset and other real-world datasets.

## Environment
Create a new conda environment and install required packages.
```
conda create -n almrr_env python=3.10
conda activate almrr_env
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Usage
Download the pretrained **[resnet50](https://download.pytorch.org/models/resnet50-19c8e357.pth)** weights to **./ALMRR/almrr/pretrained/**
### Train
```
python main.py --mode train --data_name data_name --data_path data_path --anomaly_source_path anomaly_source_path
```

### Evaluation
```
python main.py --mode evaluation --data_name data_name --data_path data_path --anomaly_source_path anomaly_source_path
```

## Acknowledgement
Code reference [DFR](https://arxiv.org/abs/2012.07122)、[DRAEM](https://arxiv.org/abs/2108.07610) and [Vision Mamba](https://arxiv.org/abs/2401.09417)，thanks for the brilliant work.
