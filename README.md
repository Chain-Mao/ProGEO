[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/progeo-generating-prompts-through-image-text/visual-place-recognition-on-msls)](https://paperswithcode.com/sota/visual-place-recognition-on-msls?p=progeo-generating-prompts-through-image-text)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/progeo-generating-prompts-through-image-text/visual-place-recognition-on-pittsburgh-30k)](https://paperswithcode.com/sota/visual-place-recognition-on-pittsburgh-30k?p=progeo-generating-prompts-through-image-text)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/progeo-generating-prompts-through-image-text/visual-place-recognition-on-sf-xl-test-v1)](https://paperswithcode.com/sota/visual-place-recognition-on-sf-xl-test-v1?p=progeo-generating-prompts-through-image-text)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/progeo-generating-prompts-through-image-text/visual-place-recognition-on-sf-xl-test-v2)](https://paperswithcode.com/sota/visual-place-recognition-on-sf-xl-test-v2?p=progeo-generating-prompts-through-image-text)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/progeo-generating-prompts-through-image-text/visual-place-recognition-on-st-lucia)](https://paperswithcode.com/sota/visual-place-recognition-on-st-lucia?p=progeo-generating-prompts-through-image-text)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/progeo-generating-prompts-through-image-text/visual-place-recognition-on-tokyo247)](https://paperswithcode.com/sota/visual-place-recognition-on-tokyo247?p=progeo-generating-prompts-through-image-text)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/progeo-generating-prompts-through-image-text/visual-place-recognition-on-pittsburgh-250k)](https://paperswithcode.com/sota/visual-place-recognition-on-pittsburgh-250k?p=progeo-generating-prompts-through-image-text)

# ProGEO

<b>ProGEO: Generating Prompts through Image-Text Contrastive Learning For Visual Geo-localization</b>

This repository contains the official python implementation for our paper at ICANN 2024 "ProGEO: Generating Prompts through Image-Text Contrastive Learning For Visual Geo-localization, Chen Mao, Jingqi Hu et al". 
Our paper are available at [here](https://arxiv.org/abs/2406.01906).

## Introduction

Visual Geo-localization (VG) refers to the process to identify the location described in query images, which is widely applied in robotics field and computer vision tasks, such as autonomous driving, metaverse, augmented reality, and SLAM. In fine-grained images lacking specific text descriptions, directly applying pure visual methods to represent neighborhood features often leads to the model focusing on overly fine-grained features, unable to fully mine the semantic information in the images. Therefore, we propose a two-stage training method to enhance visual performance and use contrastive learning to mine challenging samples. 

<img align="center" width="80%" src="https://github.com/Chain-Mao/ProGEO/blob/main/all.png">

We first leverage the multi-modal description capability of CLIP (Contrastive Language-Image Pretraining) to create a set of learnable text prompts for each geographic image feature to form vague descriptions. Then, by utilizing dynamic text prompts to assist the training of the image encoder, we enable the image encoder to learn better and more generalizable visual features. This strategy of applying text to purely visual tasks addresses the challenge of using multi-modal models for geographic images, which often suffer from a lack of precise descriptions, making them difficult to utilize widely. We validate the effectiveness of the proposed strategy on several large-scale visual geo-localization datasets, and our method achieves competitive results on multiple visual geo-localization datasets.

## Dateset

Firstly, you can download the dataset called San Francisco eXtra Large (SF-XL, go [_here_](https://forms.gle/wpyDzhDyoWLQygAT9)).
The SF-XL dataset is about 1 TB.
For training only a subset of the images is used, and you can use this subset for training, which is only 360 GB.
More information on the dataset and lightweight version that you can find on the dataset download page (go [_here_](https://forms.gle/wpyDzhDyoWLQygAT9)).

## Train

#### Stage1

The purpose of the first stage training is to generate prompts that describe the image.

<img align="center" width="80%" src="https://github.com/Chain-Mao/ProGEO/blob/main/stage1.png">

After downloading the SF-XL dataset, you can start the stage1 training as such

`$ python3 train_clip_stage1.py --train_set_folder path/to/processed/train --val_set_folder path/to/sf_xl/processed/val --test_set_folder path/to/sf_xl/processed/test --backbone CLIP-RN50 --groups_num 1`

#### Stage2

The purpose of the second stage training is to use prompts to assist the image model to complete the clustering.

<img align="center" width="80%" src="https://github.com/Chain-Mao/ProGEO/blob/main/stage2.png">

After generating the prompts through stage1 training, you can start the stage2 training as such

`$ python3 train_clip_stage2.py --train_set_folder path/to/processed/train --val_set_folder path/to/processed/val --test_set_folder path/to/processed/test --backbone CLIP-RN50 --fc_output_dim 1024 --prompt_learners path/to/logs/default/stage1/VIT16/last_prompt_learners.pth`

To change the backbone and the output descriptors dimensionality simply run 

`$ python3 train.py --backbone CLIP-ViT-B-16 --fc_output_dim 512`

Run `$ python3 train.py -h` to have a look at all the hyperparameters that you can change. You will find all hyperparameters mentioned in the paper.

## Test

You can test a trained model as such

`$ python3 eval.py --backbone CLIP-RN50 --resume_model path/to/best_model.pth --test_set_folder path/to/processed/test`

<img align="center" width="80%" src="https://github.com/Chain-Mao/ProGEO/blob/main/visual.png">

## Model Zoo

You can download the trained models from the table below, which provides links to models with different visual backbones and dimensionality of descriptors, trained on SF-XL.

| Visual Model | Dimension | Link | Password |
|----------|------|------|--------|
| CLIP-ResNet50  | 1024 | [https://pan.baidu.com/s/1X9iGoVRy_Fc0HwTSUvclVQ](https://pan.baidu.com/s/1X9iGoVRy_Fc0HwTSUvclVQ) | fw3t |
| CLIP-ResNet101 | 512 | [https://pan.baidu.com/s/1U8MYcFeRZfLz8r5Xx30eFg](https://pan.baidu.com/s/1U8MYcFeRZfLz8r5Xx30eFg) | gh6z |
| CLIP-ViT-B-16  | 512 | [https://pan.baidu.com/s/1O82EYD-0WmHC6Wx-a5B_0g](https://pan.baidu.com/s/1O82EYD-0WmHC6Wx-a5B_0g) | vzho |
| CLIP-ViT-B-32  | 512 | [https://pan.baidu.com/s/19vJv4OE31XSCYIIZ2X1qSQ](https://pan.baidu.com/s/19vJv4OE31XSCYIIZ2X1qSQ) | x0xb |


## Issues
If you have any questions regarding our code or model, feel free to open an issue or send an email to maochen981203@gmail.com

## Acknowledgements
Parts of this repo are inspired by the following repositories:
- [CoOp in PyTorch](https://github.com/KaiyangZhou/CoOp)
- [Cosplace in PyTorch](https://github.com/gmberton/CosPlace)
- [Visual Geo-localization benchmark](https://github.com/gmberton/deep-visual-geo-localization-benchmark)

## Cite
Here is the bibtex to cite our arxiv paper, the Springer version will be cited after official publication.
```
@ARTICLE{2024arXiv240601906M,
       author = {{Mao}, Chen and {Hu}, Jingqi},
        title = "{ProGEO: Generating Prompts through Image-Text Contrastive Learning for Visual Geo-localization}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Computer Vision and Pattern Recognition, Computer Science - Information Retrieval},
         year = 2024,
        month = jun,
          eid = {arXiv:2406.01906},
        pages = {arXiv:2406.01906},
          doi = {10.48550/arXiv.2406.01906},
archivePrefix = {arXiv},
       eprint = {2406.01906},
 primaryClass = {cs.CV},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024arXiv240601906M},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
