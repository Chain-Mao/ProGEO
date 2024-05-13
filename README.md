# ProGEO

<b>ProGEO: Generating Prompts through Image-Text Contrastive Learning For Visual Geo-localization, Chen Mao</b>

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rethinking-visual-geo-localization-for-large/visual-place-recognition-on-pittsburgh-250k)](https://paperswithcode.com/sota/visual-place-recognition-on-pittsburgh-250k?p=rethinking-visual-geo-localization-for-large)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rethinking-visual-geo-localization-for-large/visual-place-recognition-on-pittsburgh-30k)](https://paperswithcode.com/sota/visual-place-recognition-on-pittsburgh-30k?p=rethinking-visual-geo-localization-for-large)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rethinking-visual-geo-localization-for-large/visual-place-recognition-on-tokyo247)](https://paperswithcode.com/sota/visual-place-recognition-on-tokyo247?p=rethinking-visual-geo-localization-for-large)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rethinking-visual-geo-localization-for-large/visual-place-recognition-on-mapillary-val)](https://paperswithcode.com/sota/visual-place-recognition-on-mapillary-val?p=rethinking-visual-geo-localization-for-large)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rethinking-visual-geo-localization-for-large/visual-place-recognition-on-st-lucia)](https://paperswithcode.com/sota/visual-place-recognition-on-st-lucia?p=rethinking-visual-geo-localization-for-large)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rethinking-visual-geo-localization-for-large/visual-place-recognition-on-sf-xl-test-v1)](https://paperswithcode.com/sota/visual-place-recognition-on-sf-xl-test-v1?p=rethinking-visual-geo-localization-for-large)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rethinking-visual-geo-localization-for-large/visual-place-recognition-on-sf-xl-test-v2)](https://paperswithcode.com/sota/visual-place-recognition-on-sf-xl-test-v2?p=rethinking-visual-geo-localization-for-large)

This repository contains the official python implementation for our paper at ICANN 2024 "ProGEO: Generating Prompts through Image-Text Contrastive Learning For Visual Geo-localization, Chen Mao, Jingqi Hu et al.".

[[ArXiv](https://arxiv.org/abs/2204.afsf)]

## Dateset

Firstly, you can download the dataset called San Francisco eXtra Large (SF-XL, go [_here_](https://forms.gle/wpyDzhDyoWLQygAT9)).

The SF-XL dataset is about 1 TB.
For training only a subset of the images is used, and you can use this subset for training, which is only 360 GB.
More information on the dataset and lightweight version that you can find on the dataset download page (go [_here_](https://forms.gle/wpyDzhDyoWLQygAT9)).


## Train

#### Stage1

`$ python3 train_clip_stage1.py --train_set_folder path/to/processed/train --val_set_folder path/to/sf_xl/processed/val --test_set_folder path/to/sf_xl/processed/test --backbone CLIP-RN50 --groups_num 1`

#### Stage2

`$ python3 train_clip_stage2.py --train_set_folder path/to/processed/train --val_set_folder path/to/processed/val --test_set_folder path/to/processed/test --backbone CLIP-RN50 --fc_output_dim 1024 --prompt_learners path/to/logs/default/stage1/VIT16/last_prompt_learners.pth`

To change the backbone or the output descriptors dimensionality simply run 

`$ python3 train.py --backbone CLIP-ViT-B-16 --fc_output_dim 512`

Run `$ python3 train.py -h` to have a look at all the hyperparameters that you can change. You will find all hyperparameters mentioned in the paper.

## Test
You can test a trained model as such

`$ python3 eval.py --backbone CLIP-RN50 --resume_model path/to/best_model.pth --test_set_folder path/to/processed/test`

## Model Zoo

You can download the trained models from the table below, which provides links to models with different backbones and dimensionality of descriptors, trained on SF-XL.

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
