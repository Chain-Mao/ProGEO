# ProGEO

<b>ProGEO: Generating Prompts through Image-Text Contrastive Learning For Visual Geo-localization, Chen Mao</b>

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rethinking-visual-geo-localization-for-large/visual-place-recognition-on-pittsburgh-250k)](https://paperswithcode.com/sota/visual-place-recognition-on-pittsburgh-250k?p=rethinking-visual-geo-localization-for-large)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rethinking-visual-geo-localization-for-large/visual-place-recognition-on-pittsburgh-30k)](https://paperswithcode.com/sota/visual-place-recognition-on-pittsburgh-30k?p=rethinking-visual-geo-localization-for-large)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rethinking-visual-geo-localization-for-large/visual-place-recognition-on-tokyo247)](https://paperswithcode.com/sota/visual-place-recognition-on-tokyo247?p=rethinking-visual-geo-localization-for-large)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rethinking-visual-geo-localization-for-large/visual-place-recognition-on-mapillary-val)](https://paperswithcode.com/sota/visual-place-recognition-on-mapillary-val?p=rethinking-visual-geo-localization-for-large)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rethinking-visual-geo-localization-for-large/visual-place-recognition-on-st-lucia)](https://paperswithcode.com/sota/visual-place-recognition-on-st-lucia?p=rethinking-visual-geo-localization-for-large)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rethinking-visual-geo-localization-for-large/visual-place-recognition-on-sf-xl-test-v1)](https://paperswithcode.com/sota/visual-place-recognition-on-sf-xl-test-v1?p=rethinking-visual-geo-localization-for-large)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rethinking-visual-geo-localization-for-large/visual-place-recognition-on-sf-xl-test-v2)](https://paperswithcode.com/sota/visual-place-recognition-on-sf-xl-test-v2?p=rethinking-visual-geo-localization-for-large)

This repository contains the official python implementation for our paper at ICANN 2024 "ProGEO: Generating Prompts through Image-Text Contrastive Learning For Visual Geo-localization, Chen Mao, Jingqi Hu and etc".

[[ArXiv](https://arxiv.org/abs/2204.afsf)]

## Dateset

Firstly, you can download the dataset called San Francisco eXtra Large (SF-XL, go [_here_](https://forms.gle/wpyDzhDyoWLQygAT9)).

The SF-XL dataset is about 1 TB.
For training only a subset of the images is used, and you can use this subset for training, which is only 360 GB.
More information on the dataset and lightweight version are on the README that you can find on the dataset download page (go [_here_](https://forms.gle/wpyDzhDyoWLQygAT9) to find it).


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

## Trained Models

You can download the trained models from the table below, which provides links to models with different backbones and dimensionality of descriptors, trained on SF-XL.

<table>
  <tr>
    <th rowspan=2>Model</th>
    <th colspan=7>Dimension of Descriptors</th>
  </tr>
  <tr>
    <td>32</td>
    <td>64</td>
    <td>128</td>
    <td>256</td>
    <td>512</td>
    <td>1024</td>
    <td>2048</td>
  </tr>
  <tr>
    <td>ResNet-18</td>
    <td><a href="https://drive.google.com/file/d/1tfT8r2fBeMVAEHg2bVfCql5pV9YzK620/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1-d_Yi3ly3bY6hUW1F9w144FFKsZtYBL4/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1HaQjGY5x--Ok0RcspVVjZ0bwrAVmBvrZ/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1hjkogugTsHTQ6GTuW3MHqx-t4cXqx0uo/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1rQAC2ZddDjzwB2OVqAcNgCFEf3gLNa9U/view?usp=sharing">link</a></td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>ResNet-50</td>
    <td><a href="https://drive.google.com/file/d/18AxbLO66CO0kG05-1YrRb1YwqN7Wgp6Z/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1F2WMt7vMUqXBjsZDIwSga3N0l0r9NP2s/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/14U3jsoNEWC-QsINoVCWZaHFUGE20fIgZ/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1Q2sZPEJfHAe19JaZkdgeFotUYwKbV_x2/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1LgDaxCjbQqQWuk5qrPogfg7oN8Ksl1jh/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1VBLUiQJfmnZ4kVQIrXBW-AE1dZ3EnMv2/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1yNzxsMg34KO04UJ49ncANdCIWlB3aUGA/view?usp=sharing">link</a></td>
  </tr>
  <tr>
    <td>ResNet-101</td>
    <td><a href="https://drive.google.com/file/d/1a5FqhujOn0Pr6duKrRknoOgz8L8ckDSE/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/17C8jBQluxsbI9d8Bzf67b5OsauOJAIuX/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1w37AztnIyGVklBMtm-lwkajb0DWbYhhc/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1G5_I4vX4s4_oiAC3EWbrCyXrCOkV8Bbs/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1uBKpNfMBt6sLIjCGfH6Orx9eQdQgN-8Z/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/12BU8BgfqFYzGLXXNaKLpaAzTHuN5I9gQ/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1PF7lsSw1sFMh-Bl_xwO74fM1InyYy1t8/view?usp=sharing">link</a></td>
  </tr>
  <tr>
    <td>ResNet-152</td>
    <td><a href="https://drive.google.com/file/d/12pI1FToqKKt8I6-802CHWXDP-JmHEFSW/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1rTjlv_pNtXgxY8VELiGYvLcgXiRa2zqB/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1q5-szPBn4zL8evWmYT04wFaKjen66mrk/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1sCQMA_rsIjmD-f381I0f2yDf0At4TnSx/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1ggNYQfGSfE-dciKCS_6SKeQT76O0OXPX/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/15vBWuHVqEMxkAWWrc7IrkGsQroC65tPc/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1AlF7xPSswDLA1TdhZ9yTVBkfRnJm0Hn8/view?usp=sharing">link</a></td>
  </tr>
  <tr>
    <td>VGG-16</td>
    <td>-</td>
    <td><a href="https://drive.google.com/file/d/1YJTBwagC0v50oPydpKtsTnGZnaYOV0z-/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1vgw509lGBfJR46cGDJGkFcdBTGhIeyAH/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1-4JtACE47rkXXSAlRBFIbydimfKemdo7/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1F6CT-rnAGTTexdpLoQYncn-ooqzJe6wf/view?usp=sharing">link</a></td>
    <td>-</td>
    <td>-</td>
  </tr>
</table>

## Issues
If you have any questions regarding our code or model, feel free to open an issue or send an email to maochen981203@gmail.com

## Acknowledgements
Parts of this repo are inspired by the following repositories:
- [CoOp in PyTorch](https://github.com/KaiyangZhou/CoOp)
- [Cosplace in PyTorch](https://github.com/gmberton/CosPlace)
- [Visual Geo-localization benchmark](https://github.com/gmberton/deep-visual-geo-localization-benchmark)
