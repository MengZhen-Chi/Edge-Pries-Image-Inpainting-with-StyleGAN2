# Edge-Pries-Image-Inpainting-with-StyleGAN2

This repository contains the inference code and pretrained weights for our method proposed in the paper:

> **[Edge Priors Image Inpainting with StyleGAN2]**  
> [Mengzhen Chi, Chong Fu, Xu Zheng, Jialei Chen, Qing Li, Chiu-Wing Sham]  
> [Journal/Conference Name], [Year]  
> [DOI or arXiv Link: ]

---

## 🖼️ Overview

This project aims to perform image restoration from damaged inputs, guided by edge or sketch contours. We provide inference scripts and pretrained weights on FFHQ for generating high quality results at 256 resolutions. We also provide the edge maps of FFHQ and CelebA-HQ, you can use the pretrained weights and edge maps to evaluate the FID and Lpips scores on CelebA-HQ or other face images.

---

## 🔧 Requirements

- Python ≥ 3.10  
- PyTorch = 1.12.1  
- torchvision  
- numpy  
- OpenCV  
- tqdm  
- (Optional) CUDA-enabled GPU

Install dependencies with:

```bash
pip install -r requirements.txt
```
## 🎞️ Edge Maps
We provide edge maps extracted from the FFHQ dataset using the DexiNed algorithm, which you can use to test FFHQ images.
```
https://pan.baidu.com/s/1pPa7YVk_W3K4Ky2P5JjK8g?pwd=2296
```
## 💿 Pretrained Weights
You can obtain the network weights trained on the 256* 256-pixel FFHQ dataset at the following link.
```
https://pan.baidu.com/s/19DZOitvfQGzONQAxz6lGrQ?pwd=2296
```
