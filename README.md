# Equivariant Bayesian Hyperspectral Imaging via Mosaiced and PAN Image Fusion

[![Paper](https://ieeexplore.ieee.org/document/11480451)]

Official PyTorch implementation of the paper: **"Equivariant Bayesian Hyperspectral Imaging via Mosaiced and PAN Image Fusion"**.

## 📖 Introduction

In this repository, we provide the code for our proposed **Equivariant Bayesian Variantional Inference Framework (EBVIF)** along with implementations of **8 state-of-the-art (SOTA) competing methods**. 

## 📂 Project Structure

This repository contains **8 individual projects** (sub-folders), including:

- **Proposed Method:** `EBVIF` (Equivariant Bayesian Variantional Inference Framework)
- **Competing Methods:** `PPID_PanGAN`, `PPID_VBPN`, `SpNet_PanGAN`, `SpNet_VBPN`, `SFNet_PanGAN`, `SFNet_VBPN`, `LSAN_PanGAN`, `LSAN_VBPN`.

### Unified Workflow
Each project folder follows the exact same file structure and logic:

| File Name | Function |
| :--- | :--- |
| `GetDataSet.py` | 🛠 **Data Preparation:** Generates training/testing data from raw datasets. |
| `train.py` | 🚀 **Training:** Trains the model. |
| `generate.py` | 💾 **Inference:** Generates fusion results using pretrained weights. |
| `test.py` | 📊 **Evaluation:** Calculates quantitative metrics (PSNR, SAM, ERGAS, Q2n, QNR etc.). |
| `visualize.py` | 🎨 **Visualization:** Visualizes the generated HSI and MAE map results. |

## 📦 Pretrained Weights
Pretrained weights of all the fusion methods are packaged in [Release](https://github.com/Nan-Wong98/Equivariant-Bayesian-variational-inference-framework/releases/tag/tag1).

## ⚙️ Requirements
*   h5py==3.15.1
*   Imath==0.0.2
*   matplotlib==3.10.8
*   numpy==2.4.1
*   opencv_python==4.13.0.90
*   OpenEXR==3.4.4
*   pytorch_msssim==1.0.0
*   scipy==1.17.0
*   torch==2.10.0+cu126
*   torchvision==0.25.0+cu126
*   tqdm==4.67.1

Install dependencies via:
```bash
pip install -r requirements.txt
````

## 📝 Citation

If you find this code or our dataset useful for your research, please verify strictly and cite our paper:
```
@ARTICLE{11480451,
  author={Dian, Renwei and Wang, Nan and Guo, Anjing and Li, Shutao},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Equivariant Bayesian Hyperspectral Imaging via Mosaiced and PAN Image Fusion}, 
  year={2026},
  volume={},
  number={},
  pages={1-16},
}
```

## 📧 Contact
If any question, please contact with us.

E-mail: wangn@hnu.edu.cn
