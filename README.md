# Equivariant Bayesian Hyperspectral Imaging via Mosaiced and PAN Image Fusion (EBVIF)

[![IEEE Xplore](https://img.shields.io/badge/IEEE_Xplore-Paper-00629B?style=flat-square&logo=ieee&logoColor=white)](https://ieeexplore.ieee.org/document/11480451)

Official PyTorch implementation of the paper: **"Equivariant Bayesian Hyperspectral Imaging via Mosaiced and PAN Image Fusion (EBVIF)"**.

## 📖 Introduction

In this repository, we provide the code for our proposed **Equivariant Bayesian Variational Inference Framework (EBVIF)** along with implementations of **8 state-of-the-art (SOTA) competing methods**.

Our goal is to provide a comprehensive benchmark for mosaiced and PAN image fusion. All methods share a **unified framework** and consistent execution logic, making it easy for researchers to reproduce results and compare performance.

## 📂 Project Structure

This repository contains **13 individual projects** (sub-folders), including:

- **Proposed Method:** `EBVIF`
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
Pretrained weights of all the fusion methods are packaged in [Release](https://github.com/Nan-Wong98/Equivariant-Bayesian-variational-inference-framework/releases/tag/tag1). Please refer to the directory mapping below:

### Competing Methods
Two stage: demosaicing + pansharpening

| Component | CAVE | ICVL | Real-world | Note |
| :--- | :---: | :---: | :---: | :--- |
| **Demosaicing** | Folder "1" | Folder "3" | Folder "5" | Used for SpNet, SFNet, LSAN |
| **Pansharpening**| Folder "2" | Folder "4" | Folder "6" | Used for PanGAN, VBPN, WFANet |

> **Note:** PPID uses a traditional demosaicing algorithm and does not require pretrained weights for the first stage.

### Proposed Method (EBVIF)
**EBVIF** is a one-step fusion framework.

| Method | CAVE | ICVL | Real-world |
| :--- | :---: | :---: | :---: |
| **EBVIF (Proposed)** | Folder "1" | Folder "2" | Folder "3" |


## 💾 Data Preparation

We utilize both simulated datasets (CAVE, ICVL) and an acquired real-world dataset for our experiments. Please refer to the following instructions to prepare the data.

The dataset is organized as follows:
```
Dataset/
├── CAVE/
│ ├── train/
│ └── test/
├── ICVL/
│ ├── train/
│ └── test/
└── real_world/
├── train/
└── test/
```

### 1. Download Links
Due to the file size, we host the prepared datasets on Baidu Netdisk:

* **CAVE Dataset (Simulated)**
    * 📥 [Download Link](https://pan.baidu.com/s/1DrPQRold0AAR89hU9sfQ_Q)
    * 🔑 Access Code: `bk9p`

* **Acquired Real-world Dataset**
    * 📥 [Download Link](https://pan.baidu.com/s/12eg0GUfOky89Zkzqs6Rb0Q)
    * 🔑 Access Code: `5ywe`

### 2. ICVL Dataset Setup
The ICVL dataset is not redistributed here due to its large scale. Please download the original data from the [Official Website](https://cndaqiang.github.io/ICVL-Natural-Hyperspectral-Image-Database/index.html).

**Train/Test Split:**
Among the total 200 images, we strictly follow the split of **160 images for training** and **40 images for testing**.

<details>

'bguCAMP_0514-1711.mat', 'bguCAMP_0514-1718.mat', 'bguCAMP_0514-1724.mat', 'bgu_0403-1439.mat', 'bgu_0403-1511.mat', 'bgu_0403-1525.mat', 'BGU_0522-1217.mat', 'eve_0331-1601.mat', 'eve_0331-1618.mat', 'Flower_0325-1336.mat', 'gavyam_0823-0944.mat', 'Labtest_0910-1510.mat', 'lehavim_0910-1607.mat', 'Lehavim_0910-1629.mat', 'Lehavim_0910-1630.mat', 'Lehavim_0910-1635.mat', 'Lehavim_0910-1636.mat', 'nachal_0823-1040.mat', 'nachal_0823-1118.mat', 'nachal_0823-1127.mat', 'nachal_0823-1152.mat', 'objects_0924-1605.mat', 'objects_0924-1620.mat', 'objects_0924-1622.mat', 'objects_0924-1628.mat', 'objects_0924-1631.mat', 'objects_0924-1645.mat', 'omer_0331-1102.mat', 'omer_0331-1118.mat', 'omer_0331-1130.mat', 'peppers_0503-1330.mat', 'peppers_0503-1332.mat', 'pepper_0503-1228.mat', 'pepper_0503-1229.mat', 'prk_0328-1025.mat', 'rmt_0328-1241-1.mat', 'rmt_0328-1249-1.mat', 'rsh_0406-1413.mat', 'rsh_0406-1443.mat', 'sat_0406-1107.mat'

</details>

## ⚙️ Requirements

Install dependencies via:
```bash
pip install -r requirements.txt
````

<details>

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

</details>

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
  doi={10.1109/TPAMI.2026.3682543}
  }
```

## 📧 Contact
If any question, please contact with me.

E-mail: wangn@hnu.edu.cn

Wechat: wn343816504

## 🙏 Acknowledgments
This project is based on the Equivariant Fusion Network (EFN) framework from our previous work:

"Equivariant High-Resolution Hyperspectral Imaging via Mosaiced and PAN Image Fusion" (IEEE TIP 2026)📥[Github](https://github.com/Nan-Wong98/Equivariant-High-Resolution-Hyperspectral-Imaging-via-Mosaiced-and-PAN-Image-Fusion)

We extend the unified benchmark framework from EFN and introduce our Equivariant Bayesian Variational Inference Framework (EBVIF) with Bayesian uncertainty modeling for more complexed applications.