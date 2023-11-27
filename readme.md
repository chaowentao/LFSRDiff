This repository contains the implementation of the following paper:

  LFSRDiff: Light Field Image Super-Resolution via Diffusion Models

# Get Started

## Environment Installation

```bash
conda create -n lfsrdiff python=3.8
conda activate lfsrdiff
pip install -r requirements.txt
```

## Dataset Preparation

Following previous methods, we use five mainstream LF image datasets, i.e., EPFL, HCINew, HCIold, INRIA, and STFgantry for our LFSR experiments and convert the input LF image to YCbCr color space and only super-resolve the Y channel of the image, upsampling the Cb and Cr channel images bicubic. Please refer to https://github.com/ZhengyuLiang24/BasicLFSR


## Pretrained Model

### LF Encoder Model
We provide the pretrained LF encoder model in the folder ``pretrain_models`` of this repository.

### LFSRDiff Model
We provide the pretrained LFSRDiff model in the folder ``checkpoints`` of this repository.

## Train & Evaluate
1. Prepare datasets. Please refer to ``Dataset Preparation``.
2. Modify config files.
    - lf encoder: configs/epit/lfsr4x_pretrain.yaml
    - lfsrdiff: configs/diffsr_epit_distgunet_lfsr4x.yaml
3. Run training / evaluation code. The code is for training and testing on 1 Nvidia A100 GPU with 40G. 

### Training 4x SR

```bash
# train LF Encoder
python tasks/trainer.py --config configs/epit/lfsr4x_pretrain.yaml --exp_name epit_y_lfsr4x_c32_bs4 --reset

# train LFSRDiff
python tasks/trainer.py --config configs/diffsr_epit_distgunet_lfsr4x.yaml --exp_name diffsr_epit_distgunet_m1111_lfsr4x_c32_bs4 --reset

# train LFSRDiff with pretrained LF encoder model
python tasks/trainer.py --config configs/diffsr_pre_epit_distgunet_lfsr4x.yaml --exp_name diffsr_pre_epit_distgunet_m1111_lfsr4x_c64_bs2_1e4 --reset
```

### Evaluate 4x SR

```bash
# generate samples
python tasks/trainer.py --config configs/diffsr_epit_distgunet_lfsr4x.yaml --exp_name diffsr_epit_distgunet_m1111_lfsr4x_c32_bs4 --infer

# generate samples with pretrained model. For particularly large images, a crop test is required.
python tasks/trainer.py --config configs/diffsr_pre_epit_distgunet_lfsr4x.yaml --exp_name diffsr_pre_epit_distgunet_m1111_lfsr4x_c64_bs2_1e4 --infer

# evaluate with BasicLFSR evaluation code for PSNR/SSIM/LPIPS and modify path of ``result_dir`` 
python test_lfsrdiff_metrics_plus.py
```

