# S $^2$ CycleDiff
The python code implementation of the AAAI 2024 paper "LDS$^2$AE: Local Diffusion Shared-Specific Autoencoder for Multimodal Remote Sensing Image Classification with Arbitrary Missing Modalities"

# Requirements

- Ubuntu 20.04   cuda 11.0
- Python 3.7  Pytorch 1.7

## Hyperparameters

The optimizer is Adam.

The more detailed training settings are shown in experiments of this paper.

Training
just run the S $^2$ CycleDiff.py

Testing
just run the S $^2$ CycleDiff_test.py

# Cite
@inproceedings{qu2024missing,  
     &emsp; title={S $^2$ CycleDiff: Spatial-Spectral-Bilateral Cycle-Diffusion Frameworkfor Hyperspectral Image Super-Resolution},  
     &emsp; author={Jiahui Qu, Jie He, Wenqian Dong, Jinyu Zhao},  
     &emsp; booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},  
     &emsp; year={2024}  
}
