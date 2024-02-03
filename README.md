# S $^2$ CycleDiff
The python code implementation of the paper "LDS$^2$AE: Local Diffusion Shared-Specific Autoencoder for Multimodal Remote Sensing Image Classification with Arbitrary Missing Modalities"

# Requirements

- Ubuntu 20.04   cuda 11.0
- Python 3.7  Pytorch 1.7

## Hyperparameters

The windowsize is set to 11 for Trento and Berlin，and is set to 27 for Houston2013！

The train_num_perclass is set to 40.

The batchsize is set to 16.

The optimizer is Adam.

The more detailed training settings are shown in experiments of this paper.

Training & Testing
just run the LDS2AE_main.py

# Cite
@inproceedings{qu2024missing,  
     &emsp; title={S $^2$ CycleDiff: Spatial-Spectral-Bilateral Cycle-Diffusion Frameworkfor Hyperspectral Image Super-Resolution},  
     &emsp; author={Jiahui Qu, Jie He, Wenqian Dong, Jinyu Zhao},  
     &emsp; booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},  
     &emsp; year={2024}  
}
