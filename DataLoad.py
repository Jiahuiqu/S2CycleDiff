import numpy as np
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
import os
from torchvision import transforms
from torch import nn
import torch
import cv2

unloader = transforms.ToPILImage()


def gaussian_blur_4d_tensor(tensor, sigma):

    # 将四维张量转换为三维张量
    tensor_3d = tensor

    # 遍历三维张量的每个元素，对后两个维度进行高斯模糊
    for i in range(tensor_3d.shape[0]):
        # 使用PyTorch的高斯模糊函数进行模糊处理
        tensor_3d[i] = torch.from_numpy(cv2.GaussianBlur(tensor_3d[i].numpy(), ksize=(3, 3), sigmaX=sigma))

    # 将三维张量重新转换为四维张量
    tensor_blurred = tensor_3d

    return tensor_blurred


class DataLoad(Dataset):
    def __init__(self, root, mode):
        super(DataLoad, self).__init__()
        self.root = root
        self.mode = mode
        self.hrMS = []
        self.LRHS = []
        self.HRMS = []
        self.upsample = nn.Upsample((128, 128), mode='bicubic')
        self.upsample2 = nn.Upsample((512, 512), mode='bicubic')
        self.downsample = nn.Upsample((32, 32), mode='bicubic')

        self.image_transform_H = transforms.RandomHorizontalFlip(p=1)
        self.image_transform_V = transforms.RandomVerticalFlip(p=1)

        if self.mode == "train":
            self.gtHS_idx = os.listdir(os.path.join(self.root, "train", "gtHS"))
            self.gtHS_idx.sort(key=lambda x: int(x.split(".")[0]))

            self.hrMS_idx = os.listdir(os.path.join(self.root, "train", "HRMS"))
            self.hrMS_idx.sort(key=lambda x: int(x.split(".")[0]))


        elif self.mode == "test":
            self.gtHS_idx = os.listdir(os.path.join(self.root, "test", "gtHS"))
            self.gtHS_idx.sort(key=lambda x: int(x.split(".")[0]))

            self.hrMS_idx = os.listdir(os.path.join(self.root, "test", "HRMS"))
            self.hrMS_idx.sort(key=lambda x: int(x.split(".")[0]))


    def __len__(self):
        return len(self.gtHS_idx)

    def __getitem__(self, index):
        if self.mode == 'train':
            self.gtHS= torch.from_numpy(loadmat(os.path.join(self.root, "train", "gtHS", self.gtHS_idx[index]))['gtHS'])
            self.hrMS = torch.from_numpy(
                loadmat(os.path.join(self.root, "train", "HRMS", self.gtHS_idx[index]))['hrMS'])
            self.lrHS = torch.from_numpy(
                loadmat(os.path.join(self.root, "train", "LRHS", self.gtHS_idx[index]))['lrHS'])
            # self.lrHS = torch.zeros_like(self.gtHS)

        elif self.mode == 'test':
            self.gtHS = torch.from_numpy(loadmat(os.path.join(self.root, "test", "gtHS", self.gtHS_idx[index]))['gtHS'])
            self.hrMS = torch.from_numpy(
                loadmat(os.path.join(self.root, "test", "HRMS", self.gtHS_idx[index]))['hrMS'])
            self.lrHS = torch.from_numpy(
                loadmat(os.path.join(self.root, "test", "LRHS", self.gtHS_idx[index]))['lrHS'])
            # self.lrHS = torch.zeros_like(self.gtHS)
            # data_lrHS = self.upsample(self.downsample(torch.from_numpy(self.LRHS[index, :, :, :]).unsqueeze(0)))
            # data_Pan = self.upsample(torch.from_numpy(self.HRMS[index, :, :, :]).unsqueeze(0))
            # plt.figure(figsize=(15, 10))
            # plt.subplot(1, 3, 1)
            # plt.axis("off")
            # plt.title("MSI")
            # plt.imshow(np.transpose(torchvision.utils.make_grid(self.lrHS.squeeze(0),
            #                                                 nrow=2, padding=1, normalize=True).cpu(), (0, 1, 2))[:,:, [15,10,5]])
        # plt.subplot(1, 3, 2)
        # plt.axis("off")
        # plt.title("GT")
        # plt.imshow(np.transpose(torchvision.utils.make_grid(self.gtHS[:,[30, 15, 5],:, :].squeeze(0),
        #                                                     nrow=2, padding=1, normalize=True).cpu(), (1, 2, 0)))
        # plt.subplot(1, 3, 3)
        # plt.axis("off")
        # plt.title("HSI")
        # plt.imshow(np.transpose(torchvision.utils.make_grid(self.lrHS[:,[30, 15, 5],:, :].squeeze(0),
        #                                                     nrow=2, padding=1, normalize=True).cpu(), (1, 2, 0)))
        #     plt.show()

        # data_LRHS = loadmat(os.path.join(self.root, self.mode, "LRHS", LR_HS))['da'].reshape(102, 40, 40)
        # data_HRMS = loadmat(os.path.join(self.root, self.mode, "HRMS", HR_MS))['hrMS'].reshape(4, 160, 160)
        if self.mode == 'train':
            return self.gtHS.squeeze(0), self.hrMS.squeeze(0), self.lrHS.squeeze(0)
        else:
            return self.gtHS.squeeze(0), self.hrMS.squeeze(0), self.lrHS.squeeze(0)
