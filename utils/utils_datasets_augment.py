import os

# from torch.utils.data import Dataset
from skimage import metrics
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor
import random
import matplotlib.pyplot as plt
import torch
import numpy as np
import h5py
from torch.utils.data import DataLoader
import cv2
import einops

from utils.imresize import imresize


class TrainSetDataLoader(Dataset):
    def __init__(self, args):
        super(TrainSetDataLoader, self).__init__()
        self.angRes_in = args.angRes_in
        self.angRes_out = args.angRes_out
        self.scale_factor = args.scale_factor
        self.prob = args.prob
        self.alpha = args.alpha
        self.augment = args.augment
        if args.task == "SR":
            self.dataset_dir = (
                args.path_for_train
                + "SR_"
                + str(args.angRes_in)
                + "x"
                + str(args.angRes_in)
                + "_"
                + str(args.scale_factor)
                + "x/"
            )
        elif args.task == "RE":
            self.dataset_dir = (
                args.path_for_train
                + "RE_"
                + str(args.angRes_in)
                + "x"
                + str(args.angRes_in)
                + "_"
                + str(args.angRes_out)
                + "x"
                + str(args.angRes_out)
                + "/"
            )
            pass

        if args.data_name == "ALL":
            self.data_list = os.listdir(self.dataset_dir)
        else:
            self.data_list = [args.data_name]

        self.file_list = []
        for data_name in self.data_list:
            tmp_list = os.listdir(self.dataset_dir + data_name)
            for index, _ in enumerate(tmp_list):
                tmp_list[index] = data_name + "/" + tmp_list[index]

            self.file_list.extend(tmp_list)

        self.item_num = len(self.file_list)

    def __getitem__(self, index):
        file_name = [self.dataset_dir + self.file_list[index]]
        with h5py.File(file_name[0], "r") as hf:
            Lr_SAI_y = np.array(hf.get("Lr_SAI_y"))  # Lr_SAI_y
            Hr_SAI_y = np.array(hf.get("Hr_SAI_y"))  # Hr_SAI_y
            if self.augment == "cutmib":
                Lr_SAI_y, Hr_SAI_y = augmentation_cutmib(
                    Lr_SAI_y,
                    Hr_SAI_y,
                    self.angRes_in,
                    self.scale_factor,
                    self.prob,
                    self.alpha,
                )
            elif self.augment == "cutmib_cwt":
                Lr_SAI_y, Hr_SAI_y = augmentation_cutmib_cwt(
                    Lr_SAI_y,
                    Hr_SAI_y,
                    self.angRes_in,
                    self.scale_factor,
                    self.prob,
                    self.alpha,
                )
            elif self.augment == "cutblur":
                Lr_SAI_y, Hr_SAI_y = augmentation_cutblur(
                    Lr_SAI_y,
                    Hr_SAI_y,
                    self.angRes_in,
                    self.scale_factor,
                    self.prob,
                    self.alpha,
                )
            elif self.augment == "cutblur_cwt":
                Lr_SAI_y, Hr_SAI_y = augmentation_cutblur_cwt(
                    Lr_SAI_y,
                    Hr_SAI_y,
                    self.angRes_in,
                    self.scale_factor,
                    self.prob,
                    self.alpha,
                )
            elif self.augment == "cutblend_cwt":
                Lr_SAI_y, Hr_SAI_y = augmentation_cutblend_cwt(
                    Lr_SAI_y,
                    Hr_SAI_y,
                    self.angRes_in,
                    self.scale_factor,
                    cutmix_prob=self.prob,
                    cutmix_alpha=self.alpha,
                )
            elif self.augment == "default":
                Lr_SAI_y, Hr_SAI_y = augmentation(Lr_SAI_y, Hr_SAI_y)
            Lr_SAI_y = ToTensor()(Lr_SAI_y.copy())
            Hr_SAI_y = ToTensor()(Hr_SAI_y.copy())

        Lr_angRes_in = self.angRes_in
        Lr_angRes_out = self.angRes_out

        return Lr_SAI_y, Hr_SAI_y, [Lr_angRes_in, Lr_angRes_out]

    def __len__(self):
        return self.item_num


def MultiTestSetDataLoader(args):
    # get testdataloader of every test dataset
    data_list = None
    if args.data_name in ["ALL", "RE_Lytro", "RE_HCI"]:
        if args.task == "SR":
            # dataset_dir = args.path_for_test + 'SR_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + \
            #               str(args.scale_factor) + 'x/'
            dataset_dir = (
                args.path_for_test
                + "SR_"
                + str(args.angRes_in)
                + "x"
                + str(args.angRes_in)
                + "_"
                + str(args.scale_factor)
                + "x/"
            )
            data_list = os.listdir(dataset_dir)
        elif args.task == "RE":
            dataset_dir = (
                args.path_for_test
                + "RE_"
                + str(args.angRes_in)
                + "x"
                + str(args.angRes_in)
                + "_"
                + str(args.angRes_out)
                + "x"
                + str(args.angRes_out)
                + "/"
                + args.data_name
            )
            data_list = os.listdir(dataset_dir)
    else:
        data_list = [args.data_name]

    test_Loaders = []
    length_of_tests = 0
    for data_name in data_list:
        test_Dataset = TestSetDataLoader(
            args, data_name, Lr_Info=data_list.index(data_name)
        )
        length_of_tests += len(test_Dataset)

        test_Loaders.append(
            DataLoader(
                dataset=test_Dataset,
                num_workers=args.num_workers,
                batch_size=1,
                shuffle=False,
            )
        )

    return data_list, test_Loaders, length_of_tests


class TestSetDataLoader(Dataset):
    def __init__(self, args, data_name="ALL", Lr_Info=None):
        super(TestSetDataLoader, self).__init__()
        self.angRes_in = args.angRes_in
        self.angRes_out = args.angRes_out

        if args.task == "SR":
            self.dataset_dir = (
                args.path_for_test
                + "SR_"
                + str(args.angRes_in)
                + "x"
                + str(args.angRes_in)
                + "_"
                + str(args.scale_factor)
                + "x/"
            )
            self.data_list = [data_name]
        elif args.task == "RE":
            self.dataset_dir = (
                args.path_for_test
                + "RE_"
                + str(args.angRes_in)
                + "x"
                + str(args.angRes_in)
                + "_"
                + str(args.angRes_out)
                + "x"
                + str(args.angRes_out)
                + "/"
                + args.data_name
                + "/"
            )
            self.data_list = [data_name]

        self.file_list = []
        for data_name in self.data_list:
            tmp_list = os.listdir(self.dataset_dir + data_name)
            for index, _ in enumerate(tmp_list):
                tmp_list[index] = data_name + "/" + tmp_list[index]

            self.file_list.extend(tmp_list)

        self.item_num = len(self.file_list)

    def __getitem__(self, index):
        file_name = [self.dataset_dir + self.file_list[index]]
        with h5py.File(file_name[0], "r") as hf:
            Lr_SAI_y = np.array(hf.get("Lr_SAI_y"))
            Hr_SAI_y = np.array(hf.get("Hr_SAI_y"))
            Sr_SAI_cbcr = np.array(hf.get("Sr_SAI_cbcr"), dtype="single")
            Lr_SAI_y = np.transpose(Lr_SAI_y, (1, 0))
            Hr_SAI_y = np.transpose(Hr_SAI_y, (1, 0))
            Sr_SAI_cbcr = np.transpose(Sr_SAI_cbcr, (2, 1, 0))

        Lr_SAI_y = ToTensor()(Lr_SAI_y.copy())
        Hr_SAI_y = ToTensor()(Hr_SAI_y.copy())
        Sr_SAI_cbcr = ToTensor()(Sr_SAI_cbcr.copy())

        Lr_angRes_in = self.angRes_in
        Lr_angRes_out = self.angRes_out
        LF_name = self.file_list[index].split("/")[-1].split(".")[0]

        return Lr_SAI_y, Hr_SAI_y, Sr_SAI_cbcr, [Lr_angRes_in, Lr_angRes_out], LF_name

    def __len__(self):
        return self.item_num


def flip_SAI(data, angRes):
    if len(data.shape) == 2:
        H, W = data.shape
        data = data.reshape(H, W, 1)

    H, W, C = data.shape
    data = data.reshape(angRes, H // angRes, angRes, W // angRes, C)  # [U, H, V, W, C]
    data = data[::-1, ::-1, ::-1, ::-1, :]
    data = data.reshape(H, W, C)

    return data


def augmentation(data, label):
    if random.random() < 0.5:  # flip along W-V direction
        data = data[:, ::-1]
        label = label[:, ::-1]
    if random.random() < 0.5:  # flip along W-V direction
        data = data[::-1, :]
        label = label[::-1, :]
    if random.random() < 0.5:  # transpose between U-V and H-W
        data = data.transpose(1, 0)
        label = label.transpose(1, 0)
    return data, label


def augmentation_cutmib(data, label, angRes=5, scale_factor=4, prob=0.9, alpha=0.1):
    if random.random() < 0.5:  # flip along W-V direction
        data = data[:, ::-1]
        label = label[:, ::-1]
    if random.random() < 0.5:  # flip along W-V direction
        data = data[::-1, :]
        label = label[::-1, :]
    if random.random() < 0.5:  # transpose between U-V and H-W
        data = data.transpose(1, 0)
        label = label.transpose(1, 0)
    # data: U * patchsize//scale, V * patchsize//scale
    # label: U * patchsize, V * patchsize
    lr_H, lr_W = data.shape
    hr_H, hr_W = label.shape
    # hr_H = lr_H * scale_factor
    # hr_W = lr_W * scale_factor
    patchsize = hr_H // angRes

    label = einops.rearrange(label, "(u H) (v W) -> (u v) H W", u=angRes, v=angRes)
    data = einops.rearrange(data, "(u H) (v W) -> (u v) H W", u=angRes, v=angRes)

    label_mix = label.mean(axis=0)
    data_mix = data.mean(axis=0)

    # alpha = 0.1, prob = 0.9
    # label, data = cutmib(label, data, label_mix, data_mix, scale_factor, prob, alpha)

    if np.random.rand(1) > prob:
        label, data = cutmib(
            label, data, label_mix, data_mix, scale_factor, prob, alpha
        )

    # normalization
    # label = torch.from_numpy(label.astype(np.float32) / 255.0)
    # data = torch.from_numpy(data.astype(np.float32) / 255.0)

    # data: U * patchsize//scale, V * patchsize//scale
    # label: U * patchsize, V * patchsize
    label = einops.rearrange(label, "(u v) H W -> (u H) (v W)", u=angRes, v=angRes)
    data = einops.rearrange(data, "(u v) H W -> (u H) (v W)", u=angRes, v=angRes)

    return data, label


def augmentation_cutmib_cwt(data, label, angRes=5, scale_factor=4, prob=0.9, alpha=0.1):
    if random.random() < 0.5:  # flip along W-V direction
        data = data[:, ::-1]
        label = label[:, ::-1]
    if random.random() < 0.5:  # flip along W-V direction
        data = data[::-1, :]
        label = label[::-1, :]
    if random.random() < 0.5:  # transpose between U-V and H-W
        data = data.transpose(1, 0)
        label = label.transpose(1, 0)
    # data: U * patchsize//scale, V * patchsize//scale
    # label: U * patchsize, V * patchsize
    lr_H, lr_W = data.shape
    hr_H, hr_W = label.shape
    # hr_H = lr_H * scale_factor
    # hr_W = lr_W * scale_factor
    patchsize = hr_H // angRes

    label = einops.rearrange(label, "(u H) (v W) -> (u v) H W", u=angRes, v=angRes)
    data = einops.rearrange(data, "(u H) (v W) -> (u v) H W", u=angRes, v=angRes)

    label_mix = label.mean(axis=0)
    data_mix = data.mean(axis=0)

    # alpha = 0.1, prob = 0.9
    # label, data = cutmib(label, data, label_mix, data_mix, scale_factor, prob, alpha)

    if np.random.rand(1) > prob:
        label, data = cutmib_cwt(
            label, data, label_mix, data_mix, scale_factor, prob, alpha
        )

    # normalization
    # label = torch.from_numpy(label.astype(np.float32) / 255.0)
    # data = torch.from_numpy(data.astype(np.float32) / 255.0)

    # data: U * patchsize//scale, V * patchsize//scale
    # label: U * patchsize, V * patchsize
    label = einops.rearrange(label, "(u v) H W -> (u H) (v W)", u=angRes, v=angRes)
    data = einops.rearrange(data, "(u v) H W -> (u H) (v W)", u=angRes, v=angRes)

    return data, label


def augmentation_cutblur(data, label, angRes=5, scale_factor=4, prob=1.0, alpha=0.7):
    if random.random() < 0.5:  # flip along W-V direction
        data = data[:, ::-1]
        label = label[:, ::-1]
    if random.random() < 0.5:  # flip along W-V direction
        data = data[::-1, :]
        label = label[::-1, :]
    if random.random() < 0.5:  # transpose between U-V and H-W
        data = data.transpose(1, 0)
        label = label.transpose(1, 0)
    # data: U * patchsize//scale, V * patchsize//scale
    # label: U * patchsize, V * patchsize
    lr_H, lr_W = data.shape
    hr_H, hr_W = label.shape
    # hr_H = lr_H * scale_factor
    # hr_W = lr_W * scale_factor
    patchsize = hr_H // angRes

    label = einops.rearrange(label, "(u H) (v W) -> (u v) H W", u=angRes, v=angRes)
    data = einops.rearrange(data, "(u H) (v W) -> (u v) H W", u=angRes, v=angRes)

    # alpha = 0.7, prob = 1.0
    label, data = cutblur(label, data, scale_factor, prob, alpha)

    # normalization
    # label = torch.from_numpy(label.astype(np.float32) / 255.0)
    # data = torch.from_numpy(data.astype(np.float32) / 255.0)

    # data: U * patchsize//scale, V * patchsize//scale
    # label: U * patchsize, V * patchsize
    label = einops.rearrange(label, "(u v) H W -> (u H) (v W)", u=angRes, v=angRes)
    data = einops.rearrange(data, "(u v) H W -> (u H) (v W)", u=angRes, v=angRes)

    return data, label


def augmentation_cutblur_cwt(
    data, label, angRes=5, scale_factor=4, prob=1.0, alpha=0.7
):
    if random.random() < 0.5:  # flip along W-V direction
        data = data[:, ::-1]
        label = label[:, ::-1]
    if random.random() < 0.5:  # flip along W-V direction
        data = data[::-1, :]
        label = label[::-1, :]
    if random.random() < 0.5:  # transpose between U-V and H-W
        data = data.transpose(1, 0)
        label = label.transpose(1, 0)
    # data: U * patchsize//scale, V * patchsize//scale
    # label: U * patchsize, V * patchsize
    lr_H, lr_W = data.shape
    hr_H, hr_W = label.shape
    # hr_H = lr_H * scale_factor
    # hr_W = lr_W * scale_factor
    patchsize = hr_H // angRes

    label = einops.rearrange(label, "(u H) (v W) -> (u v) H W", u=angRes, v=angRes)
    data = einops.rearrange(data, "(u H) (v W) -> (u v) H W", u=angRes, v=angRes)

    # alpha = 0.7, prob = 1.0
    label, data = cutblur_cwt(label, data, scale_factor, prob, alpha)

    # normalization
    # label = torch.from_numpy(label.astype(np.float32) / 255.0)
    # data = torch.from_numpy(data.astype(np.float32) / 255.0)

    # data: U * patchsize//scale, V * patchsize//scale
    # label: U * patchsize, V * patchsize
    label = einops.rearrange(label, "(u v) H W -> (u H) (v W)", u=angRes, v=angRes)
    data = einops.rearrange(data, "(u v) H W -> (u H) (v W)", u=angRes, v=angRes)

    return data, label


def augmentation_cutblend_cwt(
    data,
    label,
    angRes=5,
    scale_factor=4,
    mixup_prob=1.0,
    mixup_alpha=1.2,
    cutmix_prob=1.0,
    cutmix_alpha=0.7,
):
    if random.random() < 0.5:  # flip along W-V direction
        data = data[:, ::-1]
        label = label[:, ::-1]
    if random.random() < 0.5:  # flip along W-V direction
        data = data[::-1, :]
        label = label[::-1, :]
    if random.random() < 0.5:  # transpose between U-V and H-W
        data = data.transpose(1, 0)
        label = label.transpose(1, 0)
    # data: U * patchsize//scale, V * patchsize//scale
    # label: U * patchsize, V * patchsize
    lr_H, lr_W = data.shape
    hr_H, hr_W = label.shape
    # hr_H = lr_H * scale_factor
    # hr_W = lr_W * scale_factor
    patchsize = hr_H // angRes

    label = einops.rearrange(label, "(u H) (v W) -> (u v) H W", u=angRes, v=angRes)
    data = einops.rearrange(data, "(u H) (v W) -> (u v) H W", u=angRes, v=angRes)

    label_mix = label.mean(axis=0)
    data_mix = data.mean(axis=0)

    # alpha = 0.1, prob = 0.9
    label, data = cutblend_cwt(
        label,
        data,
        label_mix,
        data_mix,
        scale_factor,
        mixup_prob,
        mixup_alpha,
        cutmix_prob,
        cutmix_alpha,
    )

    # normalization
    # label = torch.from_numpy(label.astype(np.float32) / 255.0)
    # data = torch.from_numpy(data.astype(np.float32) / 255.0)

    # data: U * patchsize//scale, V * patchsize//scale
    # label: U * patchsize, V * patchsize
    label = einops.rearrange(label, "(u v) H W -> (u H) (v W)", u=angRes, v=angRes)
    data = einops.rearrange(data, "(u v) H W -> (u H) (v W)", u=angRes, v=angRes)

    return data, label


def cutmib(im1, im2, im1_mix, im2_mix, scale, prob=1.0, alpha=0.7):
    cut_ratio = np.random.randn() * 0.01 + alpha
    an, h_lr, w_lr = im2.shape
    ch_lr, cw_lr = int(h_lr * cut_ratio), int(w_lr * cut_ratio)
    ch_hr, cw_hr = ch_lr * scale, cw_lr * scale
    cy_lr = np.random.randint(0, h_lr - ch_lr + 1)
    cx_lr = np.random.randint(0, w_lr - cw_lr + 1)
    cy_hr, cx_hr = cy_lr * scale, cx_lr * scale

    # if True:
    if np.random.random() < prob:
        if np.random.random() > 0.5:
            # print("1111")
            for i in range(an):
                im2[i, cy_lr : cy_lr + ch_lr, cx_lr : cx_lr + cw_lr] = imresize(
                    im1_mix[..., cy_hr : cy_hr + ch_hr, cx_hr : cx_hr + cw_hr],
                    scalar_scale=1 / scale,
                )
                # im2[i, cy_lr : cy_lr + ch_lr, cx_lr : cx_lr + cw_lr] = 1
        else:
            # print("2222")
            im2_aug = im2.copy()
            for i in range(an):
                im2_aug[i] = imresize(im1[i], scalar_scale=1 / scale)
                im2_aug[i, cy_lr : cy_lr + ch_lr, cx_lr : cx_lr + cw_lr] = im2_mix[
                    ..., cy_lr : cy_lr + ch_lr, cx_lr : cx_lr + cw_lr
                ]
                # im2_aug[i, cy_lr : cy_lr + ch_lr, cx_lr : cx_lr + cw_lr] = 1
            im2 = im2_aug

        return im1, im2
    else:
        return im1, im2


def cutmib_cwt(im1, im2, im1_mix, im2_mix, scale, prob=1.0, alpha=0.7):
    cut_ratio = np.random.randn() * 0.01 + alpha
    an, h_lr, w_lr = im2.shape
    ch_lr, cw_lr = int(h_lr * cut_ratio), int(w_lr * cut_ratio)
    ch_hr, cw_hr = ch_lr * scale, cw_lr * scale
    cy_lr = np.random.randint(0, h_lr - ch_lr + 1)
    cx_lr = np.random.randint(0, w_lr - cw_lr + 1)
    cy_hr, cx_hr = cy_lr * scale, cx_lr * scale
    # apply CutMIB to HR or LR
    # if True:
    if np.random.random() < prob:
        if np.random.random() > 0.5:
            # LR -> HR
            im2_aug = im2.copy()
            for i in range(an):
                im2_mix_up = imresize(im2_mix, scalar_scale=scale)
                im1_tmp = im1[i].copy()
                im1_tmp[cy_hr : cy_hr + ch_hr, cx_hr : cx_hr + cw_hr] = im2_mix_up[
                    cy_hr : cy_hr + ch_hr, cx_hr : cx_hr + cw_hr
                ]
                im2_aug[i] = imresize(im1_tmp, scalar_scale=1 / scale)
            im2 = im2_aug
        else:
            # HR -> LR
            im2_aug = im2.copy()
            for i in range(an):
                im2_aug_up = imresize(im2[i], scalar_scale=scale)
                im2_aug_up[cy_hr : cy_hr + ch_hr, cx_hr : cx_hr + cw_hr] = im1_mix[
                    cy_hr : cy_hr + ch_hr, cx_hr : cx_hr + cw_hr
                ]
                im2_aug[i] = imresize(im2_aug_up, scalar_scale=1 / scale)
            im2 = im2_aug

        return im1, im2
    else:
        return im1, im2


def cutblend_cwt(
    im1,
    im2,
    im1_mix,
    im2_mix,
    scale,
    mixup_prob=1.0,
    mixup_alpha=1.2,
    cutmix_prob=1.0,
    cutmix_alpha=0.7,
):
    cut_ratio = np.random.randn() * 0.01 + cutmix_alpha
    an, h_lr, w_lr = im2.shape
    ch_lr, cw_lr = int(h_lr * cut_ratio), int(w_lr * cut_ratio)
    ch_hr, cw_hr = ch_lr * scale, cw_lr * scale
    cy_lr = np.random.randint(0, h_lr - ch_lr + 1)
    cx_lr = np.random.randint(0, w_lr - cw_lr + 1)
    cy_hr, cx_hr = cy_lr * scale, cx_lr * scale
    # apply CutBlend to HR or LR
    # if True:

    if mixup_alpha <= 0 or np.random.rand(1) >= mixup_prob:
        v = 1
    else:
        v = np.random.beta(mixup_alpha, mixup_alpha)

    if np.random.random() < cutmix_prob:
        if np.random.random() > 0.5:
            # LR -> HR
            im2_aug = im2.copy()
            for i in range(an):
                im2_mix_up = imresize(im2_mix, scalar_scale=scale)
                im1_tmp = im1[i].copy()
                im2_mix_up_blend = v * im2_mix_up + (1 - v) * im1_tmp
                im1_tmp[
                    cy_hr : cy_hr + ch_hr, cx_hr : cx_hr + cw_hr
                ] = im2_mix_up_blend[cy_hr : cy_hr + ch_hr, cx_hr : cx_hr + cw_hr]
                im2_aug[i] = imresize(im1_tmp, scalar_scale=1 / scale)
            im2 = im2_aug
        else:
            # HR -> LR
            im2_aug = im2.copy()
            for i in range(an):
                im2_aug_up = imresize(im2[i], scalar_scale=scale)
                im1_mix_blend = v * im1_mix + (1 - v) * im2_aug_up
                im2_aug_up[
                    cy_hr : cy_hr + ch_hr, cx_hr : cx_hr + cw_hr
                ] = im1_mix_blend[cy_hr : cy_hr + ch_hr, cx_hr : cx_hr + cw_hr]
                im2_aug[i] = imresize(im2_aug_up, scalar_scale=1 / scale)
            im2 = im2_aug

        return im1, im2
    else:
        return im1, im2


def cutblur(im1, im2, scale, prob=1.0, alpha=0.7):
    cut_ratio = np.random.randn() * 0.01 + alpha
    an, h_lr, w_lr = im2.shape
    ch_lr, cw_lr = int(h_lr * cut_ratio), int(w_lr * cut_ratio)
    ch_hr, cw_hr = ch_lr * scale, cw_lr * scale
    cy_lr = np.random.randint(0, h_lr - ch_lr + 1)
    cx_lr = np.random.randint(0, w_lr - cw_lr + 1)
    cy_hr, cx_hr = cy_lr * scale, cx_lr * scale

    if np.random.random() < prob:
        if np.random.random() > 0.5:
            for i in range(an):
                im2[i, cy_lr : cy_lr + ch_lr, cx_lr : cx_lr + cw_lr] = imresize(
                    im1[i, cy_hr : cy_hr + ch_hr, cx_hr : cx_hr + cw_hr],
                    scalar_scale=1 / scale,
                )
        else:
            im2_aug = im2.copy()
            for i in range(an):
                im2_aug[i] = imresize(im1[i], scalar_scale=1 / scale)
                im2_aug[i, cy_lr : cy_lr + ch_lr, cx_lr : cx_lr + cw_lr] = im2[
                    i, cy_lr : cy_lr + ch_lr, cx_lr : cx_lr + cw_lr
                ]
            im2 = im2_aug

        return im1, im2
    else:
        return im1, im2


def cutblur_cwt(im1, im2, scale, prob=1.0, alpha=0.7):
    cut_ratio = np.random.randn() * 0.01 + alpha
    an, h_lr, w_lr = im2.shape
    ch_lr, cw_lr = int(h_lr * cut_ratio), int(w_lr * cut_ratio)
    ch_hr, cw_hr = ch_lr * scale, cw_lr * scale
    cy_lr = np.random.randint(0, h_lr - ch_lr + 1)
    cx_lr = np.random.randint(0, w_lr - cw_lr + 1)
    cy_hr, cx_hr = cy_lr * scale, cx_lr * scale
    # apply CutBlur to  to HR or LR
    if np.random.random() < prob:
        if np.random.random() > 0.5:
            # LR to HR
            im2_aug = im2.copy()
            for i in range(an):
                im2_up = imresize(im2[i], scalar_scale=scale)
                im1_tmp = im1[i].copy()
                im1_tmp[cy_hr : cy_hr + ch_hr, cx_hr : cx_hr + cw_hr] = im2_up[
                    cy_hr : cy_hr + ch_hr, cx_hr : cx_hr + cw_hr
                ]
                im2_aug[i] = imresize(im1_tmp, scalar_scale=1 / scale)
            im2 = im2_aug
        else:
            # HR to LR
            im2_aug = im2.copy()
            for i in range(an):
                im2_up = imresize(im2[i], scalar_scale=scale)
                im2_up[cy_hr : cy_hr + ch_hr, cx_hr : cx_hr + cw_hr] = im1[
                    i, cy_hr : cy_hr + ch_hr, cx_hr : cx_hr + cw_hr
                ]
                im2_aug[i] = imresize(im2_up, scalar_scale=1 / scale)
            im2 = im2_aug

        return im1, im2
    else:
        return im1, im2


# if __name__ == "__main__":
#     # dataset_dir = r'data_for_training/SR_5x5_4x'
#     file_name = r"/home/rookie/cwt/BasicLFSR/data_for_training/SR_5x5_4x/EPFL/000001.h5"
#     with h5py.File(file_name, "r") as hf:
#         Lr_SAI_y = np.array(hf.get("Lr_SAI_y"))  # Lr_SAI_y
#         Hr_SAI_y = np.array(hf.get("Hr_SAI_y"))  # Hr_SAI_y
#         Lr_SAI_y, Hr_SAI_y = augmentation(Lr_SAI_y, Hr_SAI_y, 5, 4)

#         if len(Lr_SAI_y.shape) == 2:
#             H, W = Lr_SAI_y.shape
#             Lr_SAI_y = Lr_SAI_y.reshape(H, W, 1)
#         cv2.imwrite("filename.png", Lr_SAI_y * 255)
#         if len(Hr_SAI_y.shape) == 2:
#             H, W = Hr_SAI_y.shape
#             Hr_SAI_y = Hr_SAI_y.reshape(H, W, 1)
#         cv2.imwrite("filename2.png", Hr_SAI_y * 255)

#     Lr_SAI_y = ToTensor()(Lr_SAI_y.copy())
#     Hr_SAI_y = ToTensor()(Hr_SAI_y.copy())

# Lr_angRes_in = 5
# Lr_angRes_out = 5

# return Lr_SAI_y, Hr_SAI_y, [Lr_angRes_in, Lr_angRes_out]
