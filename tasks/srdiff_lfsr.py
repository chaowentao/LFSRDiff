import matplotlib

from tasks.srdiff import SRDiffTrainer
from utils.dataset import SRDataSet

matplotlib.use("Agg")

from PIL import Image
from torchvision import transforms
import random
from utils.matlab_resize import imresize
from utils.hparams import hparams
import numpy as np
from utils.sr_utils import ycbcr2rgb, rgb2ycbcr
import imageio


class LFSRDataSet(SRDataSet):
    def __init__(self, prefix="train"):
        super(LFSRDataSet, self).__init__("train" if prefix == "train" else "test")
        self.patch_size = hparams["patch_size"]
        self.patch_size_lr = hparams["patch_size"] // hparams["sr_scale"]
        if prefix == "valid":
            self.len = hparams["eval_batch_size"] * hparams["valid_steps"]
        self.to_tensor_norm_y = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5)),
            ]
        )
        self.to_tensor_norm_cbcr = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5), (0.5, 0.5)),
            ]
        )
        self.data_aug_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20, resample=Image.BICUBIC),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
                ),
            ]
        )

    def __getitem__(self, index):
        item = self._get_item(index)
        hparams = self.hparams
        sr_scale = hparams["sr_scale"]

        img_hr = np.uint8(item["img"])
        img_lr = np.uint8(item["img_lr"])
        img_hr_ycbcr = rgb2ycbcr(img_hr / 255.0)  # [0,1]
        img_hr_y = img_hr_ycbcr[:, :, 0:1]  # get img_hr y channel
        img_lr_ycbcr = rgb2ycbcr(img_lr / 255.0)  # [0,1]
        img_lr_y = img_lr_ycbcr[:, :, 0:1]  # get img_lr y channel
        img_lr_cbcr = img_lr_ycbcr[:, :, 1:3]  # get img_lr y channel

        # TODO: clip for SRFlow
        h, w, c = img_hr_y.shape
        h = h - h % (sr_scale * 2)
        w = w - w % (sr_scale * 2)
        h_l = h // sr_scale
        w_l = w // sr_scale
        img_hr_y = img_hr_y[:h, :w]
        img_lr_y = img_lr_y[:h_l, :w_l]
        # random crop
        if self.prefix == "train":
            if self.data_augmentation and random.random() < 0.5:
                img_hr_y, img_lr_y = self.data_augment(img_hr_y, img_lr_y)
            i = random.randint(0, h - self.patch_size) // sr_scale * sr_scale
            i_lr = i // sr_scale
            j = random.randint(0, w - self.patch_size) // sr_scale * sr_scale
            j_lr = j // sr_scale
            img_hr_y = img_hr_y[i : i + self.patch_size, j : j + self.patch_size]
            img_hr = img_hr[i : i + self.patch_size, j : j + self.patch_size]
            img_lr_y = img_lr_y[
                i_lr : i_lr + self.patch_size_lr, j_lr : j_lr + self.patch_size_lr
            ]
            img_lr = img_lr[
                i_lr : i_lr + self.patch_size_lr, j_lr : j_lr + self.patch_size_lr
            ]
            img_lr_cbcr = img_lr_cbcr[
                i_lr : i_lr + self.patch_size_lr, j_lr : j_lr + self.patch_size_lr
            ]

        img_lr_y_up = imresize(img_lr_y, hparams["sr_scale"])  # np.float [H, W, C]
        img_lr_up = imresize(img_lr, hparams["sr_scale"])  # np.float [H, W, C]
        img_lr_cbcr_up = imresize(
            img_lr_cbcr, hparams["sr_scale"]
        )  # np.float [H, W, C]

        img_hr = self.to_tensor_norm(img_hr).float()
        img_hr_y = self.to_tensor_norm_y(img_hr_y).float()
        img_lr = self.to_tensor_norm(img_lr).float()
        img_lr_y = self.to_tensor_norm_y(img_lr_y).float()
        img_lr_up = self.to_tensor_norm(img_lr_up).float()
        img_lr_y_up = self.to_tensor_norm_y(img_lr_y_up).float()
        img_lr_cbcr_up = self.to_tensor_norm_cbcr(img_lr_cbcr_up).float()

        return {
            "img_hr": img_hr,
            "img_hr_y": img_hr_y,
            "img_lr": img_lr,
            "img_lr_y": img_lr_y,
            "img_lr_up": img_lr_up,
            "img_lr_y_up": img_lr_y_up,
            "img_lr_cbcr_up": img_lr_cbcr_up,
            "item_name": item["item_name"],
            "loc": np.array(item["loc"]),
            "loc_bdr": np.array(item["loc_bdr"]),
        }

    def __len__(self):
        return self.len

    def data_augment(self, img_hr, img_lr):
        sr_scale = self.hparams["sr_scale"]
        img_hr = Image.fromarray(img_hr)
        img_hr = self.data_aug_transforms(img_hr)
        img_hr = np.asarray(img_hr)  # np.uint8 [H, W, C]
        img_lr = imresize(img_hr, 1 / sr_scale)
        return img_hr, img_lr


class SRDiffLFSR(SRDiffTrainer):
    def __init__(self):
        super(SRDiffLFSR, self).__init__()
        self.dataset_cls = LFSRDataSet
