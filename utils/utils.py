import subprocess
import torch.distributed as dist
import glob
import os
import re
import lpips
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from .matlab_resize import imresize
from einops import rearrange
from utils.hparams import hparams
import torch.nn.functional as F


def reduce_tensors(metrics):
    new_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, torch.Tensor):
            dist.all_reduce(v)
            v = v / dist.get_world_size()
        if type(v) is dict:
            v = reduce_tensors(v)
        new_metrics[k] = v
    return new_metrics


def tensors_to_scalars(tensors):
    if isinstance(tensors, torch.Tensor):
        tensors = tensors.item()
        return tensors
    elif isinstance(tensors, dict):
        new_tensors = {}
        for k, v in tensors.items():
            v = tensors_to_scalars(v)
            new_tensors[k] = v
        return new_tensors
    elif isinstance(tensors, list):
        return [tensors_to_scalars(v) for v in tensors]
    else:
        return tensors


def tensors_to_np(tensors):
    if isinstance(tensors, dict):
        new_np = {}
        for k, v in tensors.items():
            if isinstance(v, torch.Tensor):
                v = v.cpu().numpy()
            if type(v) is dict:
                v = tensors_to_np(v)
            new_np[k] = v
    elif isinstance(tensors, list):
        new_np = []
        for v in tensors:
            if isinstance(v, torch.Tensor):
                v = v.cpu().numpy()
            if type(v) is dict:
                v = tensors_to_np(v)
            new_np.append(v)
    elif isinstance(tensors, torch.Tensor):
        v = tensors
        if isinstance(v, torch.Tensor):
            v = v.cpu().numpy()
        if type(v) is dict:
            v = tensors_to_np(v)
        new_np = v
    else:
        raise Exception(f"tensors_to_np does not support type {type(tensors)}.")
    return new_np


def move_to_cpu(tensors):
    ret = {}
    for k, v in tensors.items():
        if isinstance(v, torch.Tensor):
            v = v.cpu()
        if type(v) is dict:
            v = move_to_cpu(v)
        ret[k] = v
    return ret


def move_to_cuda(batch, gpu_id=0):
    # base case: object can be directly moved using `cuda` or `to`
    if callable(getattr(batch, "cuda", None)):
        return batch.cuda(gpu_id, non_blocking=True)
    elif callable(getattr(batch, "to", None)):
        return batch.to(torch.device("cuda", gpu_id), non_blocking=True)
    elif isinstance(batch, list):
        for i, x in enumerate(batch):
            batch[i] = move_to_cuda(x, gpu_id)
        return batch
    elif isinstance(batch, tuple):
        batch = list(batch)
        for i, x in enumerate(batch):
            batch[i] = move_to_cuda(x, gpu_id)
        return tuple(batch)
    elif isinstance(batch, dict):
        for k, v in batch.items():
            batch[k] = move_to_cuda(v, gpu_id)
        return batch
    return batch


def get_last_checkpoint(work_dir, steps=None):
    checkpoint = None
    last_ckpt_path = None
    ckpt_paths = get_all_ckpts(work_dir, steps)
    if len(ckpt_paths) > 0:
        last_ckpt_path = ckpt_paths[0]
        checkpoint = torch.load(last_ckpt_path, map_location="cpu")
    return checkpoint, last_ckpt_path


def get_all_ckpts(work_dir, steps=None):
    if steps is None:
        ckpt_path_pattern = f"{work_dir}/model_ckpt_steps_*.ckpt"
    else:
        ckpt_path_pattern = f"{work_dir}/model_ckpt_steps_{steps}.ckpt"
    return sorted(
        glob.glob(ckpt_path_pattern),
        key=lambda x: -int(re.findall(".*steps\_(\d+)\.ckpt", x)[0]),
    )


def load_checkpoint(model, optimizer, work_dir):
    checkpoint, _ = get_last_checkpoint(work_dir)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["state_dict"]["model"])
        model.cuda()
        optimizer.load_state_dict(checkpoint["optimizer_states"][0])
        print(f"| load model_name from '{work_dir}'.")
        training_step = checkpoint["global_step"]
        del checkpoint
        torch.cuda.empty_cache()
    else:
        training_step = 0
        model.cuda()
    return training_step


def save_checkpoint(model, optimizer, work_dir, global_step, num_ckpt_keep):
    ckpt_path = f"{work_dir}/model_ckpt_steps_{global_step}.ckpt"
    print(f"Step@{global_step}: saving model to {ckpt_path}")
    checkpoint = {"global_step": global_step}
    optimizer_states = []
    optimizer_states.append(optimizer.state_dict())
    checkpoint["optimizer_states"] = optimizer_states
    checkpoint["state_dict"] = {"model": model.state_dict()}
    torch.save(checkpoint, ckpt_path, _use_new_zipfile_serialization=False)
    for old_ckpt in get_all_ckpts(work_dir)[num_ckpt_keep:]:
        remove_file(old_ckpt)
        print(f"Delete ckpt: {os.path.basename(old_ckpt)}")


def remove_file(*fns):
    for f in fns:
        subprocess.check_call(f'rm -rf "{f}"', shell=True)


# def plot_img(img):
#     img = img.data.cpu().numpy()
#     return np.clip(img, 0, 1)


def plot_img(img):
    img = (img.data.cpu().numpy() + 1) * 0.5  # [-1,1] -> [0,1]
    return img


def load_ckpt(cur_model, ckpt_base_dir, model_name="model", force=True, strict=True):
    if os.path.isfile(ckpt_base_dir):
        base_dir = os.path.dirname(ckpt_base_dir)
        ckpt_path = ckpt_base_dir
        checkpoint = torch.load(ckpt_base_dir, map_location="cpu")
    else:
        base_dir = ckpt_base_dir
        checkpoint, ckpt_path = get_last_checkpoint(ckpt_base_dir)
    if checkpoint is not None:
        state_dict = checkpoint["state_dict"]
        if model_name in state_dict.keys():
            if len([k for k in state_dict.keys() if "." in k]) > 0:
                state_dict = {
                    k[len(model_name) + 1 :]: v
                    for k, v in state_dict.items()
                    if k.startswith(f"{model_name}.")
                }
            else:
                state_dict = state_dict[model_name]
        if not strict:
            cur_model_state_dict = cur_model.state_dict()
            unmatched_keys = []
            for key, param in state_dict.items():
                if key in cur_model_state_dict:
                    new_param = cur_model_state_dict[key]
                    if new_param.shape != param.shape:
                        unmatched_keys.append(key)
                        print("| Unmatched keys: ", key, new_param.shape, param.shape)
            for key in unmatched_keys:
                del state_dict[key]
        cur_model.load_state_dict(state_dict, strict=strict)
        print(f"| load '{model_name}' from '{ckpt_path}'.")
    else:
        e_msg = f"| ckpt not found in {base_dir}."
        if force:
            assert e_msg is False
        else:
            print(e_msg)


class Measure:
    def __init__(self, net="alex"):
        self.model = lpips.LPIPS(net=net)

    def cal_metrics(self, out, label):
        # def measure(self, imgA, imgB, img_lr, sr_scale):
        if len(label.size()) == 4:
            label = rearrange(
                label,
                "b c (a1 h) (a2 w) -> b c a1 h a2 w",
                a1=5,
                a2=5,
            )
            out = rearrange(
                out,
                "b c (a1 h) (a2 w) -> b c a1 h a2 w",
                a1=5,
                a2=5,
            )
            # lr = rearrange(
            #     lr,
            #     "b c (a1 h) (a2 w) -> b c a1 h a2 w",
            #     a1=5,
            #     a2=5,
            # )

        if len(label.size()) == 5:
            label = label.permute((0, 1, 3, 2, 4)).unsqueeze(0)
            out = out.permute((0, 1, 3, 2, 4)).unsqueeze(0)
            # lr = lr.permute((0, 1, 3, 2, 4)).unsqueeze(0)

        B, C, U, h, V, w = label.size()
        label_y = label[:, 0, :, :, :, :].data.cpu()
        out_y = out[:, 0, :, :, :, :].data.cpu()
        # lr_y = lr[:, 0, :, :, :, :].data.cpu()

        PSNR = np.zeros(shape=(B, U, V), dtype="float32")
        SSIM = np.zeros(shape=(B, U, V), dtype="float32")
        for b in range(B):
            for u in range(U):
                for v in range(V):
                    img_gt = label_y[b, u, :, v, :].numpy()
                    img_out = out_y[b, u, :, v, :].numpy()
                    # img_lr = lr_y[b, u, :, v, :].numpy()
                    # img_out_lr = imresize(img_out, 1 / hparams["sr_scale"]).astype(
                    #     np.float32
                    # )
                    PSNR[b, u, v] = psnr(img_gt, img_out)
                    SSIM[b, u, v] = ssim(img_gt, img_out, gaussian_weights=True)
        PSNR_mean = PSNR.sum() / np.sum(PSNR > 0)
        SSIM_mean = SSIM.sum() / np.sum(SSIM > 0)
        return PSNR_mean, SSIM_mean

    def cal_metrics_plus(self, out, label, lr):
        # def measure(self, imgA, imgB, img_lr, sr_scale):
        if len(label.size()) == 4:
            label = rearrange(
                label,
                "b c (a1 h) (a2 w) -> b c a1 h a2 w",
                a1=5,
                a2=5,
            )
            out = rearrange(
                out,
                "b c (a1 h) (a2 w) -> b c a1 h a2 w",
                a1=5,
                a2=5,
            )
            lr = rearrange(
                lr,
                "b c (a1 h) (a2 w) -> b c a1 h a2 w",
                a1=5,
                a2=5,
            )

        if len(label.size()) == 5:
            label = label.permute((0, 1, 3, 2, 4)).unsqueeze(0)
            out = out.permute((0, 1, 3, 2, 4)).unsqueeze(0)
            lr = lr.permute((0, 1, 3, 2, 4)).unsqueeze(0)

        B, C, U, h, V, w = label.size()
        label_y = label[:, 0, :, :, :, :].data.cpu()
        out_y = out[:, 0, :, :, :, :].data.cpu()
        lr_y = lr[:, 0, :, :, :, :].data.cpu()

        PSNR = np.zeros(shape=(B, U, V), dtype="float32")
        SSIM = np.zeros(shape=(B, U, V), dtype="float32")
        LPIPS = np.zeros(shape=(B, U, V), dtype="float32")
        LR_PSNR = np.zeros(shape=(B, U, V), dtype="float32")
        for b in range(B):
            for u in range(U):
                for v in range(V):
                    img_gt = label_y[b, u, :, v, :].numpy()
                    img_out = out_y[b, u, :, v, :].numpy()
                    img_lr = lr_y[b, u, :, v, :].numpy()
                    img_out_lr = imresize(img_out, 1 / hparams["sr_scale"]).astype(
                        np.float32
                    )
                    PSNR[b, u, v] = psnr(img_gt, img_out)
                    SSIM[b, u, v] = ssim(img_gt, img_out, gaussian_weights=True)
                    LPIPS[b, u, v] = self.lpips(img_gt * 255.0, img_out * 255.0)
                    LR_PSNR[b, u, v] = psnr(img_out_lr, img_lr)
        PSNR_mean = PSNR.sum() / np.sum(PSNR > 0)
        SSIM_mean = SSIM.sum() / np.sum(SSIM > 0)
        LPIPS_mean = LPIPS.sum() / np.sum(LPIPS > 0)
        LR_PSNR_mean = LR_PSNR.sum() / np.sum(LR_PSNR > 0)

        return PSNR_mean, SSIM_mean, LPIPS_mean, LR_PSNR_mean

    def measure(self, imgA, imgB, img_lr, sr_scale):
        """

        Args:
            imgA: [C, H, W] uint8 or torch.FloatTensor [0,1]
            imgB: [C, H, W] uint8 or torch.FloatTensor [0,1]
            img_lr: [C, H, W] uint8  or torch.FloatTensor [0,1]
            sr_scale:

        Returns: dict of metrics

        """
        if isinstance(imgA, torch.Tensor):
            imgA = (
                np.round((imgA.cpu().numpy()) * 255.0)
                .clip(min=0, max=255)
                .astype(np.uint8)
            )
            imgB = (
                np.round((imgB.cpu().numpy()) * 255.0)
                .clip(min=0, max=255)
                .astype(np.uint8)
            )
            img_lr = (
                np.round((img_lr.cpu().numpy()) * 255.0)
                .clip(min=0, max=255)
                .astype(np.uint8)
            )

        imgA = imgA.transpose(1, 2, 0)
        imgA_lr = imresize(imgA, 1 / sr_scale)
        imgB = imgB.transpose(1, 2, 0)
        img_lr = img_lr.transpose(1, 2, 0)
        psnr = self.psnr(imgA, imgB)
        ssim = self.ssim(imgA, imgB)
        lpips = self.lpips(imgA, imgB)
        lr_psnr = self.psnr(imgA_lr, img_lr)
        res = {"psnr": psnr, "ssim": ssim, "lpips": lpips, "lr_psnr": lr_psnr}
        return {k: float(v) for k, v in res.items()}

    def lpips(self, imgA, imgB, model=None):
        device = next(self.model.parameters()).device
        tA = t(imgA).to(device)
        tB = t(imgB).to(device)
        dist01 = self.model.forward(tA, tB).item()
        return dist01

    def ssim(self, imgA, imgB):
        score, diff = ssim(imgA, imgB, full=True, multichannel=True, data_range=255)
        return score

    def psnr(self, imgA, imgB):
        return psnr(imgA, imgB, data_range=255)


def t(img):
    def to_4d(img):
        assert len(img.shape) == 3
        img_new = np.expand_dims(img, axis=0)
        assert len(img_new.shape) == 4
        return img_new

    def to_CHW(img):
        if len(img.shape) == 2:
            return np.expand_dims(img, axis=0)
        return np.transpose(img, [2, 0, 1])

    def to_tensor(img):
        return torch.Tensor(img)

    return to_tensor(to_4d(to_CHW(img))) / 127.5 - 1


def ImageExtend(Im, bdr):
    [_, _, h, w] = Im.size()
    Im_lr = torch.flip(Im, dims=[-1])
    Im_ud = torch.flip(Im, dims=[-2])
    Im_diag = torch.flip(Im, dims=[-1, -2])

    Im_up = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_mid = torch.cat((Im_lr, Im, Im_lr), dim=-1)
    Im_down = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_Ext = torch.cat((Im_up, Im_mid, Im_down), dim=-2)
    Im_out = Im_Ext[:, :, h - bdr[0] : 2 * h + bdr[1], w - bdr[2] : 2 * w + bdr[3]]

    return Im_out


def LFdivide(data, angRes, patch_size, stride):
    data = rearrange(data, "(a1 h) (a2 w) -> (a1 a2) 1 h w", a1=angRes, a2=angRes)
    [_, _, h0, w0] = data.size()
    # print(data.size())
    bdr = (patch_size - stride) // 2
    numU = (h0 + bdr * 2 - 1) // stride
    numV = (w0 + bdr * 2 - 1) // stride
    data_pad = ImageExtend(data, [bdr, bdr + stride - 1, bdr, bdr + stride - 1])
    # pad = torch.nn.ReflectionPad2d(padding=(bdr, bdr+stride-1, bdr, bdr+stride-1))
    # data_pad = pad(data)
    subLF = F.unfold(data_pad, kernel_size=patch_size, stride=stride)
    subLF = rearrange(
        subLF,
        "(a1 a2) (h w) (n1 n2) -> n1 n2 (a1 h) (a2 w)",
        a1=angRes,
        a2=angRes,
        h=patch_size,
        w=patch_size,
        n1=numU,
        n2=numV,
    )

    return subLF


def LFintegrate(subLF, angRes, pz, stride, h, w):
    if subLF.dim() == 4:
        subLF = rearrange(
            subLF, "n1 n2 (a1 h) (a2 w) -> n1 n2 a1 a2 h w", a1=angRes, a2=angRes
        )
        pass
    bdr = (pz - stride) // 2
    outLF = subLF[:, :, :, :, bdr : bdr + stride, bdr : bdr + stride]
    outLF = rearrange(outLF, "n1 n2 a1 a2 h w -> a1 a2 (n1 h) (n2 w)")
    outLF = outLF[:, :, 0:h, 0:w]

    return outLF


def rgb2ycbcr(x):
    y = np.zeros(x.shape, dtype="double")
    y[:, :, 0] = 65.481 * x[:, :, 0] + 128.553 * x[:, :, 1] + 24.966 * x[:, :, 2] + 16.0
    y[:, :, 1] = (
        -37.797 * x[:, :, 0] - 74.203 * x[:, :, 1] + 112.000 * x[:, :, 2] + 128.0
    )
    y[:, :, 2] = (
        112.000 * x[:, :, 0] - 93.786 * x[:, :, 1] - 18.214 * x[:, :, 2] + 128.0
    )

    y = y / 255.0
    return y


def ycbcr2rgb(x):
    mat = np.array(
        [
            [65.481, 128.553, 24.966],
            [-37.797, -74.203, 112.0],
            [112.0, -93.786, -18.214],
        ]
    )
    mat_inv = np.linalg.inv(mat)
    offset = np.matmul(mat_inv, np.array([16, 128, 128]))
    mat_inv = mat_inv * 255

    y = np.zeros(x.shape, dtype="double")
    y[:, :, 0] = (
        mat_inv[0, 0] * x[:, :, 0]
        + mat_inv[0, 1] * x[:, :, 1]
        + mat_inv[0, 2] * x[:, :, 2]
        - offset[0]
    )
    y[:, :, 1] = (
        mat_inv[1, 0] * x[:, :, 0]
        + mat_inv[1, 1] * x[:, :, 1]
        + mat_inv[1, 2] * x[:, :, 2]
        - offset[1]
    )
    y[:, :, 2] = (
        mat_inv[2, 0] * x[:, :, 0]
        + mat_inv[2, 1] * x[:, :, 1]
        + mat_inv[2, 2] * x[:, :, 2]
        - offset[2]
    )
    return y
