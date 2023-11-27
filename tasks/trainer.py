import importlib
import os
import subprocess
import sys

sys.path.append("./")
import imageio
import torch
from PIL import Image
from tqdm import tqdm
from utils.sr_utils import ycbcr2rgb, rgb2ycbcr
from einops import rearrange

# from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from utils.utils_datasets import TrainSetDataLoader, MultiTestSetDataLoader
from utils.utils import LFdivide, LFintegrate
from tensorboardX import SummaryWriter
import random

import numpy as np
from utils.utils import (
    plot_img,
    move_to_cuda,
    load_checkpoint,
    save_checkpoint,
    tensors_to_scalars,
    load_ckpt,
    Measure,
)


from utils.hparams import hparams, set_hparams


class Trainer:
    def __init__(self):
        self.logger = self.build_tensorboard(
            save_dir=hparams["work_dir"], name="tb_logs"
        )
        self.measure = Measure()
        self.dataset_cls = None
        self.metric_keys = ["psnr", "ssim", "lpips", "lr_psnr"]
        self.work_dir = hparams["work_dir"]
        self.first_val = True

    def build_tensorboard(self, save_dir, name, **kwargs):
        log_dir = os.path.join(save_dir, name)
        os.makedirs(log_dir, exist_ok=True)
        return SummaryWriter(log_dir=log_dir, **kwargs)

    def build_train_dataloader(self):
        dataset = self.dataset_cls("train")
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=hparams["batch_size"],
            shuffle=True,
            pin_memory=False,
            num_workers=hparams["num_workers"],
        )

    def build_val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_cls("valid"),
            batch_size=hparams["eval_batch_size"],
            shuffle=False,
            pin_memory=False,
        )

    def build_test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_cls("test"),
            batch_size=hparams["eval_batch_size"],
            shuffle=False,
            pin_memory=False,
        )

    def build_model(self):
        raise NotImplementedError

    def sample_and_test(self, sample):
        raise NotImplementedError

    def build_optimizer(self, model):
        raise NotImplementedError

    def build_scheduler(self, optimizer):
        raise NotImplementedError

    def training_step(self, batch):
        raise NotImplementedError

    def train(self, args):
        seed = args.seed
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        """CPU or Cuda"""
        device = torch.device(args.device)
        if "cuda" in args.device:
            torch.cuda.set_device(device)

        """ DATA Training LOADING """
        print("\nLoad Training Dataset ...")
        train_Dataset = TrainSetDataLoader(args)
        print("The number of training data is: %d" % len(train_Dataset))
        train_loader = torch.utils.data.DataLoader(
            dataset=train_Dataset,
            num_workers=hparams["num_workers"],
            batch_size=hparams["batch_size"],
            shuffle=True,
        )

        """ DATA Validation LOADING """
        print("\nLoad Validation Dataset ...")
        test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(args)
        print("The number of validation data is: %d" % length_of_tests)

        model = self.build_model()
        optimizer = self.build_optimizer(model)
        self.global_step = training_step = load_checkpoint(
            model, optimizer, hparams["work_dir"]
        )
        self.scheduler = scheduler = self.build_scheduler(optimizer)
        scheduler.step(training_step)

        while self.global_step < hparams["max_updates"]:
            print("\nIter %d /%s:" % (training_step, hparams["max_updates"]))
            train_pbar = tqdm(
                train_loader,
                # initial=training_step,
                total=len(train_loader),
                dynamic_ncols=True,
                unit="step",
            )
            loss_iter_train = []
            for batch in train_pbar:
                if (
                    training_step % hparams["val_check_interval"] == 0
                    and training_step > 0
                ):
                    if (training_step + 1) % hparams["val_check_interval"] == 0:
                        # if "enc_ckpt" not in hparams.keys() or hparams["enc_ckpt"] != "pretrain_models/EPIT_5x5_2x_model.pth":
                        with torch.no_grad():
                            model.eval()
                            self.validate(training_step, args)
                    save_checkpoint(
                        model,
                        optimizer,
                        self.work_dir,
                        training_step,
                        hparams["num_ckpt_keep"],
                    )
                model.train()
                batch = move_to_cuda(batch)
                # test_combined_ycbcr(batch)
                # exit(0)
                losses, total_loss = self.training_step(batch)
                optimizer.zero_grad()

                total_loss.backward()
                optimizer.step()
                training_step += 1
                scheduler.step(training_step)
                self.global_step = training_step
                if training_step % 100 == 0:
                    # print(training_step)
                    self.log_metrics(
                        {f"tr/{k}": v for k, v in losses.items()}, training_step
                    )
                    # print({f"tr/{k}": v for k, v in losses.items()}, training_step)
                train_pbar.set_postfix(**tensors_to_scalars(losses))
                loss_iter_train.append(total_loss.data.cpu())
            loss_epoch_train = float(np.array(loss_iter_train).mean())
            print("\nThe mean loss is: %f" % loss_epoch_train)

    def validate(self, training_step, args):
        """DATA Validation LOADING"""
        print("\nLoad Validation Dataset ...")
        test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(args)
        print("The number of validation data is: %d" % length_of_tests)

        # val_dataloader = self.build_val_dataloader()
        # use Stanford as validation set, just for convenience
        pbar = tqdm(enumerate(test_Loaders[4]), total=len(test_Loaders[4]))
        psnr_iter_test = []
        ssim_iter_test = []
        for batch_idx, batch in pbar:
            batch = move_to_cuda(batch)
            img, en_out, ret = self.sample_and_test(batch)
            img_lr_y, img_hr_y, _, _, _ = batch
            if img is not None:
                self.logger.add_image(
                    f"Pred_{batch_idx}", plot_img(img[0]), self.global_step
                )
                if hparams.get("aux_l1_loss"):
                    self.logger.add_image(
                        f"en_out_{batch_idx}", plot_img(en_out[0]), self.global_step
                    )
                if self.global_step <= hparams["val_check_interval"]:
                    self.logger.add_image(
                        f"HR_{batch_idx}", plot_img(img_hr_y[0]), self.global_step
                    )
                    self.logger.add_image(
                        f"LR_{batch_idx}", plot_img(img_lr_y[0]), self.global_step
                    )
            metrics = {}
            metrics.update({k: np.mean(ret[k]) for k in self.metric_keys})
            pbar.set_postfix(**tensors_to_scalars(metrics))
            psnr_iter_test.append(ret["psnr"])
            ssim_iter_test.append(ret["ssim"])
        if hparams["infer"]:
            print("Val results:", metrics)
        else:
            metrics["psnr"] = float(np.array(psnr_iter_test).mean())
            metrics["ssim"] = float(np.array(ssim_iter_test).mean())
            self.log_metrics({f"val/{k}": v for k, v in metrics.items()}, training_step)
            print("Val results:", metrics)

    def test(self, args):
        seed = args.seed
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = self.build_model()
        optimizer = self.build_optimizer(model)
        self.global_step = training_step = load_checkpoint(
            model, optimizer, hparams["work_dir"]
        )
        optimizer = None
        self.results = {k: 0 for k in self.metric_keys}
        self.n_samples = 0
        self.gen_dir = f"{hparams['work_dir']}/results_{self.global_step}_{hparams['gen_dir_name']}_{args.seed}"
        if hparams["test_save_png"]:
            # subprocess.check_call(f"rm -rf {self.gen_dir}", shell=True)
            os.makedirs(f"{self.gen_dir}/outputs", exist_ok=True)

        self.model.sample_tqdm = False
        torch.backends.cudnn.benchmark = False
        if hparams["test_save_png"]:
            if hasattr(self.model.denoise_fn, "make_generation_fast_"):
                self.model.denoise_fn.make_generation_fast_()

        with torch.no_grad():
            model.eval()
            # test_dataloader = self.build_test_dataloader()
            """DATA Validation LOADING"""
            print("\nLoad Validation Dataset ...")
            test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(args)
            print("The number of validation data is: %d" % length_of_tests)

            for index, test_name in enumerate(test_Names):
                test_loader = test_Loaders[index]
                # if index != 2:
                # if index <= 1:
                #     continue
                pbar = tqdm(enumerate(test_loader), total=len(test_loader))
                for batch_idx, batch in pbar:
                    move_to_cuda(batch)
                    gen_dir = self.gen_dir
                    img_lr_y, img_hr_y, img_lr_cbcr_up, _, item_names = batch
                    img_lr_y_up = F.interpolate(
                        img_lr_y,
                        scale_factor=hparams["sr_scale"],
                        mode="bicubic",
                        align_corners=False,
                    )
                    if hparams["save_intermediate"]:
                        item_name = item_names[0]
                        img_y, en_out, imgs_y = self.model.sample(
                            img_lr_y,
                            img_lr_y_up,
                            img_hr_y.shape,
                            save_intermediate=True,
                        )
                        os.makedirs(
                            f"{gen_dir}/intermediate/{item_name}", exist_ok=True
                        )
                        Image.fromarray(
                            self.tensor2img_ycbcr(img_hr_y, img_lr_cbcr_up)[0]
                        ).save(f"{gen_dir}/intermediate/{item_name}/G.png")

                        for i, (m, x_recon) in enumerate(tqdm(imgs_y)):
                            if (
                                i % (hparams["timesteps"] // 20) == 0
                                or i == hparams["timesteps"] - 1
                            ):
                                t_batched = torch.stack(
                                    [torch.tensor(i).to(img_y.device)] * img_y.shape[0]
                                )
                                x_t = self.model.q_sample(
                                    self.model.img2res(img_hr_y, img_lr_y_up),
                                    t=t_batched,
                                )
                                Image.fromarray(
                                    self.tensor2img_ycbcr(x_t, img_lr_cbcr_up)[0]
                                ).save(
                                    f"{gen_dir}/intermediate/{item_name}/noise1_{i:03d}.png"
                                )
                                Image.fromarray(
                                    self.tensor2img_ycbcr(m, img_lr_cbcr_up)[0]
                                ).save(
                                    f"{gen_dir}/intermediate/{item_name}/noise_{i:03d}.png"
                                )
                                Image.fromarray(
                                    self.tensor2img_ycbcr(x_recon, img_lr_cbcr_up)[0]
                                ).save(
                                    f"{gen_dir}/intermediate/{item_name}/{i:03d}.png"
                                )
                        return {}
                    # For particularly large images, a crop test is required.
                    if index != 2:
                        # if False:
                        res = self.sample_and_test(batch)
                        if len(res) == 3:
                            img_sr_y, en_out, ret = res
                        else:
                            img_sr_y, ret = res
                            en_out = img_sr_y
                    else:
                        args.patch_size_for_test = 256 // 2  # 512
                        args.stride_for_test = 128 // 2  # 256
                        args.minibatch_for_test = 1
                        img_lr_y = img_lr_y.squeeze().to(
                            args.device
                        )  # numU, numV, h*angRes, w*angRes
                        img_hr_y = img_hr_y.squeeze().to(args.device)
                        img_lr_cbcr_up = img_lr_cbcr_up

                        """ Crop LFs into Patches """
                        subLFin_lr = LFdivide(
                            img_lr_y,
                            args.angRes_in,
                            args.patch_size_for_test,
                            args.stride_for_test,
                        )
                        subLF_hr = LFdivide(
                            img_hr_y,
                            args.angRes_in,
                            args.patch_size_for_test * args.scale_factor,
                            args.stride_for_test * args.scale_factor,
                        )
                        numU, numV, H, W = subLFin_lr.size()
                        subLFin_lr = rearrange(
                            subLFin_lr, "n1 n2 a1h a2w -> (n1 n2) 1 a1h a2w"
                        )
                        subLF_hr = rearrange(
                            subLF_hr, "n1 n2 a1h a2w -> (n1 n2) 1 a1h a2w"
                        )
                        subLFout_sr = torch.zeros(
                            numU * numV,
                            1,
                            args.angRes_in
                            * args.patch_size_for_test
                            * args.scale_factor,
                            args.angRes_in
                            * args.patch_size_for_test
                            * args.scale_factor,
                        )
                        subLFout_en = torch.zeros(
                            numU * numV,
                            1,
                            args.angRes_in
                            * args.patch_size_for_test
                            * args.scale_factor,
                            args.angRes_in
                            * args.patch_size_for_test
                            * args.scale_factor,
                        )

                        """ SR the Patches """
                        for i in range(0, numU * numV, args.minibatch_for_test):
                            img_lr_y_crop = subLFin_lr[
                                i : min(i + args.minibatch_for_test, numU * numV),
                                :,
                                :,
                                :,
                            ]
                            img_hr_y_crop = subLF_hr[
                                i : min(i + args.minibatch_for_test, numU * numV),
                                :,
                                :,
                                :,
                            ]
                            batch_crop = (
                                img_lr_y_crop,
                                img_hr_y_crop,
                                _,
                                _,
                                item_names,
                            )
                            res_crop = self.sample_and_test(batch_crop)
                            img_sr_y_crop, en_out_crop, ret = res_crop
                            subLFout_sr[
                                i : min(i + args.minibatch_for_test, numU * numV),
                                :,
                                :,
                                :,
                            ] = img_sr_y_crop
                            subLFout_en[
                                i : min(i + args.minibatch_for_test, numU * numV),
                                :,
                                :,
                                :,
                            ] = en_out_crop
                        subLFout_sr = rearrange(
                            subLFout_sr,
                            "(n1 n2) 1 a1h a2w -> n1 n2 a1h a2w",
                            n1=numU,
                            n2=numV,
                        )
                        subLFout_en = rearrange(
                            subLFout_en,
                            "(n1 n2) 1 a1h a2w -> n1 n2 a1h a2w",
                            n1=numU,
                            n2=numV,
                        )

                        """ Restore the Patches to LFs """
                        img_sr_y = LFintegrate(
                            subLFout_sr,
                            args.angRes_out,
                            args.patch_size_for_test * args.scale_factor,
                            args.stride_for_test * args.scale_factor,
                            img_hr_y.size(-2) // args.angRes_out,
                            img_hr_y.size(-1) // args.angRes_out,
                        )
                        img_sr_y = rearrange(img_sr_y, "a1 a2 h w -> 1 1 (a1 h) (a2 w)")
                        en_out = LFintegrate(
                            subLFout_en,
                            args.angRes_out,
                            args.patch_size_for_test * args.scale_factor,
                            args.stride_for_test * args.scale_factor,
                            img_hr_y.size(-2) // args.angRes_out,
                            img_hr_y.size(-1) // args.angRes_out,
                        )
                        en_out = rearrange(en_out, "a1 a2 h w -> 1 1 (a1 h) (a2 w)")
                    if img_sr_y is not None:
                        metrics = list(self.metric_keys)
                        for k in metrics:
                            self.results[k] += ret[k]
                        self.n_samples += ret["n_samples"]
                        print(
                            {
                                k: round(self.results[k] / self.n_samples, 3)
                                for k in metrics
                            },
                            "total:",
                            self.n_samples,
                        )
                        if hparams["test_save_png"] and img_sr_y is not None:
                            img_sr = self.tensor2img_ycbcr(img_sr_y, img_lr_cbcr_up)
                            en_out = self.tensor2img_ycbcr(en_out, img_lr_cbcr_up)
                            img_lr_up = self.tensor2img_ycbcr(
                                img_lr_y_up, img_lr_cbcr_up
                            )
                            for item_name, hr_p, lr_up, en_o in zip(
                                item_names,
                                img_sr,
                                img_lr_up,
                                en_out,
                            ):
                                item_name = os.path.splitext(item_name)[0]
                                hr_p = Image.fromarray(hr_p)
                                lr_up = Image.fromarray(lr_up)
                                en_o = Image.fromarray(en_o)
                                hr_p.save(f"{gen_dir}/outputs/{item_name}[SR].png")
                                # lr_up.save(f"{gen_dir}/outputs/{item_name}[UP].png")
                                # en_o.save(f"{gen_dir}/outputs/{item_name}[EN].png")

    # utils
    def log_metrics(self, metrics, step):
        metrics = self.metrics_to_scalars(metrics)
        logger = self.logger
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            logger.add_scalar(k, v, step)

    def metrics_to_scalars(self, metrics):
        new_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()

            if type(v) is dict:
                v = self.metrics_to_scalars(v)

            new_metrics[k] = v

        return new_metrics

    @staticmethod
    def tensor2img(img):
        img = np.round((img.permute(0, 2, 3, 1).cpu().numpy()) * 255.0)
        img = img.clip(min=0, max=255).astype(np.uint8)
        return img

    @staticmethod
    def tensor2img_ycbcr(img_y, img_cbcr):
        img_y = img_y.permute(0, 2, 3, 1).cpu().numpy()
        img_cbcr = img_cbcr.permute(0, 2, 3, 1).cpu().numpy()
        img = np.concatenate((img_y, img_cbcr), axis=3)
        img = (ycbcr2rgb(img).clip(0, 1) * 255).astype("uint8")
        # img = img.clip(min=0, max=255).astype(np.uint8)
        return img


if __name__ == "__main__":
    from option_me import args

    set_hparams()

    # pass
    pkg = ".".join(hparams["trainer_cls"].split(".")[:-1])
    cls_name = hparams["trainer_cls"].split(".")[-1]
    trainer = getattr(importlib.import_module(pkg), cls_name)()
    if not hparams["infer"]:
        trainer.train(args)
    else:
        trainer.test(args)
