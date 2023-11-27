import os.path

import torch
from models.diffsr_modules import (
    Unet,
    RRDBNet,
    EPITNet,
    DistgUnet
)
from models.diffusion import GaussianDiffusion
from tasks.trainer import Trainer
from utils.hparams import hparams
from utils.utils import load_ckpt
import torch.nn.functional as F


class SRDiffTrainer(Trainer):
    def build_model(self):
        hidden_size = hparams["hidden_size"]
        dim_mults = hparams["unet_dim_mults"]
        dim_mults = [int(x) for x in dim_mults.split("|")]

        if hparams["use_enc"]:
            if hparams["enc_fn"] == "rrdb":
                enc_fn = RRDBNet(
                    1,
                    1,
                    hparams["enc_num_feat"],
                    hparams["enc_num_block"],
                    hparams["enc_num_feat"] // 2,
                )
            if hparams["enc_fn"] == "epit":
                enc_fn = EPITNet(
                    hparams["enc_num_feat"],
                )
            if hparams["enc_ckpt"] != "" and os.path.exists(hparams["enc_ckpt"]):
                load_ckpt(enc_fn, hparams["enc_ckpt"])
        else:
            enc_fn = None

        if "use_distgunet" in hparams.keys() and hparams["use_distgunet"]:
            denoise_fn = DistgUnet(
                hidden_size,
                out_dim=1,  # 3
                cond_dim=hparams["enc_num_feat"],
                dim_mults=dim_mults,
            )
        else:
            denoise_fn = Unet(
                hidden_size,
                out_dim=1,  # 3
                cond_dim=hparams["enc_num_feat"],
                dim_mults=dim_mults,
            )

        self.model = GaussianDiffusion(
            denoise_fn=denoise_fn,
            enc_net=enc_fn,
            timesteps=hparams["timesteps"],
            loss_type=hparams["loss_type"],
        )
        self.global_step = 0
        return self.model

    def sample_and_test(self, sample):
        ret = {k: 0 for k in self.metric_keys}
        ret["n_samples"] = 0
        img_lr_y, img_hr_y, _, _, _ = sample
        # img_hr_y = sample["img_hr_y"]
        # img_lr_y = sample["img_lr_y"]
        img_lr_y_up = F.interpolate(
            img_lr_y,
            scale_factor=hparams["sr_scale"],
            mode="bicubic",
            align_corners=False,
        )
        img_sr, distg_out = self.model.sample(img_lr_y, img_lr_y_up, img_hr_y.shape)
        img_sr = img_sr.clamp(0, 1)
        s = self.measure.cal_metrics(img_sr, img_hr_y)
        ret["psnr"] = s[0]
        ret["ssim"] = s[1]
        # ret["lpips"] += s["lpips"]
        # ret["lr_psnr"] += s["lr_psnr"]
        ret["n_samples"] += img_sr.shape[0]
        return img_sr, distg_out, ret

    def build_optimizer(self, model):
        params = list(model.named_parameters())
        # if not hparams["fix_enc"]:
        if hparams["fix_enc"]:
            params = [p for p in params if "enc" not in p[0]]
        params = [p[1] for p in params]
        total = sum([param.nelement() for param in params])
        # total = sum([param.nelement() for param in net.parameters()])
        print("   Number of parameters: %.4fM" % (total / 1e6))
        return torch.optim.Adam(params, lr=hparams["lr"])

    def build_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.StepLR(
            optimizer, hparams["decay_steps"], gamma=0.5
        )

    def training_step(self, batch):
        # img_hr_y = batch["img_hr_y"]
        # img_lr_y = batch["img_lr_y"]
        # img_lr_y_up = batch["img_lr_y_up"]
        img_lr_y, img_hr_y, [Lr_angRes_in, Lr_angRes_out] = batch
        img_lr_y_up = F.interpolate(
            img_lr_y,
            scale_factor=hparams["sr_scale"],
            mode="bicubic",
            align_corners=False,
        )
        losses, _, _ = self.model(img_hr_y, img_lr_y, img_lr_y_up)
        total_loss = sum(losses.values())
        return losses, total_loss
