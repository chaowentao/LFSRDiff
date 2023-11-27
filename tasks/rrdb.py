import torch
import torch.nn.functional as F
from models.diffsr_modules import RRDBNet

# from tasks.srdiff_celeb import CelebDataSet
# from tasks.srdiff_df2k import Df2kDataSet
from utils.hparams import hparams
from tasks.trainer import Trainer
from tasks.srdiff_lfsr import LFSRDataSet


class RRDBTask(Trainer):
    def build_model(self):
        hidden_size = hparams["hidden_size"]
        self.model = RRDBNet(
            1, 1, hidden_size, hparams["num_block"], hidden_size // 2
        )  # y channel
        return self.model

    def build_optimizer(self, model):
        return torch.optim.Adam(model.parameters(), lr=hparams["lr"])

    def build_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer, 200000, 0.5)

    def training_step(self, sample):
        img_lr_y, img_hr_y, [Lr_angRes_in, Lr_angRes_out] = sample
        # img_hr_y = sample["img_hr_y"]
        # img_lr_y = sample["img_lr_y"]
        p = self.model(img_lr_y)
        loss = F.l1_loss(p, img_hr_y, reduction="mean")
        return {"l": loss, "lr": self.scheduler.get_last_lr()[0]}, loss


    def sample_and_test(self, sample):
        ret = {k: 0 for k in self.metric_keys}
        ret["n_samples"] = 0
        img_lr_y, img_hr_y, _, _, _ = sample
        # img_hr_y = sample["img_hr_y"]
        # img_lr_y = sample["img_lr_y"]
        img_sr = self.model(img_lr_y)
        img_sr = img_sr.clamp(0, 1)
        for b in range(img_sr.shape[0]):
            s = self.measure.measure(
                img_sr[b], img_hr_y[b], img_lr_y[b], hparams["sr_scale"]
            )
            ret["psnr"] += s["psnr"]
            ret["ssim"] += s["ssim"]
            ret["lpips"] += s["lpips"]
            ret["lr_psnr"] += s["lr_psnr"]
            ret["n_samples"] += 1
        return img_sr, img_sr, ret


class RRDBLFSRTask(RRDBTask):
    def __init__(self):
        super().__init__()
        self.dataset_cls = LFSRDataSet
