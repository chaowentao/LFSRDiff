import functools
import torch
from torch import nn
import torch.nn.functional as F
from utils.hparams import hparams
from .module_util import make_layer, initialize_weights
from .commons import Mish, SinusoidalPosEmb, RRDB, Residual, Rezero, LinearAttention
from .commons import ResnetBlock, Upsample, Block, Downsample
import math

# from utils.utils import LF_rgb2ycbcr, LF_ycbcr2rgb, LF_interpolate
from einops import rearrange

class EPITNet(nn.Module):
    def __init__(self, channels=64):
        super(EPITNet, self).__init__()
        self.angRes = 5
        self.channels = channels
        self.factor = hparams["sr_scale"]

        #################### Initial Feature Extraction #####################
        self.conv_init0 = nn.Sequential(
            nn.Conv3d(1, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False)
        )
        self.conv_init = nn.Sequential(
            nn.Conv3d(
                channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(
                channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(
                channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        ############# Deep Spatial-Angular Correlation Learning #############
        self.altblock = nn.Sequential(
            AltFilter(self.angRes, channels),
            AltFilter(self.angRes, channels),
            AltFilter(self.angRes, channels),
            AltFilter(self.angRes, channels),
            AltFilter(self.angRes, channels),
        )

        ########################### UP-Sampling #############################
        self.upsampling = nn.Sequential(
            nn.Conv2d(
                channels,
                channels * self.factor**2,
                kernel_size=1,
                padding=0,
                bias=False,
            ),
            nn.PixelShuffle(self.factor),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, 1, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, lr, get_fea=None):
        # Bicubic
        # lr = MacPI2SAI(lr, self.factor)
        # lr = (lr + 1) / 2  # [-1,1]->[0,1]
        lr_upscale = interpolate(
            lr, self.angRes, scale_factor=self.factor, mode="bicubic"
        )
        # Initial Feature Extraction

        lr = rearrange(
            lr, "b c (a1 h) (a2 w) -> b c (a1 a2) h w", a1=self.angRes, a2=self.angRes
        )

        buffer = self.conv_init0(lr)
        buffer = self.conv_init(buffer) + buffer  # [B, C, A^2, h, w]

        # Deep Spatial-Angular Correlation Learning
        buffer = self.altblock(buffer) + buffer

        # UP-Sampling
        buffer = rearrange(
            buffer, "b c (u v) h w -> b c (u h) (v w)", u=self.angRes, v=self.angRes
        )
        buffer_SAI = self.upsampling(buffer)
        out = buffer_SAI + lr_upscale
        # out = SAI2MacPI(out, self.factor)  # SAI2MacPI
        # out = out.clamp(0, 1)
        # out = out * 2 - 1  # [-1,1]
        if get_fea:
            return out, buffer
        else:
            return out

class BasicTrans(nn.Module):
    def __init__(self, channels, spa_dim, num_heads=8, dropout=0.0):
        super(BasicTrans, self).__init__()
        self.linear_in = nn.Linear(channels, spa_dim, bias=False)
        self.norm = nn.LayerNorm(spa_dim)
        self.attention = nn.MultiheadAttention(spa_dim, num_heads, dropout, bias=False)
        nn.init.kaiming_uniform_(self.attention.in_proj_weight, a=math.sqrt(5))
        self.attention.out_proj.bias = None
        self.attention.in_proj_bias = None
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(spa_dim),
            nn.Linear(spa_dim, spa_dim * 2, bias=False),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(spa_dim * 2, spa_dim, bias=False),
            nn.Dropout(dropout),
        )
        self.linear_out = nn.Linear(spa_dim, channels, bias=False)

    def gen_mask(self, h: int, w: int, k_h: int, k_w: int):
        attn_mask = torch.zeros([h, w, h, w])
        k_h_left = k_h // 2
        k_h_right = k_h - k_h_left
        k_w_left = k_w // 2
        k_w_right = k_w - k_w_left
        for i in range(h):
            for j in range(w):
                temp = torch.zeros(h, w)
                temp[
                    max(0, i - k_h_left) : min(h, i + k_h_right),
                    max(0, j - k_w_left) : min(w, j + k_w_right),
                ] = 1
                attn_mask[i, j, :, :] = temp

        attn_mask = rearrange(attn_mask, "a b c d -> (a b) (c d)")
        attn_mask = (
            attn_mask.float()
            .masked_fill(attn_mask == 0, float("-inf"))
            .masked_fill(attn_mask == 1, float(0.0))
        )

        return attn_mask

    def forward(self, buffer):
        [_, _, n, v, w] = buffer.size()
        attn_mask = self.gen_mask(v, w, self.mask_field[0], self.mask_field[1]).to(
            buffer.device
        )

        epi_token = rearrange(buffer, "b c n v w -> (v w) (b n) c")
        epi_token = self.linear_in(epi_token)

        epi_token_norm = self.norm(epi_token)
        epi_token = (
            self.attention(
                query=epi_token_norm,
                key=epi_token_norm,
                value=epi_token,
                attn_mask=attn_mask,
                need_weights=False,
            )[0]
            + epi_token
        )

        epi_token = self.feed_forward(epi_token) + epi_token
        epi_token = self.linear_out(epi_token)
        buffer = rearrange(epi_token, "(v w) (b n) c -> b c n v w", v=v, w=w, n=n)

        return buffer


class AltFilter(nn.Module):
    def __init__(self, angRes, channels):
        super(AltFilter, self).__init__()
        self.angRes = angRes
        self.epi_trans = BasicTrans(channels, channels * 2)
        self.conv = nn.Sequential(
            nn.Conv3d(
                channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(
                channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(
                channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False
            ),
        )

    def forward(self, buffer):
        shortcut = buffer
        [_, _, _, h, w] = buffer.size()
        self.epi_trans.mask_field = [self.angRes * 2, 11]

        # Horizontal
        buffer = rearrange(
            buffer, "b c (u v) h w -> b c (v w) u h", u=self.angRes, v=self.angRes
        )
        buffer = self.epi_trans(buffer)
        buffer = rearrange(
            buffer,
            "b c (v w) u h -> b c (u v) h w",
            u=self.angRes,
            v=self.angRes,
            h=h,
            w=w,
        )
        buffer = self.conv(buffer) + shortcut

        # Vertical
        buffer = rearrange(
            buffer, "b c (u v) h w -> b c (u h) v w", u=self.angRes, v=self.angRes
        )
        buffer = self.epi_trans(buffer)
        buffer = rearrange(
            buffer,
            "b c (u h) v w -> b c (u v) h w",
            u=self.angRes,
            v=self.angRes,
            h=h,
            w=w,
        )
        buffer = self.conv(buffer) + shortcut

        return buffer


def interpolate(x, angRes, scale_factor, mode):
    [B, _, H, W] = x.size()
    h = H // angRes
    w = W // angRes
    x_upscale = x.view(B, 1, angRes, h, angRes, w)
    x_upscale = (
        x_upscale.permute(0, 2, 4, 1, 3, 5).contiguous().view(B * angRes**2, 1, h, w)
    )
    x_upscale = F.interpolate(
        x_upscale, scale_factor=scale_factor, mode=mode, align_corners=False
    )
    x_upscale = x_upscale.view(B, angRes, angRes, 1, h * scale_factor, w * scale_factor)
    x_upscale = (
        x_upscale.permute(0, 3, 1, 4, 2, 5)
        .contiguous()
        .view(B, 1, H * scale_factor, W * scale_factor)
    )  # [B, 1, A*h*S, A*w*S]

    return x_upscale


class get_loss(nn.Module):
    def __init__(self, args):
        super(get_loss, self).__init__()
        self.criterion_Loss = torch.nn.L1Loss()

    def forward(self, SR, HR, degrade_info=None):
        loss = self.criterion_Loss(SR, HR)

        return loss


def weights_init(m):
    pass

class DisentgBlock(nn.Module):
    def __init__(self, angRes, channels, channels_out=None):
        super(DisentgBlock, self).__init__()
        if channels_out is None:
            channels_out = channels
        SpaChannel, AngChannel, EpiChannel = channels, channels // 4, channels // 2

        self.SpaConv = nn.Sequential(
            nn.Conv2d(
                channels,
                SpaChannel,
                kernel_size=3,
                stride=1,
                dilation=int(angRes),
                padding=int(angRes),
                bias=False,
            ),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(
                SpaChannel,
                SpaChannel,
                kernel_size=3,
                stride=1,
                dilation=int(angRes),
                padding=int(angRes),
                bias=False,
            ),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.AngConv = nn.Sequential(
            nn.Conv2d(
                channels,
                AngChannel,
                kernel_size=angRes,
                stride=angRes,
                padding=0,
                bias=False,
            ),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(
                AngChannel,
                angRes * angRes * AngChannel,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.LeakyReLU(0.1, inplace=True),
            nn.PixelShuffle(angRes),
        )
        self.EPIConv = nn.Sequential(
            nn.Conv2d(
                channels,
                EpiChannel,
                kernel_size=[1, angRes * angRes],
                stride=[1, angRes],
                padding=[0, angRes * (angRes - 1) // 2],
                bias=False,
            ),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(
                EpiChannel,
                angRes * EpiChannel,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.LeakyReLU(0.1, inplace=True),
            PixelShuffle1D(angRes),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(
                SpaChannel + AngChannel + 2 * EpiChannel,
                channels_out,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(
                channels_out,
                channels_out,
                kernel_size=3,
                stride=1,
                dilation=int(angRes),
                padding=int(angRes),
                bias=False,
            ),
        )

    def forward(self, x, use_res=True):
        feaSpa = self.SpaConv(x)
        feaAng = self.AngConv(x)
        feaEpiH = self.EPIConv(x)
        feaEpiV = self.EPIConv(x.permute(0, 1, 3, 2).contiguous()).permute(0, 1, 3, 2)
        buffer = torch.cat((feaSpa, feaAng, feaEpiH, feaEpiV), dim=1)
        buffer = self.fuse(buffer)
        if use_res:
            return buffer + x
        else:
            return buffer

class PixelShuffle1D(nn.Module):
    """
    1D pixel shuffler
    Upscales the last dimension (i.e., W) of a tensor by reducing its channel length
    inout: x of size [b, factor*c, h, w]
    output: y of size [b, c, h, w*factor]
    """

    def __init__(self, factor):
        super(PixelShuffle1D, self).__init__()
        self.factor = factor

    def forward(self, x):
        b, fc, h, w = x.shape
        c = fc // self.factor
        x = x.contiguous().view(b, self.factor, c, h, w)
        x = x.permute(0, 2, 3, 4, 1).contiguous()  # b, c, h, w, factor
        y = x.view(b, c, h, w * self.factor)
        return y


def MacPI2SAI(x, angRes):
    out = []
    for i in range(angRes):
        out_h = []
        for j in range(angRes):
            out_h.append(x[:, :, i::angRes, j::angRes])
        out.append(torch.cat(out_h, 3))
    out = torch.cat(out, 2)
    return out


def SAI2MacPI(x, angRes):
    b, c, hu, wv = x.shape
    h, w = hu // angRes, wv // angRes
    tempU = []
    for i in range(h):
        tempV = []
        for j in range(w):
            tempV.append(x[:, :, i::h, j::w])
        tempU.append(torch.cat(tempV, dim=3))
    out = torch.cat(tempU, dim=2)
    return out


class DistgResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=0, groups=8):
        super().__init__()
        angRes = 5
        if time_emb_dim > 0:
            self.mlp = nn.Sequential(Mish(), nn.Linear(time_emb_dim, dim_out))

        self.block1 = DisentgBlock(angRes, dim, dim_out)
        self.block2 = DisentgBlock(angRes, dim_out, dim_out)
        self.res_conv = nn.Conv2d(
            dim,
            dim_out,
            kernel_size=3,
            stride=1,
            dilation=int(angRes),
            padding=int(angRes),
            bias=False,
        )

    def forward(self, x, time_emb=None, cond=None):
        h = self.block1(x, use_res=False)
        if time_emb is not None:
            h += self.mlp(time_emb)[:, :, None, None]
        if cond is not None:
            h += cond
        h = self.block2(h, use_res=False)
        return h + self.res_conv(x)
class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(RRDBNet, self).__init__()
        self.angRes = 5
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        if hparams["sr_scale"] == 4:
            self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        if hparams["sr_scale"] == 8:
            self.upconv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, get_fea=False):
        feas = []
        # x = (x + 1) / 2  # [-1,1]->[0,1]
        fea_first = fea = self.conv_first(x)
        for l in self.RRDB_trunk:
            fea = l(fea)
            feas.append(fea)
        trunk = self.trunk_conv(fea)
        fea = fea_first + trunk
        feas.append(fea)

        fea = self.lrelu(
            self.upconv1(F.interpolate(fea, scale_factor=2, mode="nearest"))
        )
        if hparams["sr_scale"] == 4:
            fea = self.lrelu(
                self.upconv2(F.interpolate(fea, scale_factor=2, mode="nearest"))
            )
        if hparams["sr_scale"] == 8:
            fea = self.lrelu(
                self.upconv3(F.interpolate(fea, scale_factor=2, mode="nearest"))
            )
        fea_hr = self.HRconv(fea)
        out = self.conv_last(self.lrelu(fea_hr))
        # out = rearrange(
        #     out, "(b a1 a2) c h w  -> b c (a1 h) (a2 w)", a1=self.angRes, a2=self.angRes
        # )
        # out = out.clamp(0, 1)
        # out = out * 2 - 1  # [-1,1]
        if get_fea:
            return out, feas
        else:
            return out


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=0, groups=8):
        super().__init__()
        if time_emb_dim > 0:
            self.mlp = nn.Sequential(Mish(), nn.Linear(time_emb_dim, dim_out))

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None, cond=None):
        h = self.block1(x)
        if time_emb is not None:
            h += self.mlp(time_emb)[:, :, None, None]
        if cond is not None:
            h += cond
        h = self.block2(h)
        return h + self.res_conv(x)

class Unet(nn.Module):
    def __init__(self, dim, out_dim=None, dim_mults=(1, 2, 4, 8), cond_dim=32):
        super().__init__()
        self.angRes = 5
        # dims = [3, *map(lambda m: dim * m, dim_mults)]
        dims = [1, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        groups = 0

        if hparams["enc_fn"] == "rrdb":
            self.cond_proj = nn.ConvTranspose2d(
                cond_dim * ((hparams["enc_num_block"] + 1) // 3),
                # cond_dim,
                dim,
                hparams["sr_scale"] * 2,
                hparams["sr_scale"],
                hparams["sr_scale"] // 2,
            )
        else:
            self.cond_proj = nn.ConvTranspose2d(
                # cond_dim * ((hparams["num_block"] + 1) // 3),
                cond_dim,
                dim,
                hparams["sr_scale"] * 2,
                hparams["sr_scale"],
                hparams["sr_scale"] // 2,
            )

        self.time_pos_emb = SinusoidalPosEmb(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), Mish(), nn.Linear(dim * 4, dim)
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_out, time_emb_dim=dim, groups=groups),
                        ResnetBlock(dim_out, dim_out, time_emb_dim=dim, groups=groups),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim, groups=groups)
        if hparams["use_attn"]:  # defalut false
            self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim, groups=groups)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        ResnetBlock(
                            dim_out * 2, dim_in, time_emb_dim=dim, groups=groups
                        ),
                        ResnetBlock(dim_in, dim_in, time_emb_dim=dim, groups=groups),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            Block(dim, dim, groups=groups), nn.Conv2d(dim, out_dim, 1)
        )

        if hparams["res"] and hparams["up_input"]:
            self.up_proj = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(3, dim, 3),
            )
        if hparams["use_wn"]:  # defalut false
            self.apply_weight_norm()
        if hparams["weight_init"]:  # defalut false
            self.apply(initialize_weights)

    def apply_weight_norm(self):
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)
                # print(f"| Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def forward(self, x, time, cond, img_lr_up):
        t = self.time_pos_emb(time)
        t = self.mlp(t)

        h = []
        if hparams["enc_fn"] == "rrdb":
            cond = self.cond_proj(torch.cat(cond[2::3], 1))
        else:
            cond = self.cond_proj(cond)

        for i, (resnet, resnet2, downsample) in enumerate(self.downs):
            x = resnet(x, t)
            x = resnet2(x, t)
            if i == 0:
                x = x + cond
                if hparams["res"] and hparams["up_input"]:
                    x = x + self.up_proj(img_lr_up)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        if hparams["use_attn"]:
            x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = upsample(x)

        return self.final_conv(x)

class DistgUnet(nn.Module):
    def __init__(self, dim, out_dim=None, dim_mults=(1, 2, 4, 8), cond_dim=32):
        super().__init__()
        angRes = 5
        # dims = [3, *map(lambda m: dim * m, dim_mults)]
        # dims = [1, *map(lambda m: dim * m, dim_mults)]
        dims = [*map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        groups = 0

        if hparams["enc_fn"] == "rrdb":
            self.cond_proj = nn.ConvTranspose2d(
                cond_dim * ((hparams["enc_num_block"] + 1) // 3),
                # cond_dim,
                dim,
                hparams["sr_scale"] * 2,
                hparams["sr_scale"],
                hparams["sr_scale"] // 2,
            )
        else:
            self.cond_proj = nn.ConvTranspose2d(
                # cond_dim * ((hparams["num_block"] + 1) // 3),
                cond_dim,
                dim,
                hparams["sr_scale"] * 2,
                hparams["sr_scale"],
                hparams["sr_scale"] // 2,
            )

        self.time_pos_emb = SinusoidalPosEmb(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), Mish(), nn.Linear(dim * 4, dim)
        )
        self.init_conv = nn.Conv2d(
            1,
            dim,
            kernel_size=3,
            stride=1,
            dilation=angRes,
            padding=angRes,
            bias=False,
        )
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        DistgResnetBlock(
                            dim_in, dim_out, time_emb_dim=dim, groups=groups
                        ),
                        DistgResnetBlock(
                            dim_out, dim_out, time_emb_dim=dim, groups=groups
                        ),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = DistgResnetBlock(
            mid_dim, mid_dim, time_emb_dim=dim, groups=groups
        )
        if hparams["use_attn"]:  # defalut false
            self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = DistgResnetBlock(
            mid_dim, mid_dim, time_emb_dim=dim, groups=groups
        )

        # for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        DistgResnetBlock(
                            dim_out * 2, dim_in, time_emb_dim=dim, groups=groups
                        ),
                        DistgResnetBlock(
                            dim_in, dim_in, time_emb_dim=dim, groups=groups
                        ),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        # self.final_conv = nn.Sequential(
        #     Block(dim, dim, groups=groups), nn.Conv2d(dim, out_dim, 1)
        # )

        self.final_conv = nn.Conv2d(
            dim,
            out_dim,
            kernel_size=3,
            stride=1,
            dilation=int(angRes),
            padding=int(angRes),
            bias=False,
        )
        if hparams["res"] and hparams["up_input"]:
            self.up_proj = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(3, dim, 3),
            )
        if hparams["use_wn"]:  # defalut false
            self.apply_weight_norm()
        if hparams["weight_init"]:  # defalut false
            self.apply(initialize_weights)

    def apply_weight_norm(self):
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)
                # print(f"| Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def forward(self, x, time, cond, img_lr_up):
        t = self.time_pos_emb(time)
        t = self.mlp(t)

        h = []
        if hparams["enc_fn"] == "rrdb":
            cond = self.cond_proj(torch.cat(cond[2::3], 1))
        else:
            cond = self.cond_proj(cond)
        x = self.init_conv(x)
        x = x + cond
        if hparams["res"] and hparams["up_input"]:
            x = x + self.up_proj(img_lr_up)
        for i, (resnet, resnet2, downsample) in enumerate(self.downs):
            x = resnet(x, t)
            x = resnet2(x, t)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        if hparams["use_attn"]:
            x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = upsample(x)

        return self.final_conv(x)

    def make_generation_fast_(self):
        def remove_weight_norm(m):
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(remove_weight_norm)
