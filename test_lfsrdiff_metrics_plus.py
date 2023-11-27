from utils.utils2 import ExcelFile_plus, rgb2ycbcr, ycbcr2rgb, cal_metrics_plus
from utils.utils_datasets import MultiTestSetDataLoader
from collections import OrderedDict
from tqdm import tqdm
import imageio
import torch
import numpy as np
import os
from pathlib import Path
import lpips

# from option_me import args
import torch.nn.functional as F

# import ExcelFile


def main(args):
    """CPU or Cuda"""
    result_dir = Path("checkpoints/diffsr_epit_distgunet_m1111_lfsr4x_c32_bs4/results_200000_4X_42/outputs")
    result_dir.mkdir(exist_ok=True)
    """ DATA TEST LOADING """
    print("\nLoad Test Dataset ...")
    test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(args)
    print("The number of test data is: %d" % length_of_tests)

    """ TEST on every dataset """
    print("\nStart test...")

    """Set alex model for LPIPS"""
    net = "alex"  # "vgg"
    model = lpips.LPIPS(net=net)

    """Create Excel for PSNR/SSIM"""
    excel_file = ExcelFile_plus()

    psnr_testset = []
    ssim_testset = []
    lpips_testset = []
    lr_psnr_testset = []
    for index, test_name in enumerate(test_Names):
        # if index != 1:
        #     continue
        test_loader = test_Loaders[index]
        # if test_name != "Stanford_Gantry":
        #     continue
        (
            psnr_iter_test,
            ssim_iter_test,
            lpips_iter_test,
            lr_psnr_iter_test,
            LF_name,
        ) = test(test_loader, result_dir, model)
        excel_file.write_sheet(
            test_name,
            LF_name,
            psnr_iter_test,
            ssim_iter_test,
            lpips_iter_test,
            lr_psnr_iter_test,
        )

        psnr_epoch_test = float(np.array(psnr_iter_test).mean())
        ssim_epoch_test = float(np.array(ssim_iter_test).mean())
        lpips_epoch_test = float(np.array(lpips_iter_test).mean())
        lr_psnr_epoch_test = float(np.array(lr_psnr_iter_test).mean())
        psnr_testset.append(psnr_epoch_test)
        ssim_testset.append(ssim_epoch_test)
        lpips_testset.append(lpips_epoch_test)
        lr_psnr_testset.append(lr_psnr_epoch_test)
        print(
            "Test on %s, psnr/ssim is %.3f/%.4f, lpips/lr_psnr is %.4f/%.3f"
            % (
                test_name,
                psnr_epoch_test,
                ssim_epoch_test,
                lpips_epoch_test,
                lr_psnr_epoch_test,
            )
        )
        pass

    psnr_mean_test = float(np.array(psnr_testset).mean())
    ssim_mean_test = float(np.array(ssim_testset).mean())
    lpips_mean_test = float(np.array(lpips_testset).mean())
    lr_psnr_mean_test = float(np.array(lr_psnr_testset).mean())
    excel_file.add_sheet(
        "ALL",
        "Average",
        psnr_mean_test,
        ssim_mean_test,
        lpips_mean_test,
        lr_psnr_mean_test,
    )
    print(
        "The mean psnr on testsets is %.3f, mean ssim is %.4f, mean lpips is %.4f, lr_psnr is %.3f"
        % (psnr_mean_test, ssim_mean_test, lpips_mean_test, lr_psnr_mean_test)
    )
    excel_file.xlsx_file.save(str(result_dir) + "/sr_evaluation.xls")

    pass


def test(test_loader, save_dir, model):
    device = torch.device(args.device)
    LF_iter_test = []
    psnr_iter_test = []
    ssim_iter_test = []
    lpips_iter_test = []
    lr_psnr_iter_test = []
    for idx_iter, (Lr_SAI_y, Hr_SAI_y, Sr_SAI_cbcr, data_info, LF_name) in tqdm(
        enumerate(test_loader), total=len(test_loader), ncols=70
    ):
        Lr_angRes_in = 5
        scale_factor = 4
        # # Lr_SAI_y = Lr_SAI_y  # numU, numV, h*angRes, w*angRes
        # Hr_SAI_y = Hr_SAI_y  # numU*h*angRes, numV*w*angRes
        # Sr_SAI_cbcr = Sr_SAI_cbcr

        [Lr_angRes_in, Lr_angRes_out] = data_info
        data_info[0] = Lr_angRes_in[0].item()
        data_info[1] = Lr_angRes_out[0].item()

        # Lr_SAI = imageio.imread(os.path.join(save_dir, LF_name[0] + "[LR].png"))
        # Hr_SAI = imageio.imread(os.path.join(save_dir, LF_name[0] + "[HR].png"))
        Sr_SAI = imageio.imread(os.path.join(save_dir, LF_name[0] + "[SR].png"))
        # Sr_SAI = imageio.imread(os.path.join(save_dir, LF_name[0] + "[EN].png"))
        U = V = data_info[0]
        H = Sr_SAI.shape[0] // U
        W = Sr_SAI.shape[1] // V

        """ macro -> SAI """
        # Hr_SAI_ = np.zeros_like(Hr_SAI)
        # Sr_SAI_ = np.zeros_like(Sr_SAI)
        # for u in range(U):
        #     for v in range(V):
        #         Hr_SAI_[u * H : (u + 1) * H, v * W : (v + 1) * W, :] = Hr_SAI[
        #             u::U, v::V, :
        #         ]
        #         Sr_SAI_[u * H : (u + 1) * H, v * W : (v + 1) * W, :] = Sr_SAI[
        #             u::U, v::V, :
        #         ]
        # imageio.imwrite("test_SAI.png", Hr_SAI_)

        # Hr_SAI_ = Hr_SAI
        Sr_SAI_ = Sr_SAI
        """ rgb -> ycbcr """
        # Hr_SAI_ycbcr = rgb2ycbcr(Hr_SAI_ / 255.0)
        Sr_SAI_ycbcr = rgb2ycbcr(Sr_SAI_ / 255.0)

        # Hr_SAI_y = Hr_SAI_ycbcr[:, :, 0]
        # Hr_SAI_y = Hr_SAI_y / 255.0
        Sr_SAI_y = Sr_SAI_ycbcr[:, :, 0]

        """ Save Converted RGB """
        # Sr_SAI_ycbcr = np.concatenate(
        #     (Sr_SAI_y[:, :, None], Sr_SAI_cbcr.squeeze().permute(1, 2, 0).numpy()),
        #     axis=2,
        # )
        # Sr_SAI_rgb = (ycbcr2rgb(Sr_SAI_ycbcr).clip(0, 1) * 255).astype("uint8")
        # # path = LF_name[0] + ".bmp"
        # # imageio.imwrite(path, Sr_SAI_rgb)

        """ Add dimision to  1, 1, (a1 h) (a2 w) """
        # Hr_SAI_y_ = torch.from_numpy(Hr_SAI_y[None, None, :, :])  # 1, 1, (a1 h) (a2 w)
        Lr_SAI_y_ = Lr_SAI_y
        Hr_SAI_y_ = Hr_SAI_y
        Sr_SAI_y_ = torch.from_numpy(Sr_SAI_y[None, None, :, :])

        """ Calculate the PSNR & SSIM """
        psnr, ssim, lpips, lr_psnr = cal_metrics_plus(
            args, Hr_SAI_y_, Sr_SAI_y_, Lr_SAI_y_, model
        )
        print(
            "Test on %s, psnr/ssim is %.3f/%.4f, lpips/lr_psnr is %.4f/%.3f"
            % (LF_name[0], psnr, ssim, lpips, lr_psnr)
        )
        psnr_iter_test.append(psnr)
        ssim_iter_test.append(ssim)
        lpips_iter_test.append(lpips)
        lr_psnr_iter_test.append(lr_psnr)
        LF_iter_test.append(LF_name[0])

    return (
        psnr_iter_test,
        ssim_iter_test,
        lpips_iter_test,
        lr_psnr_iter_test,
        LF_iter_test,
    )


if __name__ == "__main__":
    from option_me import args

    main(args)
