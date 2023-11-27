import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="SR", help="SR, RE")

# LF_SR
parser.add_argument("--angRes", type=int, default=5, help="angular resolution")
parser.add_argument("--scale_factor", type=int, default=4, help="4, 2")

parser.add_argument("--model_name", type=str, default="LFT", help="model name")
parser.add_argument(
    "--use_pre_ckpt", type=bool, default=False, help="use pre model ckpt"
)
parser.add_argument(
    "--path_pre_pth", type=str, default="./pth/", help="path for pre model ckpt"
)
parser.add_argument(
    "--data_name",
    type=str,
    default="ALL",
    help="EPFL, HCI_new, HCI_old, INRIA_Lytro, Stanford_Gantry, ALL(of Five Datasets)",
)
parser.add_argument("--path_for_train", type=str, default="./data_for_training/")
parser.add_argument("--path_for_test", type=str, default="./data_for_test/")
parser.add_argument("--path_log", type=str, default="./log/")

parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--lr", type=float, default=2e-4, help="initial learning rate")
parser.add_argument(
    "--decay_rate", type=float, default=0, help="weight decay [default: 1e-4]"
)
parser.add_argument(
    "--n_steps", type=int, default=15, help="number of epochs to update learning rate"
)
parser.add_argument("--gamma", type=float, default=0.5, help="gamma")
parser.add_argument("--epoch", default=51, type=int, help="Epoch to run [default: 50]")

parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--gpus", type=int, default=1)
parser.add_argument("--patch_size_for_test", type=int, default=32)
parser.add_argument("--stride_for_test", type=int, default=16)
parser.add_argument("--minibatch_for_test", type=int, default=1)
parser.add_argument(
    "--num_workers", type=int, default=2, help="num workers of the Data Loader"
)
parser.add_argument(
    "--seed", type=int, default=42, metavar="S", help="random seed (default: 0)"
)
parser.add_argument(
    "--local_rank",
    dest="local_rank",
    type=int,
    default=0,
)
# alpha and prob should be carefully selected，
parser.add_argument("--augment", type=str, default="default")
parser.add_argument("--alpha", type=float, default=0.1, help="alpha")
parser.add_argument("--prob", type=float, default=0.9, help="prob")

# SR3 diff
parser.add_argument(
    "-c",
    "--config",
    type=str,
    default="sr_sr3_16_128.json",
    help="JSON file for configuration",
)
parser.add_argument(
    "-p",
    "--phase",
    type=str,
    choices=["train", "val"],
    help="Run either train(training) or val(generation)",
    default="train",
)
parser.add_argument("-gpu", "--gpu_ids", type=str, default=None)
parser.add_argument("-debug", "-d", action="store_true")
parser.add_argument("-enable_wandb", action="store_true")
parser.add_argument("-log_wandb_ckpt", action="store_true")
parser.add_argument("-log_eval", action="store_true")

parser.add_argument("--exp_name", type=str, default="")
parser.add_argument("--reset", action="store_true")
parser.add_argument("--infer", action="store_true")

args = parser.parse_args()


if args.task == "SR":
    args.angRes_in = args.angRes
    args.angRes_out = args.angRes
    # args.patch_size_for_test = 32
    # args.stride_for_test = 16
    # args.minibatch_for_test = 1

del args.angRes

if __name__ == "__main__":
    args
