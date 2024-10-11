import argparse
from almrr import ALMRR
import os


def config():
    parser = argparse.ArgumentParser(description="Settings of ALMRR")

    parser.add_argument('--mode', type=str, choices=["train", "evaluation"],
                        default="train", help="train or evaluation")

    parser.add_argument('--save_path', type=str, default=os.getcwd(), help="saving path")
    parser.add_argument('--img_size', type=int, default=(256, 256), help="image size (hxw)")
    parser.add_argument('--device', type=str, default="cuda:0", help="device for training and testing")
    parser.add_argument('--backbone', type=str, default="resnet", help="backbone net")
    parser.add_argument('--upsample', type=str, default="bilinear", help="operation for resizing cnn map")
    parser.add_argument('--is_agg', type=bool, default=True, help="if to aggregate the features")
    parser.add_argument('--featmap_size', type=int, nargs="+", default=(256, 256), help="feat map size (hxw)")
    parser.add_argument('--kernel_size', type=int, nargs="+", default=(4, 4), help="aggregation kernel (hxw)")
    parser.add_argument('--stride', type=int, nargs="+", default=(4, 4), help="stride of the kernel (hxw)")
    parser.add_argument('--dilation', type=int, default=1, help="dilation of the kernel")
    parser.add_argument('--data_name', type=str, default='', help="data name")
    parser.add_argument('--train_data_path', type=str, default='', help="training data path")
    parser.add_argument('--anomaly_source_path', type=str, default='', help="anomaly data path")
    parser.add_argument('--batch_size', type=int, default=4, help="batch size for training")
    parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
    parser.add_argument('--epochs', type=int, default=2000, help="epochs for training")
    args = parser.parse_args()

    return args

if __name__ == "__main__":

    cfg = config()
    almrr = ALMRR(cfg)

    if cfg.mode == "train":
        almrr.train()
    else:
        almrr.test()




