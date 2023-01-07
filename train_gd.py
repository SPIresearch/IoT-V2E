import argparse
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
import torch.nn as nn
from dataset.dataloader_manger import get_loader_muda
from model.unet3d import MUNet3D
from utils.metric.get_mertic import get_metrics
from utils.loss import bce_dice
from trainer.gd_trainer import GD_trainer
from utils.logger import Logger as Log
from utils.random_seed import set_random_seed
from model.discriminator import CNNDomainDiscriminator
from model.dann import DomainAdversarialLoss

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default='/workspace',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='prostate')
    parser.add_argument("--name", type=str, default='gd')
    parser.add_argument("--modality", type=str, default='t2w')
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--world_size", type=int,default=4)
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--epoch", type=int, default=100,
                        help="epoch number (default: 100)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help='batch size (default: 16)')
    parser.add_argument("--random_seed", type=int, default=2020,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=5,
                        help="print interval of loss (default: 50)")
    parser.add_argument("--val_interval", type=int, default=10,
                        help="epoch interval for eval (default: 10)")
    return parser

def main():
    args = get_argparser().parse_args()
    args.lr = args.lr * args.world_size

    cudnn.enabled = True
    cudnn.benchmark = True
    dist.init_process_group(backend="nccl")
    args.rank = dist.get_rank()
    args.world_size = 3 
    torch.cuda.set_device(args.rank)
    args.device = torch.device("cuda", args.rank)
    print(f"[init] == local rank: {args.rank}  world_size'{args.world_size}")
    set_random_seed(args.random_seed)

    Log.init(logfile_level="info",
             stdout_level="info",
             log_file="./log/gd_prostate.log",
             log_format="%(asctime)s %(levelname)-7s %(message)s",
             rewrite= True)


    src_loader, tgt_loader, test_loader = get_loader_muda(args)
    metric = get_metrics()
    trainer = GD_trainer( metric, src_loader, tgt_loader, test_loader, args)
    trainer.train()
    trainer.test()

main()

