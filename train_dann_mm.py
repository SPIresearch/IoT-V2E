import argparse
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
import torch.nn as nn
from dataset.dataloader_manger import get_loader_muda_xiao
from model.model_xiao_zigai_finetune import Xiao_Fusion_mm
from utils.metric.get_mertic import get_metrics
from utils.loss import calc_class_weight, weighted_CrossEntropyLoss
from trainer.mmuda_trainer import MMUDA_trainer
from utils.logger import Logger as Log
from utils.random_seed import set_random_seed
from model.discriminator import DomainDiscriminator
from model.dann import DomainAdversarialLoss

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default='./',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='prostate')
    parser.add_argument("--name", type=str, default='mm_dann')
    parser.add_argument("--modality", type=str, default='mydata')
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--world_size", type=int, default=4)
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--epoch", type=int, default=3,
                        help="epoch number (default: 100)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help='batch size (default: 16)')
    parser.add_argument("--random_seed", type=int, default=2020,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=5,
                        help="print interval of loss (default: 50)")
    parser.add_argument("--val_interval", type=int, default=5,
                        help="epoch interval for eval (default: 10)")

    return parser

def main():
    args = get_argparser().parse_args()
    args.lr = args.lr * args.world_size

    cudnn.enabled = True
    cudnn.benchmark = True
    dist.init_process_group(backend="nccl")
    args.rank = dist.get_rank()
    args.world_size = 4
    torch.cuda.set_device(args.rank)
    args.device = torch.device("cuda", args.rank)
    print(f"[init] == local rank: {args.rank}  world_size'{args.world_size}")
    # args.device = torch.device("cuda:0")
    set_random_seed(args.random_seed)

    Log.init(logfile_level="info",
             stdout_level="info",
             log_file="guorongxiao/EEG_Video_Fusion/log/mm_dann_eeg.log",
             log_format="%(asctime)s %(levelname)-7s %(message)s",
             rewrite= True)


    src_loader, tgt_loader, test_loader = get_loader_muda_xiao(args)
    # model = MUNet3D(1,1).to(args.device)
    model = Xiao_Fusion_mm(n_classes=5, video_input_channels=3, EEG_lastfeatures=512, Video_lastfeatures=512).to(args.device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[args.rank], output_device=args.rank)#, broadcast_buffers=False, find_unused_parameters=True)
    discriminator = DomainDiscriminator(in_feature=512, hidden_size=64)
    daloss = DomainAdversarialLoss(discriminator).to(args.device)
    daloss = DDP(daloss, device_ids=[args.rank], output_device=args.rank)#, broadcast_buffers=False, find_unused_parameters=False)

    metric = get_metrics()

    data_count = [60026, 5948, 40791, 20304, 12293]  # yong shu ju de count
    weights_for_each_class = calc_class_weight(data_count)
    loss_fn = weighted_CrossEntropyLoss

    optimizer = torch.optim.Adam(list(model.parameters())+list(daloss.parameters()), lr=7e-4, weight_decay=1e-5) #torch.optim.SGD(self.model.parameters(), lr=args.lr, weight_decay=0.0001 ,momentum=0.9) #torch.optim.Adam(self.network.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    trainer = MMUDA_trainer(model, daloss, optimizer, loss_fn, metric, src_loader, tgt_loader, test_loader, args, weights_for_each_class)
    if args.rank == 0:
        Log.info("start train")
    trainer.train()
    trainer.test()

main()

