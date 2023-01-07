import argparse
from ast import arg
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
import torch.nn as nn
from dataset.dataloader_manger import get_test_loader
from model.unet3d import UNet3D
from utils.metric.get_mertic import get_metrics
from utils.loss import bce_dice
from trainer.base_trainer import Base_trainer
from utils.logger import Logger as Log
from utils.random_seed import set_random_seed
def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default='/workspace',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='prostate')
    parser.add_argument("--modality", type=str, default='t2w')
    parser.add_argument("--local_rank", type=int,default=0)
    parser.add_argument("--world_size", type=int,default=4)
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--epoch", type=int, default=100,
                        help="epoch number (default: 100)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help='batch size (default: 16)')
    parser.add_argument("--random_seed", type=int, default=1,
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
    set_random_seed(args.random_seed)

    Log.init(logfile_level="info",
             stdout_level="info",
             log_file="./log/base_prostate_{}.log".format(args.modality),
             log_format="%(asctime)s %(levelname)-7s %(message)s",
             rewrite= True)


    test_loader = get_test_loader(args)
    model = UNet3D(1,1).cuda()

    path = 'checkpoints/best_%s_of_%s.pth' % (args.dataset, args.modality)
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for index, train_batch in enumerate(test_loader):
            x, y, _ = train_batch
            x, y = x.to(self.device), y.to(self.device)
            feat, pred = self.model(x)
            print(x.shape)
            # dsc = self.metric["dice"](pred, y)
            # hd95 = self.metric["hd95"](pred, y)
            # self.hd95.update(hd95)
            # self.dsc.update(dsc)
    eval_time = time.time() - start_time
    Log.info("Test, Dsc%8f, HD95%8f, Val time=%f" % (self.dsc.avg, self.hd95.avg, eval_time))

main()

