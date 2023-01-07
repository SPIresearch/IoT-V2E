import argparse

import numpy as np
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
import torch.nn as nn
from dataset.dataloader_manger import get_loader_so
from utils.metric.get_mertic import get_metrics
from utils.loss import calc_class_weight, weighted_CrossEntropyLoss
from trainer.mm_trainer import MM_trainer
from utils.logger import Logger as Log
from utils.random_seed import set_random_seed
from models.pcme import PCME
import os
import numpy

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_training_path", type=str, default= r'/media/spi/pic',
                        help="path to train Dataset")
    parser.add_argument("--video_query_path", type=str, default= r'/media/spi/pic',
                        help="path to query Dataset")
    parser.add_argument("--video_database_path", type=str, default=r'/media/spi/pic',
                        help="path to database Dataset")
    parser.add_argument("--annotation_path", type=str, default=r'/home/spi/ucf101_01.json',
                        help="path to database Dataset")
    parser.add_argument("--source_eeg_path", type=str, default= r'/home/spi/FED-RE/DATA/EEG_Clip_C4',
                        help="path to database Dataset")
    parser.add_argument("--dataset", type=str, default='prostate')
    parser.add_argument("--name", type=str, default='mm_base')
    parser.add_argument("--modality", type=str, default='mydata')
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--epoch", type=int, default=200,
                        help="epoch number (default: 100)")
    parser.add_argument("--lr", type=float, default=7e-4,
                        help="learning rate (default: 0.0007)")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="learning rate decay (default: 1e-5)")
    parser.add_argument("--batch_size", type=int, default=256,
                        help='batch size (default: 64)')
    parser.add_argument("--random_seed", type=int, default=2022,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 50)")
    parser.add_argument("--save_interval", type=int, default=5,
                        help="epoch interval for eval (default: 5)")
    parser.add_argument("--eval_interval", type=int, default=5,
                        help="epoch interval for eval (default: 5)")
    parser.add_argument("--save_hashcode_path", type=str, default='/home/spi/FED-RE/Cross_Modal_Retrirval/hashbank/',
                        help="path to save hash code ")
    parser.add_argument("--save_checkpointpath", type=str, default='/home/spi/FED-RE/Cross_Modal_Retrirval/checkpoints/',
                        help="path to save checkpointpath")
    parser.add_argument("--save_pklpath", type=str,
                        default='/home/spi/FED-RE/Cross_Modal_Retrirval/checkpoints/pkl/',
                        help="path to save checkpointpath")
    parser.add_argument("--retrieval_map_k", type=int, default=50,
                        help="MAP(default: 100)")
    parser.add_argument("--label_dim", type=int, default=5,
                        help="classification labels,w,n1,n2,n3,rem")
    return parser

def main():


    args = get_argparser().parse_args()
    args.lr = args.lr * args.world_size

    cudnn.enabled = True
    cudnn.benchmark = True

    #dist.init_process_group(backend="nccl")
    #args.rank = dist.get_rank()
    args.rank = 0

    args.world_size = 1
    torch.cuda.set_device(args.rank)
    args.device = torch.device("cuda", args.rank)
    print(f"[init] == local rank: {args.rank}  world_size'{args.world_size}")
    set_random_seed(args.random_seed)

    Log.init(logfile_level="info",
             stdout_level="info",
             log_file="log/mm_retrieval.log",
             log_format="%(asctime)s %(levelname)-7s %(message)s",
             rewrite= False)  #越来越多



    # a1=np.array(train_loader.dataset.data)
    # np.save("/home/spi/trainset.npy",a1)

    model= PCME().to(args.device)

    all_model_weights="/home/spi/FED-RE/Cross_Modal_Retrirval/checkpoints/eval_save_best.pth"

    r3d_weights=torch.load("/media/spi/sleep/data/results/save_50.pth")

    pre_r3d_dict = r3d_weights["state_dict"]
    for key in list(pre_r3d_dict.keys()):
        pre_r3d_dict["ir_enc.cnn."+key] = pre_r3d_dict.pop(key) #改权重名，加前缀

    attn_weights = torch.load("/home/spi/FED-RE/Cross_Modal_Retrirval/pre_attn.pth")
    for key in list(attn_weights.keys()):
        attn_weights["eeg_enc.eeg_method."+key] = attn_weights.pop(key)

    # missing_keys, unexp_keys= model.load_state_dict(pre_r3d_dict,strict=False)
    # missing_keys, unexp_keys = model.load_state_dict(attn_weights, strict=False)

    model.load_state_dict(torch.load(all_model_weights))
    # for key in list(net_weights.keys()):
    #     if key not in missing_keys :
    #         if key not in list(pre_r3d_dict.keys()):
    #             print(key)


    #model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    #model = DDP(model, device_ids=[args.rank], output_device=args.rank,broadcast_buffers=False,find_unused_parameters=True)
    #model = DDP(model, device_ids=[args.rank], output_device=args.rank)
    net_weights = model.state_dict()
    metric = get_metrics()

    train_loader, query_loader, database_loader = get_loader_so(args)
    data_count = [60026, 5948, 40791, 20304, 12293]  # yong shu ju de count
    weights_for_each_class = calc_class_weight(data_count)
    loss_fn = weighted_CrossEntropyLoss

    print('enter trainer')
    trainer = MM_trainer(model, loss_fn, metric, train_loader, query_loader, database_loader, args, weights_for_each_class)


    trainer.test()



main()