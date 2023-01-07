import torch
import time
from utils.scheduler import PolyLR
from utils.average_meter import AverageMeter
from utils.logger import Logger as Log
import torch.distributed as dist
from torch.nn import functional as F
from utils.metric.get_mertic import dice_coef
import numpy as np
from model.unet3d import GD3DUNet
from model.generator import Generator
from utils.loss import ce_dice
from model.jsd import JSD
from model.discriminator import CNNDomainDiscriminator
from model.dann import DomainAdversarialLoss
from torch.nn.parallel import DistributedDataParallel as DDP
from itertools import cycle
from utils.weight import get_weight
from utils.entropy import Entropy

class GD_trainer:
    def __init__(self, metric, src_loader, tgt_loader, test_loader, args):
        self.device = args.device
        self.args = args

        self.src_loader = src_loader
        self.tgt_loader = tgt_loader
        self.test_loader = test_loader

        self.model = GD3DUNet(1,2).to(args.device)
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model = DDP(self.model, device_ids=[args.rank], output_device=args.rank)

        discriminator = CNNDomainDiscriminator()
        self.daloss = DomainAdversarialLoss(discriminator).to(args.device)
        self.daloss = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.daloss)
        self.daloss = DDP(self.daloss, device_ids=[args.rank], output_device=args.rank)

        self.loss_fn = ce_dice
        self.metric = metric
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=7e-4, weight_decay=1e-5) 
        self.scheduler = PolyLR(self.optimizer, len(self.src_loader)*args.epoch)

        self.gen_loss_fn = JSD()
        self.gen = Generator().to(args.device)
        self.gen = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.gen)
        self.gen = DDP(self.gen, device_ids=[args.rank], output_device=args.rank)

        self.gen_optimizer = torch.optim.Adam(self.gen.parameters(), lr=7e-4, weight_decay=1e-5) 
        self.gen_scheduler = PolyLR(self.gen_optimizer, len(self.src_loader)*args.epoch)

       

        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()

        self.src_losses = AverageMeter()
        self.da_losses = AverageMeter()
        self.val_losses = AverageMeter()
        self.gen_losses = AverageMeter()
        self.gd_losses = AverageMeter()
        self.norm = AverageMeter()
        self.best_dice = 0

    def train(self):
        start_time = time.time()
        total_iters = len(self.src_loader)
        for epoch in range(self.args.epoch):
            self.model.train()
            self.src_loader.sampler.set_epoch(epoch)
            self.tgt_loader.sampler.set_epoch(epoch)
            cyc_tgt_loader = iter(cycle(self.tgt_loader))
            for index, train_batch in enumerate(self.src_loader):
                start_time = time.time()
                sm1, sm2, sy, _ = train_batch
                tm1, _, _ = next(cyc_tgt_loader)
                sm1, sm2, sy, tm1 = sm1.to(self.device), sm2.to(self.device), sy.to(self.device), tm1.to(self.device)

                self.data_time.update(time.time() - start_time)
                f1, f2, logit1, logit2 = self.model(sm1, sm2)
                ft, logitt = self.model(m1=tm1, m2=None)

                # train generator 
                self.gen.train()
                with torch.no_grad():
                    p1 = F.softmax(logit1, dim=1)
                    p = F.softmax(logit1 + logit2, dim=1)
                p_h = self.gen(p1)
                gen_loss = self.gen_loss_fn(p_h, p)
                self.gen_optimizer.zero_grad()
                gen_loss.backward()
                self.gen_optimizer.step()
                self.gen_scheduler.step()
                self.gen_losses.update(gen_loss.item())

                # train main model
                self.gen.eval()
                pt = F.softmax(logitt, dim=1)
                with torch.no_grad():
                    delt = self.gen(pt) - pt

                src_loss = self.loss_fn(logit1 + logit2, sy)
                da_loss = self.daloss(f1, ft)
                gd_loss = (get_weight(pt) * (delt*logitt).sum(dim=1)).sum() + Entropy(pt)
                q = torch.norm(logitt,p=2,dim=1).mean()
                all_loss = src_loss + da_loss + gd_loss/q

                

                self.norm.update(q.item())
                self.src_losses.update(src_loss.item())
                self.da_losses.update(da_loss.item())
                self.gd_losses.update(gd_loss.item())
                

                self.optimizer.zero_grad()
                all_loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.batch_time.update(time.time() - start_time)         
                if index % self.args.print_interval == 0 and self.args.rank == 0:
                    Log.info("Epoch %d, Itrs %d/%d, Gen Loss%8f, Src Loss%8f, Da Loss%8f, Gd Loss%8f, Lr=%8f, Iter time=%f, norm=%f" %
                            (epoch, index, total_iters, self.gen_losses.avg, self.src_losses.avg,  self.da_losses.avg, self.gd_losses.avg, self.scheduler.get_lr()[0], self.batch_time.avg, self.norm.avg))
                self.batch_time.reset()
                self.data_time.reset()
                self.src_losses.reset()
                self.da_losses.reset()
                self.gd_losses.reset()
                self.gen_losses.reset()
            
            if epoch % self.args.val_interval == 0:
                self.val(epoch)

            
    def val(self, epoch):
        self.val_losses.reset()
        self.model.eval()
        start_time = time.time()
        with torch.no_grad():
            for index, train_batch in enumerate(self.test_loader):
                x, y, _ = train_batch
                x, y = x.to(self.device), y.to(self.device)
                feat, logit = self.model(x)
                pred = torch.softmax(logit, dim=1)
                dice = dice_coef(pred[:,1,...,], y)
                dist.all_reduce(dice/self.args.world_size) 
                self.val_losses.update(dice.item())
        eval_time = time.time() - start_time
        this_loss = self.val_losses.avg
        if self.args.rank == 0:
            Log.info("Validate, Dice%8f, Val time=%f" % (this_loss, eval_time))
            if this_loss > self.best_dice:
                self.best_dice = this_loss
                path = 'checkpoints/best_%s_of_%s_%s.pth' % (self.args.dataset, self.args.modality, self.args.name)
                torch.save({
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "best_score": this_loss,
                }, path)
                print("Model saved as %s" % path)

    def test(self):
        path = 'checkpoints/best_%s_of_%s_%s.pth' % (self.args.dataset, self.args.modality, self.args.name)
        print(path)
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        start_time = time.time()
        dscs = [] 
        hd95s = []
        with torch.no_grad():
            for index, train_batch in enumerate(self.test_loader):
                x, y, idx = train_batch
                x, y = x.to(self.device), y.to(self.device)
                feat, logit = self.model(x)
                pred = torch.softmax(logit, dim=1)
                if self.args.rank == 0:
                    print((pred > 0.5).sum())
                if (pred > 0.5).sum()<1:
                    continue
                dsc = self.metric["dice"](pred, y)
                dscs.append(dsc.item())
                hd95 = self.metric["hd95"](pred, y)
                hd95s.append(hd95)
        eval_time = time.time() - start_time
        dscs = np.array(dscs)
        hd95s = np.array(hd95s)
        dscs = np.array(dscs)
        Log.info("Test, Dsc: %8f, HD95: %8f, Val time=%f" % (dscs.mean(), hd95s.mean(), eval_time))

        # if self.args.rank == 0:
        #     Log.info("Test, Dsc: %8f, Val time=%f" % (dscs.mean(), eval_time))
        #     if this_dics > self.best_dice:
        #         self.best_dice = this_dics
        #         path = 'checkpoints/best_%s_of_%s_%s.pth' % (self.args.dataset, self.args.modality, self.args.name)
        #         torch.save({
        #         "model_state": self.model.state_dict(),
        #         "optimizer_state": self.optimizer.state_dict(),
        #         "scheduler_state": self.scheduler.state_dict()
        #         }, path)
        #         print("Model saved as %s" % path)

def reduce_tensor(inp):
    with torch.no_grad():
        reduced_inp = inp
        dist.reduce(reduced_inp, dst=0)
    return reduced_inp
