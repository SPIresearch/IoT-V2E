import torch
import time
from utils.scheduler import PolyLR
from utils.average_meter import AverageMeter
from utils.logger import Logger as Log
import torch.distributed as dist
from torch.nn import functional as F
from utils.metric.get_mertic import f1_test, accuracy_test, f1, accuracy
import numpy as np
from itertools import cycle
class UDA_trainer:
    def __init__(self, model, daloss, optimizer, loss_fn, metric, src_loader, tgt_loader, test_loader, args, weight_class):
        self.device = args.device
        self.args = args
        self.model = model
        self.loss_fn = loss_fn
        self.metric = metric
        self.daloss = daloss
        self.optimizer = optimizer
        self.scheduler = PolyLR(self.optimizer, len(src_loader)*args.epoch)
        self.src_loader = src_loader
        self.tgt_loader = tgt_loader
        self.test_loader = test_loader
        self.weight_class = weight_class

        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.src_losses = AverageMeter()
        self.da_losses = AverageMeter()

        self.val_acc = AverageMeter()
        self.val_mf1 = AverageMeter()
        self.test_acc = AverageMeter()
        self.test_mf1 = AverageMeter()

        self.best_dice = -1

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
                sm1, _, sy, _ = train_batch
                tm1, _, _ = next(cyc_tgt_loader)
                sm1, sy, tm1 = sm1.to(self.device), sy.to(self.device), tm1.to(self.device)
                self.data_time.update(time.time() - start_time)

                f1, score = self.model(m1=sm1)
                ft, pt = self.model(m1=tm1)
                src_loss = self.loss_fn(score, sy, self.weight_class, self.device)
                da_loss = self.daloss(f1, ft)
                self.src_losses.update(src_loss.item())
                self.da_losses.update(da_loss.item())
                loss = src_loss + da_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.batch_time.update(time.time() - start_time)         
                if index % self.args.print_interval == 0 and self.args.rank == 0:
                    Log.info("Epoch %d, Itrs %d/%d, Src_Loss=%8f, Da_Loss=%8f, Lr=%8f, Iter_time=%f" %
                            (epoch, index, total_iters, self.src_losses.avg, self.da_losses.avg, self.scheduler.get_lr()[0], self.batch_time.avg))
                self.batch_time.reset()
                self.data_time.reset()
                self.da_losses.reset()
                self.src_losses.reset()
            
            if epoch % self.args.val_interval == 0:
                self.val(epoch)

            
    def val(self, epoch):
        # self.val_acc.reset()
        # self.val_mf1.reset()
        self.model.eval()
        y_predict = np.array([])
        y_true = np.array([])
        start_time = time.time()
        with torch.no_grad():
            for index, train_batch in enumerate(self.test_loader):
                x, y, _ = train_batch
                x, y = x.to(self.device), y.to(self.device)
                feat, score = self.model(x)

                # 我的传统方法
                pred = (score.max(1, keepdim=True)[1] + 1).cpu().numpy()
                label = y.cpu().numpy()
                y_predict = np.append(y_predict, pred)
                y_true = np.append(y_true, label)

                # # 分布式方法
                # acc = accuracy(score, y)
                # acc = torch.from_numpy(acc)
                # dist.all_reduce(acc/self.args.world_size)
                # mf1 = f1(score, y)
                # mf1 = torch.from_numpy(mf1)
                # dist.all_reduce(mf1/self.args.world_size)
                #
                # self.val_acc.update(acc.item())
                # self.val_mf1.update(mf1.item())

        # 我的传统方法
        this_acc = accuracy_test(y_predict, y_true)
        this_mf1 = f1_test(y_predict, y_true)
        y_predict = np.array([])
        y_true = np.array([])

        eval_time = time.time() - start_time

        # # 分布式方法
        # this_acc = self.val_acc.avg
        # this_mf1 = self.val_mf1.avg

        if self.args.rank == 0:
            Log.info("Validate, acc: %8f, mf1: %8f, Val_time=%f" % (this_acc, this_mf1, eval_time))
            if this_mf1 > self.best_dice:
                self.best_dice = this_mf1
                path = 'guorongxiao/EEG_Video_Fusion/checkpoints/best_%s_of_%s_%s.pth' % (self.args.dataset, self.args.modality, self.args.name)
                torch.save({
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "best_score": this_mf1,
                }, path)
                print("Model saved as %s" % path)

    def test(self):
        path = 'guorongxiao/EEG_Video_Fusion/checkpoints/best_%s_of_%s_%s.pth' % (self.args.dataset, self.args.modality, self.args.name)
        print(path)
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        start_time = time.time()

        y_predict = np.array([])
        y_true = np.array([])

        with torch.no_grad():
            for index, train_batch in enumerate(self.test_loader):
                x, y, idx = train_batch
                x, y = x.to(self.device), y.to(self.device)
                feat, score = self.model(x)
                pred = (score.max(1, keepdim=True)[1] + 1).cpu().numpy()
                label = y.cpu().numpy()
                # if self.args.rank == 0:
                #     print((pred > 0.5).sum())
                # if (pred > 0.5).sum()<1:
                #     continue
                y_predict = np.append(y_predict, pred)
                y_true = np.append(y_true, label)
                # dsc = self.metric["dice"](pred, y)
                # dscs.append(dsc.item())
                # hd95 = self.metric["hd95"](pred, y)
                # hd95s.append(hd95)

        # dscs = np.array(dscs)
        # hd95s = np.array(hd95s)
        # dscs = np.array(dscs)
        acc_final = accuracy_test(y_predict, y_true)
        f1_final = f1_test(y_predict, y_true)
        y_predict = np.array([])
        y_true = np.array([])
        eval_time = time.time() - start_time

        Log.info("Test, acc: %8f, mf1: %8f, Val_time=%f" % (acc_final, f1_final, eval_time))

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
