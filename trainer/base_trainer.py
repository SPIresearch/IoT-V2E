import torch
import time
from utils.scheduler import PolyLR
from utils.average_meter import AverageMeter
from utils.logger import Logger as Log
import torch.distributed as dist
from torch.nn import functional as F
from utils.metric.get_mertic import f1_test, accuracy_test, f1, accuracy
import numpy as np
class Base_trainer:
    def __init__(self, model, loss_fn, metric, train_loader, val_loader, test_loader, args, weight_class):
        self.device = args.device
        self.args = args
        self.model = model
        self.loss_fn = loss_fn
        self.metric = metric
        self.optimizer = torch.optim.Adam(model.parameters(), lr=7e-4, weight_decay=1e-5) #torch.optim.SGD(self.model.parameters(), lr=args.lr, weight_decay=0.0001 ,momentum=0.9) #torch.optim.Adam(self.network.parameters(),lr=args.lr,weight_decay=args.weight_decay)
        self.scheduler = PolyLR(self.optimizer, len(train_loader)*args.epoch)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.weight_class = weight_class

        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.train_losses = AverageMeter()

        self.val_losses = AverageMeter()
        self.val_acc = AverageMeter()
        self.val_mf1 = AverageMeter()

        self.best_dice = -1

    def train(self):
        start_time = time.time()
        total_iters = len(self.train_loader)
        for epoch in range(self.args.epoch):
            self.model.train()
            self.train_loader.sampler.set_epoch(epoch)
            for index, train_batch in enumerate(self.train_loader):
                start_time = time.time()
                x, _, y, _ = train_batch
                x, y = x.to(self.device), y.to(self.device)
                self.data_time.update(time.time() - start_time)
                feat, pred = self.model(x)
                loss = self.loss_fn(pred, y, self.weight_class, self.device)
                # backward_loss = loss
                # display_loss = reduce_tensor(backward_loss) / 4
               
                self.train_losses.update(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.batch_time.update(time.time() - start_time)         
                if index % self.args.print_interval == 0 and self.args.rank == 0:
                    Log.info("Epoch %d, Itrs %d/%d, Loss%8f, Lr=%8f, Iter time=%f" %
                            (epoch, index, total_iters, self.train_losses.avg, self.scheduler.get_lr()[0], self.batch_time.avg))
                self.batch_time.reset()
                self.data_time.reset()
                self.train_losses.reset()
            
            if epoch % self.args.val_interval == 0:
                self.val(epoch)

            
    def val(self, epoch):
        self.val_losses.reset()
        self.model.eval()
        y_predict = np.array([])
        y_true = np.array([])
        start_time = time.time()
        with torch.no_grad():
            for index, train_batch in enumerate(self.val_loader):
                x, _, y, _ = train_batch
                x, y = x.to(self.device), y.to(self.device)
                feat, score = self.model(x)

                loss = self.loss_fn(score, y, self.weight_class, self.device)
                print(loss)
                self.val_losses.update(loss.item())
                # dice = dice_coef(pred, y)
                # dist.all_reduce(dice/self.args.world_size)

                # 我的传统方法
                pred = (score.max(1, keepdim=True)[1] + 1).cpu().numpy()
                label = y.cpu().numpy()
                y_predict = np.append(y_predict, pred)
                y_true = np.append(y_true, label)

        this_loss = self.val_losses.avg
        this_acc = accuracy_test(y_predict, y_true)
        this_mf1 = f1_test(y_predict, y_true)


        y_predict = np.array([])
        y_true = np.array([])

        eval_time = time.time() - start_time

        if self.args.rank == 0:
            Log.info("Validate, loss%8f, acc%8f, mf1%8f, Val time=%f" % (this_loss, this_acc, this_mf1, eval_time))
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
                y_predict = np.append(y_predict, pred)
                y_true = np.append(y_true, label)

                # if self.args.rank == 0:
                #     print((pred > 0.5).sum())
                # if (pred > 0.5).sum()<1:
                #     continue
                # dsc = self.metric["dice"](pred, y)
                # dscs.append(dsc.item())
                # hd95 = self.metric["hd95"](pred, y)
                # hd95s.append(hd95)

        acc_final = accuracy_test(y_predict, y_true)
        f1_final = f1_test(y_predict, y_true)
        y_predict = np.array([])
        y_true = np.array([])

        eval_time = time.time() - start_time
        # dscs = np.array(dscs)
        # hd95s = np.array(hd95s)
        # dscs = np.array(dscs)
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
