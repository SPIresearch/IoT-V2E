import torch
import time
from utils.scheduler import PolyLR
from utils.average_meter import AverageMeter
from utils.logger import Logger as Log
import torch.distributed as dist
from torch.nn import functional as F
from utils.metric.get_mertic import dice_coef
import numpy as np
class Base_trainer:
    def __init__(self, model, loss_fn, metric, train_loader, val_loader, test_loader, args):
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

        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.train_losses = AverageMeter()
        self.val_losses = AverageMeter()

        self.best_dice = 0

    def train(self):
        start_time = time.time()
        total_iters = len(self.train_loader)
        for epoch in range(self.args.epoch):
            self.model.train()
            self.train_loader.sampler.set_epoch(epoch)
            for index, train_batch in enumerate(self.train_loader):
                start_time = time.time()
                x, y, _ = train_batch
                x, y = x.to(self.device), y.to(self.device)
                self.data_time.update(time.time() - start_time)
                feat, pred = self.model(x)
                loss = self.loss_fn(pred, y)
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
        start_time = time.time()
        with torch.no_grad():
            for index, train_batch in enumerate(self.val_loader):
                x, y, _ = train_batch
                x, y = x.to(self.device), y.to(self.device)
                feat, pred = self.model(x)
                dice = dice_coef(pred, y)
                dist.all_reduce(dice/self.args.world_size) 
                self.val_losses.update(dice.item())
        eval_time = time.time() - start_time
        this_loss = self.val_losses.avg
        if self.args.rank == 0:
            Log.info("Validate, Dice%8f, Val time=%f" % (this_loss, eval_time))
            if this_loss > self.best_dice:
                self.best_dice = this_loss
                path = 'checkpoints/best_%s_of_%s_lr_%f.pth' % (self.args.dataset, self.args.modality, self.args.lr)
                torch.save({
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "best_score": this_loss,
                }, path)
                print("Model saved as %s" % path)

    def test(self):
        if self.args.rank == 0:
            path = 'checkpoints/best_%s_of_%s_lr_%f.pth' % (self.args.dataset, self.args.modality, self.args.lr)
            print(path)
            checkpoint = torch.load(path, map_location=torch.device('cpu'))
            self.model.load_state_dict(checkpoint["model_state"])
            self.model.eval()
            start_time = time.time()
            hd95s = []
            dscs = [] 
            with torch.no_grad():
                for index, train_batch in enumerate(self.test_loader):
                    x, y, idx = train_batch
                    x, y = x.to(self.device), y.to(self.device)
                    feat, pred = self.model(x)
                    # pred = (pred > 0.5)
                    # np.savez('q/{}.npz'.format(idx), pred = pred.cpu().numpy(), y=y.cpu().numpy())
                    hd95 = self.metric["hd95"](pred, y)
                    hd95s.append(hd95)
                    dsc = self.metric["dice"](pred, y)
                    dscs.append(dsc.item())
                    print(dsc)
            eval_time = time.time() - start_time
            hd95s = np.array(hd95s)
            dscs = np.array(dscs)
            Log.info("Test, Dsc: %8f, HD95: %8f, Val time=%f" % (dscs.mean(), hd95s.mean(), eval_time))

def reduce_tensor(inp):
    with torch.no_grad():
        reduced_inp = inp
        dist.reduce(reduced_inp, dst=0)
    return reduced_inp
