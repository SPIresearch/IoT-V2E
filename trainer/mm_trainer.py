import torch
import time
from torch.nn.functional import one_hot
from utils.scheduler import PolyLR
from utils.average_meter import AverageMeter
from utils.logger import Logger as Log
import torch.distributed as dist
from torch.nn import functional as F
from utils.metric.get_mertic import f1_test, accuracy_test, f1, accuracy
import numpy as np
from utils.retrieval_utils import write_pickle, calc_map_k, calc_map_rad, pr_curve, p_top_k
import os
from criterions.probemb import MCSoftContrastiveLoss
from criterions.quanization import Cal_quanization_LOSS
from criterions.codebalance import Code_Balance_Loss



class MM_trainer:
    def __init__(self, model, loss_fn, metric, train_loader, val_loader, test_loader, args, weight_class):
        self.since = time.time()
        self.device = args.device
        self.args = args
        self.model = model
        self.loss_fn = loss_fn
        self.metric = metric
        #self.optimizer = torch.optim.Adam(model.parameters(), lr=7e-4, weight_decay=1e-5) #torch.optim.SGD(self.model.parameters(), lr=args.lr, weight_decay=0.0001 ,momentum=0.9) #torch.optim.Adam(self.network.parameters(),lr=args.lr,weight_decay=args.weight_decay)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
        self.scheduler = PolyLR(self.optimizer, len(train_loader)*args.epoch)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.weight_class = weight_class

        self.save_uncert_path= "/home/spi/FED-RE/Cross_Modal_Retrirval/uncert_bank/"
        self.save_hashcode_path = args.save_hashcode_path
        self.retrieval_map_k = args.retrieval_map_k
        self.checkpointpath = args.save_checkpointpath
        self.hash_dim = 64  # 哈希码长度

        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.train_losses = AverageMeter()
        self.train_contra_losses = AverageMeter()
        self.train_q_losses = AverageMeter()
        self.train_cb_losses = AverageMeter()

        self.val_losses = AverageMeter()
        self.val_acc = AverageMeter()
        self.val_mf1 = AverageMeter()

        self.test_losses = AverageMeter()

        self.best_dice = -1

        self.loss_c= MCSoftContrastiveLoss()
        self.B, self.H_eeg, self.H_ir = self.init_hashes()
        self.calc_quantization_loss = Cal_quanization_LOSS()

        self.un_eeg,self.un_ir= self.init_uncert()  #不确定性

        self.beta= 0.001  #量化前边系数
        self.calc_code_balance_loss = Code_Balance_Loss()
        self.gamma=0.01  #gamma hyperparameter (Lcb)

        self.B_val, self.H_eeg_val, self.H_ir_val =self.init_val_hashes()

        self.maps_max = {'i2t': 0., 't2i': 0., 'i2i': 0., 't2t': 0., 'avg': 0.}
        self.maps = {'i2t': [], 't2i': [], 'i2i': [], 't2t': [], 'avg': []}

    def init_hashes(self):
        """
        Initialize hash values (either zeros or random, see below)

        :return: initialized hash values
        """
        dataset_size = len(self.train_loader.dataset)
        B = torch.randn(dataset_size, self.hash_dim).sign().to(self.device)
        H_eeg = torch.zeros(dataset_size, self.hash_dim).sign().to(self.device)
        H_ir = torch.zeros(dataset_size, self.hash_dim).sign().to(self.device)

        return B, H_eeg, H_ir

    def init_uncert(self):
        """
        Initialize hash values (either zeros or random, see below)

        :return: initialized hash values
        """
        dataset_size = len(self.train_loader.dataset)
        un_eeg = torch.zeros(dataset_size, self.hash_dim).to(self.device)
        un_ir = torch.zeros(dataset_size, self.hash_dim).to(self.device)

        return un_eeg, un_ir

    def init_val_hashes(self):
            """
            Initialize hash values (either zeros or random, see below)

            :return: initialized hash values
            """
            dataset_size = len(self.val_loader.dataset)
            B_val = torch.randn(dataset_size, self.hash_dim).sign().to(self.device)
            H_eeg_val = torch.zeros(dataset_size, self.hash_dim).sign().to(self.device)
            H_ir_val = torch.zeros(dataset_size, self.hash_dim).sign().to(self.device)

            return B_val, H_eeg_val, H_ir_val



    def train(self):
        start_time = time.time()
        total_iters = len(self.train_loader)
        for epoch in range(self.args.epoch):
            self.model.train()
            #self.train_loader.sampler.set_epoch(epoch) #分布式
            for index, train_batch in enumerate(self.train_loader):
                start_time = time.time()
                m1, m2, y, index_inall = train_batch
                m1, m2, y = m1.to(self.device), m2.to(self.device), y.to(self.device)
                self.data_time.update(time.time() - start_time)


                output, K_eeg, K_ir = self.model(m1, m2)
                loss_contra, loss_dict = self.loss_c(**output)  # 软对比

                self.un_eeg[index_inall,:]=output["image_logsigma"]
                self.un_ir[index_inall,:]= output["caption_logsigma"]

                self.H_eeg[index_inall, :] = K_eeg
                self.H_ir[index_inall, :] = K_ir

                loss_quant = self.calc_quantization_loss(K_eeg, K_ir, index_inall, self.B) * self.beta

                loss_code_balance = self.calc_code_balance_loss(K_eeg, K_ir) * self.gamma
                loss = loss_contra + loss_quant + loss_code_balance


                self.train_losses.update(loss.item())
                self.train_contra_losses.update(loss.item())
                self.train_q_losses.update(loss_quant.item())
                self.train_cb_losses.update(loss_code_balance.item())

                self.optimizer.zero_grad()
                loss.backward()

                self.optimizer.step()
                self.scheduler.step()
                self.batch_time.update(time.time() - start_time)         
                if index % self.args.print_interval == 0 and self.args.rank == 0:
                    Log.info("Epoch %d, Itrs %d/%d, Loss%8f, Contrastive_Loss%8f, Q_Loss%8f, Code_Balance_Loss%8f, Lr=%8f, Iter time=%f" %
                            (epoch, index, total_iters, self.train_losses.avg, self.train_contra_losses.avg, self.train_q_losses.avg, self.train_cb_losses.avg, self.scheduler.get_lr()[0], self.batch_time.avg))
                self.batch_time.reset()
                self.data_time.reset()
                self.train_losses.reset()
                self.train_contra_losses.reset()
                self.train_q_losses.reset()
                self.train_cb_losses.reset()
            
            # if epoch % self.args.val_interval == 0:
            #     self.val(epoch)

            self.B = ((self.H_eeg.detach() + self.H_ir.detach()) / 2).sign()
            if epoch % self.args.eval_interval == 0:    #验证
                self.eval(epoch)
            if epoch % self.args.save_interval == 0:
                # self.val(epoch)
                # 保存
                path = '/home/spi/FED-RE/Cross_Modal_Retrirval/checkpoints/epoch_%s_%s_of_%s_%s.pth' % (epoch,
                                                                                              self.args.dataset,
                                                                                              self.args.modality,
                                                                                              self.args.name)
                torch.save({
                    "epoch": epoch,
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "scheduler_state": self.scheduler.state_dict(),
                }, path)

                #self.save_hash_codes(epoch)
                print("Model saved as %s" % path)


###################################################################### nomee
    def val(self, epoch):
        self.val_losses.reset()
        self.model.eval()
        y_predict = np.array([])
        y_true = np.array([])
        start_time = time.time()
        with torch.no_grad():
            for index, train_batch in enumerate(self.val_loader):
                eeg, ir, label, index_inall = train_batch
                eeg,ir, y = eeg.to(self.device), ir.to(self.device), label.to(self.device)
                output, K_eeg, K_ir = self.model(eeg,ir)

                self.H_eeg_val[index_inall, :] = K_eeg
                self.H_ir_val[index_inall, :] = K_ir

                loss = self.loss_fn(score, y, self.weight_class, self.device)
                self.val_losses.update(loss.item())
                # dice = dice_coef(pred, y)
                # dist.all_reduce(dice/self.args.world_size)

                # 我的传统方法
                pred = (score.max(1, keepdim=True)[1]).cpu().numpy()
                label = y.cpu().numpy()
                y_predict = np.append(y_predict, pred)
                y_true = np.append(y_true, label)

        self.B_val = ((self.H_eeg_val.detach() + self.H_ir_val.detach()) / 2).sign()
        this_loss = self.val_losses.avg
        this_acc = accuracy_test(y_predict, y_true)
        this_mf1 = f1_test(y_predict, y_true)


        y_predict = np.array([])
        y_true = np.array([])

        eval_time = time.time() - start_time

        if self.args.rank == 0:
            Log.info("Validate, loss:%8f, acc:%8f, mf1:%8f, Val time=%f" % (this_loss, this_acc, this_mf1, eval_time))
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

#################################################################
    ##retrieval_han
    def eval(self, epoch):
        """
        Evaluate model. Calculate MAPs for current epoch
        Save model and hashes if current epoch is the best

        :param: epoch: current epoch
        """
        self.model.eval()

        qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY, indexes = self.get_codes_labels_indexes()   #求解query和databse的hash码

        mapi2t, mapt2i, mapi2i, mapt2t, mapavg = self.calc_maps_k(qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY,       #根据top N 计算MAP
                                                                  self.retrieval_map_k)

        map_k_5 = (mapi2t, mapt2i, mapi2i, mapt2t, mapavg)
        map_k_10 = self.calc_maps_k(qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY, 10)
        map_k_200 = self.calc_maps_k(qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY, 100)
        map_r = self.calc_maps_rad(qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY, [0, 1, 2, 3, 4, 5])       #根据hamming半径计算MAP
        p_at_k = self.calc_p_top_k(qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY)
        maps_eval = (map_k_5, map_k_10, map_k_200, map_r, p_at_k)

        self.update_maps_dict(mapi2t, mapt2i, mapi2i, mapt2t, mapavg)

        if mapavg > self.maps_max['avg']:
            self.update_max_maps_dict(mapi2t, mapt2i, mapi2i, mapt2t, mapavg)

            self.save_model()
            self.save_hash_codes(epoch)
            self.save_uncert(epoch)

        self.save_model('last')

        write_pickle(os.path.join(self.args.save_pklpath, str(epoch)+'_maps_eval.pkl'), maps_eval)

        self.model.train()

    def get_codes_labels_indexes(self):
        """
        Generate binary codes from duplet dataloaders for query and response

        :param: remove_replications: remove replications from dataset

        :returns: hash codes and labels for query and response, sample indexes
        """
        # hash X, hash Y, labels X/Y, image replication factor, indexes X, indexes Y
        qBX, qBY, qLXY, irf_q, (qIX, qIY) = self.generate_codes_from_dataloader(self.val_loader)
        # hash X, hash Y, labels X/Y, image replication factor, indexes X, indexes Y
        rBX, rBY, rLXY, irf_db, (rIX, rIY) = self.generate_codes_from_dataloader(self.train_loader)   #检索database

        # get Y Labels
        qLY = qLXY
        rLY = rLXY

        # X modality sometimes contains replicated samples (see datasets), remove them by selecting each nth element
        # remove replications for hash codes
        qBX = self.get_each_nth_element(qBX, irf_q)
        rBX = self.get_each_nth_element(rBX, irf_db)
        # remove replications for labels
        qLX = self.get_each_nth_element(qLXY, irf_q)
        rLX = self.get_each_nth_element(rLXY, irf_db)
        # remove replications for indexes
        qIX = self.get_each_nth_element(qIX, irf_q)
        rIX = self.get_each_nth_element(rIX, irf_db)

        return qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY, (qIX, qIY, rIX, rIY)

    @staticmethod
    def get_each_nth_element(arr, n):
        """
        intentionally ugly solution, needed to avoid query replications during test/validation

        :return: array
        """
        return arr[::n]

    def calc_maps_k(self, qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY, k):
        """
        Calculate MAPs, in regards to K

        :param: qBX: query hashes, modality X
        :param: qBY: query hashes, modality Y
        :param: rBX: response hashes, modality X
        :param: rBY: response hashes, modality Y
        :param: qLX: query labels, modality X
        :param: qLY: query labels, modality Y
        :param: rLX: response labels, modality X
        :param: rLY: response labels, modality Y
        :param: k: k

        :returns: MAPs
        """
        mapi2t = calc_map_k(qBX, rBY, qLX, rLY, k)
        mapt2i = calc_map_k(qBY, rBX, qLY, rLX, k)
        mapi2i = calc_map_k(qBX, rBX, qLX, rLX, k)
        mapt2t = calc_map_k(qBY, rBY, qLY, rLY, k)

        avg = (mapi2t.item() + mapt2i.item() + mapi2i.item() + mapt2t.item()) * 0.25

        mapi2t, mapt2i, mapi2i, mapt2t, mapavg = mapi2t.item(), mapt2i.item(), mapi2i.item(), mapt2t.item(), avg

        s = 'Valid: mAP@{:2d}, avg: {:3.3f}, i->t: {:3.3f}, t->i: {:3.3f}, i->i: {:3.3f}, t->t: {:3.3f}'
        Log.info(s.format(k, mapavg, mapi2t, mapt2i, mapi2i, mapt2t))

        return mapi2t, mapt2i, mapi2i, mapt2t, mapavg


    def calc_maps_rad(self, qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY, rs):
        """
        Calculate MAPs, in regard to Hamming radius

        :param: qBX: query hashes, modality X
        :param: qBY: query hashes, modality Y
        :param: rBX: response hashes, modality X
        :param: rBY: response hashes, modality Y
        :param: qLX: query labels, modality X
        :param: qLY: query labels, modality Y
        :param: rLX: response labels, modality X
        :param: rLY: response labels, modality Y
        :param: rs: hamming radiuses to output

        :returns: MAPs
        """
        mapsi2t = calc_map_rad(qBX, rBY, qLX, rLY)
        mapst2i = calc_map_rad(qBY, rBX, qLY, rLX)
        mapsi2i = calc_map_rad(qBX, rBX, qLX, rLX)
        mapst2t = calc_map_rad(qBY, rBY, qLY, rLY)

        mapsi2t, mapst2i, mapsi2i, mapst2t = mapsi2t.numpy(), mapst2i.numpy(), mapsi2i.numpy(), mapst2t.numpy()

        s = 'Valid: mAP HR{}, i->t: {:3.3f}, t->i: {:3.3f}, i->i: {:3.3f}, t->t: {:3.3f}'
        for r in rs:
            Log.info(s.format(r, mapsi2t[r], mapst2i[r], mapsi2i[r], mapst2t[r]))

        return mapsi2t, mapst2i, mapsi2i, mapst2t

    def calc_pr_curves(self, qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY):
        """
        Calculate PR-curves

        :param: qBX: query hashes, modality X
        :param: qBY: query hashes, modality Y
        :param: rBX: response hashes, modality X
        :param: rBY: response hashes, modality Y
        :param: qLX: query labels, modality X
        :param: qLY: query labels, modality Y
        :param: rLX: response labels, modality X
        :param: rLY: response labels, modality Y

        :returns: PR-curves dictionary
        """
        p_i2t, r_i2t = pr_curve(qBX, rBY, qLX, rLY, tqdm_label='I2T')
        p_t2i, r_t2i = pr_curve(qBY, rBX, qLY, rLX, tqdm_label='T2I')
        p_i2i, r_i2i = pr_curve(qBX, rBX, qLX, rLX, tqdm_label='I2I')
        p_t2t, r_t2t = pr_curve(qBY, rBY, qLY, rLY, tqdm_label='T2T')

        pr_dict = {'pi2t': p_i2t, 'ri2t': r_i2t,
                   'pt2i': p_t2i, 'rt2i': r_t2i,
                   'pi2i': p_i2i, 'ri2i': r_i2i,
                   'pt2t': p_t2t, 'rt2t': r_t2t}

        Log.info('Precision-recall values: {}'.format(pr_dict))

        return pr_dict



    def save_hash_codes(self,epoch):
            """
            Save hash codes on a disk
            """
            with torch.cuda.device(self.device):
                torch.save([self.H_eeg, self.H_ir], os.path.join(self.save_hashcode_path, 'hash_codes_eeg_ir'+str(epoch)+".pth"))
            with torch.cuda.device(self.device):
                torch.save(self.B, os.path.join(self.save_hashcode_path, 'hash_code'+str(epoch)+".pth"))

    def save_uncert(self, epoch):
        """
        Save uncert on a disk
        """
        with torch.cuda.device(self.device):
            torch.save([self.un_eeg, self.un_ir],
                       os.path.join(self.save_uncert_path, 'uncert_eeg_ir' + str(epoch) + ".pth"))

    def save_model(self, tag='best'):
        """
        Save model on the disk

        :param: tag: name tag
        """
        self.model.save("eval_save" + '_' + str(tag) + '.pth', self.checkpointpath, cuda_device=self.device)

    def calc_p_top_k(self, qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY):
        """
        Calculate P@K values

        :param: qBX: query hashes, modality X
        :param: qBY: query hashes, modality Y
        :param: rBX: response hashes, modality X
        :param: rBY: response hashes, modality Y
        :param: qLX: query labels, modality X
        :param: qLY: query labels, modality Y
        :param: rLX: response labels, modality X
        :param: rLY: response labels, modality Y

        :returns: P@K values
        """
        k = [1, 5, 10, 20, 50] + list(range(100, 1001, 100))

        pk_i2t = p_top_k(qBX, rBY, qLX, rLY, k, tqdm_label='I2T')
        pk_t2i = p_top_k(qBY, rBX, qLY, rLX, k, tqdm_label='T2I')
        pk_i2i = p_top_k(qBX, rBX, qLX, rLX, k, tqdm_label='I2I')
        pk_t2t = p_top_k(qBY, rBY, qLY, rLY, k, tqdm_label='T2T')

        pk_dict = {'k': k,
                   'pki2t': pk_i2t,
                   'pkt2i': pk_t2i,
                   'pki2i': pk_i2i,
                   'pkt2t': pk_t2t}

        Log.info('P@K values: {}'.format(pk_dict))

        return pk_dict

    def update_maps_dict(self, mapi2t, mapt2i, mapi2i, mapt2t, mapavg):
        """
        Update MAPs dictionary (append new values)

        :param: mapi2t: I-to-T MAP
        :param: mapt2i: T-to-I MAP
        :param: mapi2i: I-to-I MAP
        :param: mapt2t: T-to-T MAP
        :param: mapavg: average MAP
        """
        self.maps['i2t'].append(mapi2t)
        self.maps['t2i'].append(mapt2i)
        self.maps['i2i'].append(mapi2i)
        self.maps['t2t'].append(mapt2t)
        self.maps['avg'].append(mapavg)


    def update_max_maps_dict(self, mapi2t, mapt2i, mapi2i, mapt2t, mapavg):
        """
        Update max MAPs dictionary (replace values)

        :param: mapi2t: I-to-T MAP
        :param: mapt2i: T-to-I MAP
        :param: mapi2i: I-to-I MAP
        :param: mapt2t: T-to-T MAP
        :param: mapavg: average MAP
        """
        self.maps_max['i2t'] = mapi2t
        self.maps_max['t2i'] = mapt2i
        self.maps_max['i2i'] = mapi2i
        self.maps_max['t2t'] = mapt2t
        self.maps_max['avg'] = mapavg


    def generate_codes_from_dataloader(self, dataloader):
        """
        Generate binary codes from duplet dataloader

        :param: dataloader: duplet dataloader

        :returns: hash codes for given duplet dataloader, image replication factor of dataset
        """
        num = len(dataloader.dataset)

        irf = dataloader.dataset.image_replication_factor=1   #原来的意思 是一个图对应多个text，但是本任务是一一对应的了

        Bi = torch.zeros(num, self.hash_dim).to(self.device)
        Bt = torch.zeros(num, self.hash_dim).to(self.device)
        L = torch.zeros(num, self.args.label_dim).to(self.device)

        dataloader_idxs = []

        # for i, input_data in tqdm(enumerate(test_dataloader)):
        for i, (img, txt, label, idx) in enumerate(dataloader):
            sample_idxs=(idx,idx)  #?
            dataloader_idxs = self.stack_idxs(dataloader_idxs, sample_idxs)
            img = img.to(self.device)
            txt = txt.to(self.device)
            if len(label.shape) == 1:
                label = one_hot(label, num_classes=self.args.label_dim).to(self.device)
            else:
                label.to(self.device)
            bi = self.model.generate_img_code(img) #生成
            bt = self.model.generate_txt_code(txt)  #生成
            idx_end = min(num, (i + 1) * self.args.batch_size)
            Bi[i * self.args.batch_size: idx_end, :] = bi.data
            Bt[i * self.args.batch_size: idx_end, :] = bt.data
            L[i * self.args.batch_size: idx_end, :] = label.data

        Bi = torch.sign(Bi)
        Bt = torch.sign(Bt)
        return Bi, Bt, L, irf, dataloader_idxs

    @staticmethod
    def stack_idxs(idxs, idxs_batch):
        if len(idxs) == 0:
            return [ib for ib in idxs_batch]
        else:
            return [torch.hstack(i).detach() for i in zip(idxs, idxs_batch)]
#######################
    def test(self):
        """
        Test model. Calculate MAPs, PR-curves and P@K values.
        """

        """

               :param: qBX: query hashes, modality X
               :param: qBY: query hashes, modality Y
               :param: rBX: response hashes, modality X
               :param: rBY: response hashes, modality Y
               :param: qLX: query labels, modality X
               :param: qLY: query labels, modality Y
               :param: rLX: response labels, modality X
               :param: rLY: response labels, modality Y

               """
        self.model.to(self.device).train()

        self.model.eval()

        qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY, indexes = self.get_codes_labels_indexes()
        print(qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY)

        maps = self.calc_maps_k(qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY, self.args.retrieval_map_k)
        map_dict = self.make_maps_dict(*maps)
        pr_dict = self.calc_pr_curves(qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY)
        pk_dict = self.calc_p_top_k(qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY)
        self.save_test_results_dicts(map_dict, pr_dict, pk_dict)

        self.model.train()

        current = time.time()
        delta = current - self.since
        Log.info('Test complete in {:.0f}m {:.0f}s'.format(delta // 60, delta % 60))


    def save_test_results_dicts(self, map_dict, pr_dict, pk_dict):
        """
        Save test results dictionary

        :param: map_dict: MAPs dictionary
        :param: pr_dict: PR-curves dictionary
        :param: pk_dict: P@K values dictionary
        """
        write_pickle(os.path.join(self.args.save_pklpath, 'map_dict.pkl'), map_dict)
        write_pickle(os.path.join(self.args.save_pklpath, 'pr_dict.pkl'), pr_dict)
        write_pickle(os.path.join(self.args.save_pklpath, 'pk_dict.pkl'), pk_dict)

    def make_maps_dict(self, mapi2t, mapt2i, mapi2i, mapt2t, mapavg):
        """
        Make MAP dict from MAP values

        :param: mapi2t: I-to-T MAP
        :param: mapt2i: T-to-I MAP
        :param: mapi2i: I-to-I MAP
        :param: mapt2t: T-to-T MAP
        :param: mapavg: Average MAP

        :returns: MAPs dictionary
        """

        map_dict = {'mapi2t': mapi2t, 'mapt2i': mapt2i, 'mapi2i': mapi2i, 'mapt2t': mapt2t, 'mapavg': mapavg}

        s = 'Avg MAP: {:3.3f}, MAPs: i->t: {:3.3f}, t->i: {:3.3f}, i->i: {:3.3f}, t->t: {:3.3f}'
        Log.info(s.format(mapavg, mapi2t, mapt2i, mapi2i, mapt2t))

        return map_dict


def reduce_tensor(inp):
    with torch.no_grad():
        reduced_inp = inp
        dist.reduce(reduced_inp, dst=0)
    return reduced_inp
