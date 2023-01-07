from dataset.prostate import UCL, Promise12, UCLm
import dataset.transforms as T
import dataset.transformsm as Tm
from torch.utils import data
import torch
from utils.logger import Logger as Log
from dataset.EEGVideo_DataSet.dataset import get_training_eegandir_data, get_validation_eegandir_data
from dataset.EEGVideo_DataSet.temporal_transforms import (LoopPadding, TemporalRandomCrop,
                                 TemporalCenterCrop, TemporalEvenCrop,
                                 SlidingWindow, TemporalSubsampling)
from dataset.EEGVideo_DataSet.temporal_transforms import Compose as TemporalCompose
from dataset.EEGVideo_DataSet.spatial_transforms import (Compose, Normalize, Resize, CenterCrop,
                                CornerCrop, MultiScaleCornerCrop,
                                RandomResizedCrop, RandomHorizontalFlip,
                                ToTensor, ScaleValue, ColorJitter,
                                PickFirstChannels)
from EEG_seg_jiqun.data_loader.data_loaders import data_generator_np_OnlyShhs, LoadDataset_index_shhs



def get_loader(args):
    """ Dataset And Augmentation
    """
    if args.dataset == 'prostate':
        train_transform = T.Compose([
            T.RandomZoom(),
            T.RandomCrop((16,96,96)),
            T.RandomFlip(),
            T.RandomRotate(),
            T.RandomRotate90(),
            T.Normalize(),
            T.Standardize(),
            T.AdditiveGaussianNoise(),
            T.ToTensor()
        ])
        val_transform = T.Compose([
            T.CenterCrop((16,96,96)),
            T.Normalize(),
            T.Standardize(),
            #T.AdditiveGaussianNoise(),
            T.ToTensor()
        ])
        test_transform = T.Compose([
            T.Normalize(),
            T.Standardize(),
            T.ToTensor()
        ])


        #train_dst = UCL(root=args.data_root, modality=args.modality, transforms=train_transform)
        # train_dst = UCL(root=args.data_root, modality=args.modality, transforms=train_transform, file='dataset/prostate/ucltrain.csv')
        # val_dst = UCL(root=args.data_root, modality=args.modality, transforms=val_transform, file = 'dataset/prostate/ucltest.csv' )
        #test_dst = UCL(root=args.data_root, modality=args.modality,transforms=test_transform, file = 'dataset/prostate/ucltest.csv' )
        train_dst = UCL(root=args.data_root, modality=args.modality, transforms=train_transform, file='dataset/prostate/ucltrain.csv')
        val_dst = UCL(root=args.data_root, modality=args.modality, transforms=val_transform, file = 'dataset/prostate/ucltest.csv' )
        test_dst = Promise12(root=args.data_root,transforms=test_transform, file = 'dataset/prostate/PROMISE12.csv' )

        
        train_loader = data.DataLoader(
        train_dst, batch_size=args.batch_size,  num_workers=8,pin_memory=True,
        sampler=torch.utils.data.distributed.DistributedSampler(
            dataset=train_dst),
        drop_last=False)  # drop_last=True to ignore single-image batches.

        val_loader = data.DataLoader(
            val_dst, batch_size=8,  num_workers=8,pin_memory=True,
            sampler=torch.utils.data.distributed.DistributedSampler(
                dataset=val_dst)
            )

        test_loader = data.DataLoader(
            test_dst, batch_size=1, num_workers=8,pin_memory=True)

    print("Dataset: %s, Train set: %d, Val set: %d, Test set %d"%
          (args.dataset, len(train_dst), len(val_dst), len(test_dst)))

    return train_loader, val_loader, test_loader

def get_loaderm(args):
    """ Dataset And Augmentation
    """
    if args.dataset == 'prostate':
        train_transform = Tm.Compose([
            Tm.RandomZoom(),
            Tm.RandomCrop((16,96,96)),
            Tm.RandomFlip(),
            Tm.RandomRotate(),
            Tm.RandomRotate90(),
            Tm.Normalize(),
            Tm.Standardize(),
            Tm.AdditiveGaussianNoise(),
            Tm.ToTensor()
        ])
        val_transform = T.Compose([
            T.CenterCrop((16,96,96)),
            T.Normalize(),
            T.Standardize(),
            #T.AdditiveGaussianNoise(),
            T.ToTensor()
        ])
        test_transform = T.Compose([
            T.Normalize(),
            T.Standardize(),
            T.ToTensor()
        ])


        #train_dst = UCL(root=args.data_root, modality=args.modality, transforms=train_transform)
        # train_dst = UCL(root=args.data_root, modality=args.modality, transforms=train_transform, file='dataset/prostate/ucltrain.csv')
        # val_dst = UCL(root=args.data_root, modality=args.modality, transforms=val_transform, file = 'dataset/prostate/ucltest.csv' )
        #test_dst = UCL(root=args.data_root, modality=args.modality,transforms=test_transform, file = 'dataset/prostate/ucltest.csv' )
        train_dst = UCLm(root=args.data_root, transforms=train_transform, file='dataset/prostate/ucltrain.csv')
        val_dst = UCL(root=args.data_root, modality=args.modality, transforms=val_transform, file = 'dataset/prostate/ucltest.csv' )
        test_dst = Promise12(root=args.data_root,transforms=test_transform, file = 'dataset/prostate/PROMISE12.csv' )

        
        train_loader = data.DataLoader(
        train_dst, batch_size=args.batch_size,
            num_workers=8,
            pin_memory=True,
        sampler=torch.utils.data.distributed.DistributedSampler(
            dataset=train_dst),
        drop_last=False)  # drop_last=True to ignore single-image batches.

        val_loader = data.DataLoader(
            val_dst,
            batch_size=8,
            num_workers=8,
            pin_memory=True,
            sampler=torch.utils.data.distributed.DistributedSampler(
                dataset=val_dst)
            )

        test_loader = data.DataLoader(
            test_dst,
            batch_size=1,
            num_workers=8,
            pin_memory=True)

    print("Dataset: %s, Train set: %d, Val set: %d, Test set %d"%
          (args.dataset, len(train_dst), len(val_dst), len(test_dst)))

    return train_loader, val_loader, test_loader

def get_loader_uda(args):
    """ Dataset And Augmentation
    """
    if args.dataset == 'prostate':
        src_transform = T.Compose([
            T.RandomZoom(),
            T.RandomCrop((16,96,96)),
            T.RandomFlip(),
            T.RandomRotate(),
            T.RandomRotate90(),
            T.Normalize(),
            T.Standardize(),
            T.AdditiveGaussianNoise(),
            T.ToTensor()
        ])

        tgt_transform = T.Compose([
            T.RandomZoom(),
            T.RandomCrop((16,96,96)),
            T.RandomFlip(),
            T.RandomRotate(),
            T.RandomRotate90(),
            T.Normalize(),
            T.Standardize(),
            T.AdditiveGaussianNoise(),
            T.ToTensor()
        ])

        test_transform = T.Compose([
            T.Normalize(),
            T.Standardize(),
            T.ToTensor()
        ])


        src_dst = UCL(root=args.data_root, modality='t2w',  transforms=src_transform, file='dataset/prostate/ucltrain.csv')
        tgt_dst = Promise12(root=args.data_root,transforms=tgt_transform, file = 'dataset/prostate/PROMISE12.csv' )
        test_dst = Promise12(root=args.data_root,transforms=test_transform, file = 'dataset/prostate/PROMISE12.csv' )

        
        src_loader = data.DataLoader(
            src_dst, batch_size=args.batch_size,  num_workers=8,pin_memory=True,
            sampler=torch.utils.data.distributed.DistributedSampler(
                dataset=src_dst),
            drop_last=False)  # drop_last=True to ignore single-image batches.

        tgt_loader = data.DataLoader(
            tgt_dst, batch_size=args.batch_size,  num_workers=8,pin_memory=True,
            sampler=torch.utils.data.distributed.DistributedSampler(
                dataset=tgt_dst)
            )

        test_loader = data.DataLoader(
            test_dst, batch_size=1, num_workers=8,pin_memory=True)

    print("Dataset: %s, Train set: %d, Val set: %d, Test set %d"%
          (args.dataset, len(src_dst), len(tgt_dst), len(test_dst)))

    return src_loader, tgt_loader, test_loader

def get_loader_muda(args):
    """ Dataset And Augmentation
    """
    if args.dataset == 'prostate':
        src_transform = Tm.Compose([
            Tm.RandomZoom(),
            Tm.RandomCrop((16,96,96)),
            Tm.RandomFlip(),
            Tm.RandomRotate(),
            Tm.RandomRotate90(),
            Tm.Normalize(),
            Tm.Standardize(),
            Tm.AdditiveGaussianNoise(),
            Tm.ToTensor()
        ])
        tgt_transform = T.Compose([
            T.RandomZoom(),
            T.RandomCrop((16,96,96)),
            T.RandomFlip(),
            T.RandomRotate(),
            T.RandomRotate90(),
            T.Normalize(),
            T.Standardize(),
            T.AdditiveGaussianNoise(),
            T.ToTensor()
        ])

        test_transform = T.Compose([
            T.Normalize(),
            T.Standardize(),
            T.ToTensor()
        ])

        src_dst = UCLm(root=args.data_root, transforms=src_transform, file='dataset/prostate/ucltrain.csv')
        tgt_dst = Promise12(root=args.data_root,transforms=tgt_transform, file = 'dataset/prostate/PROMISE12.csv' )
        test_dst = Promise12(root=args.data_root,transforms=test_transform, file = 'dataset/prostate/PROMISE12.csv' )

        
        src_loader = data.DataLoader(
            src_dst, batch_size=args.batch_size,  num_workers=8,pin_memory=True,
            # sampler=torch.utils.data.distributed.DistributedSampler(
            #     dataset=src_dst),
            drop_last=False)  # drop_last=True to ignore single-image batches.

        tgt_loader = data.DataLoader(
            tgt_dst, batch_size=8,  num_workers=8,pin_memory=True,
            # sampler=torch.utils.data.distributed.DistributedSampler(
            #     dataset=tgt_dst)
            )

        test_loader = data.DataLoader(
            test_dst, batch_size=1, num_workers=8,pin_memory=True)

    print("Dataset: %s, Train set: %d, Val set: %d, Test set %d"%
          (args.dataset, len(src_dst), len(tgt_dst), len(test_dst)))

    return src_loader, tgt_loader, test_loader


def get_normalize_method(mean, std, no_mean_norm, no_std_norm):
    if no_mean_norm:
        if no_std_norm:
            return Normalize([0, 0, 0], [1, 1, 1])
        else:
            return Normalize([0, 0, 0], std)
    else:
        if no_std_norm:
            return Normalize(mean, [1, 1, 1])
        else:
            return Normalize(mean, std)


def img_preprocess(train_spat_crop, train_temp_crop, input_typee):

    sample_size = 112
    sample_duration = 16
    sample_t_stride = 1
    # train_crop = "random"
    train_crop_min_scale = 0.25
    train_crop_min_ratio = 0.75
    no_hflip = False
    colorjitter = False
    # train_t_crop = "random"
    mean = [0.4345, 0.4051, 0.3775]
    std = [0.2768, 0.2713, 0.2737]
    no_mean_norm = False
    no_std_norm = False
    value_scale = 1

    assert train_spat_crop in ['random', 'corner', 'center']
    spatial_transform = []
    if train_spat_crop == 'random':
        spatial_transform.append(
            RandomResizedCrop(
                sample_size, (train_crop_min_scale, 1.0),
                (train_crop_min_ratio, 1.0 / train_crop_min_ratio)))
    normalize = get_normalize_method(mean, std, no_mean_norm,
                                     no_std_norm)
    if not no_hflip:
        spatial_transform.append(RandomHorizontalFlip())
    if colorjitter:
        spatial_transform.append(ColorJitter())
    spatial_transform.append(ToTensor())
    if input_typee == 'flow':
        spatial_transform.append(PickFirstChannels(n=2))
    spatial_transform.append(ScaleValue(value_scale))
    spatial_transform.append(normalize)
    spatial_transform = Compose(spatial_transform)

    assert train_temp_crop in ['random', 'center']
    temporal_transform = []
    if sample_t_stride > 1:
        temporal_transform.append(TemporalSubsampling(sample_t_stride))
    if train_temp_crop == 'random':
        temporal_transform.append(TemporalRandomCrop(sample_duration))
    elif train_temp_crop == 'center':
        temporal_transform.append(TemporalCenterCrop(sample_duration))
    temporal_transform = TemporalCompose(temporal_transform)

    return spatial_transform, temporal_transform


def get_loader_muda_xiao(args):
    """ Dataset And Augmentation
    """
    video_training_path = r'LD-Data/fc_rs_image'
    video_validation_path = r'LD-Data/20val_img_fiveclass_rs'
    source_eeg_path = r'/home/spi/FED-RE/DATA/EEG_Clip_C4'
    annotation_path = r'/home/spi/FED-RE/DATA/list/ucf101_1.json'
    dataset = 'ucf101'
    input_type = 'rgb'
    file_type = 'jpg'
    spat_tran_form = 'random'
    temp_transform = 'random'
    spatial_transform, temporal_transform = img_preprocess(spat_tran_form, temp_transform, input_type)

    # target_eeg_path = r'LD-Data/SHHS/Resample329'
    target_eeg_path = r'guorongxiao/SampleData/SHHS'


    if args.dataset == 'prostate':

        # src_dst = UCLm(root=args.data_root, transforms=src_transform, file='dataset/prostate/ucltrain.csv')
        # tgt_dst = Promise12(root=args.data_root, transforms=tgt_transform, file='dataset/prostate/PROMISE12.csv')
        # test_dst = Promise12(root=args.data_root, transforms=test_transform, file='dataset/prostate/PROMISE12.csv')
        if args.rank == 0:
            Log.info(" data loading start ")

        src_dst = get_training_eegandir_data(video_training_path, source_eeg_path, annotation_path, dataset, input_type,
                                             file_type, spatial_transform=spatial_transform,
                                             temporal_transform=temporal_transform, supply_video_path=None,rank=args.rank)
        if args.rank == 0:
            Log.info("source data loaded, sum:{}".format(len(src_dst.data)))
        # tgt_files = load_folds_data_shhs(target_eeg_path)
        # tgt_dst, tgt_count = data_generator_np_OnlyShhs(tgt_files)
        tgt_dst = LoadDataset_index_shhs(target_eeg_path)
        if args.rank == 0:
            Log.info("target data loaded, sum:{}".format(len(tgt_dst)))
        test_dst = tgt_dst
        if args.rank == 0:
            Log.info("test data loaded, sum:{}".format(len(test_dst)))
        src_loader = data.DataLoader(
            src_dst, batch_size=args.batch_size,
            num_workers=8,
            pin_memory=False,
            sampler=torch.utils.data.distributed.DistributedSampler(dataset=src_dst),
            drop_last=True)  # drop_last=True to ignore single-image batches.

        tgt_loader = data.DataLoader(
            tgt_dst,
            batch_size=args.batch_size,
            num_workers=8,
            pin_memory=False,
            sampler=torch.utils.data.distributed.DistributedSampler(dataset=tgt_dst),
            drop_last=True
        )

        test_loader = data.DataLoader(
            test_dst,
            batch_size=args.batch_size,
            num_workers=8,
            pin_memory=False,
            # sampler=torch.utils.data.distributed.DistributedSampler(dataset=test_dst),
            drop_last=False
        )
    if args.rank == 0:
        Log.info("Dataset: %s, Train set: %d, Val set: %d, Test set %d" %
              (args.dataset, len(src_dst), len(tgt_dst), len(test_dst)))

    return src_loader, tgt_loader, test_loader

def get_loader_so(args):
    """ Dataset And Augmentation
    """
    video_training_path = args.video_training_path
    video_query_path = args.video_query_path
    video_database_path = args.video_database_path
    source_eeg_path = args.source_eeg_path
    annotation_path = args.annotation_path
    dataset = 'ucf101'
    input_type = 'rgb'
    file_type = 'jpg'
    spat_tran_form = 'random'
    temp_transform = 'random'
    spatial_transform, temporal_transform = img_preprocess(spat_tran_form, temp_transform, input_type)

    #target_eeg_path = r'/home/spi/FED-RE/DATA/EEG_Clip_C4'
    # target_eeg_path = r'LD-Data/SHHS/Resample329'
    # target_eeg_path = r'guorongxiao/SampleData/SHHS_Clip'


    if args.dataset == 'prostate':

        # src_dst = UCLm(root=args.data_root, transforms=src_transform, file='dataset/prostate/ucltrain.csv')
        # tgt_dst = Promise12(root=args.data_root, transforms=tgt_transform, file='dataset/prostate/PROMISE12.csv')
        # test_dst = Promise12(root=args.data_root, transforms=test_transform, file='dataset/prostate/PROMISE12.csv')
        if args.rank == 0:
            Log.info(" data loading start ")

        train_dst = get_training_eegandir_data(video_training_path, source_eeg_path, annotation_path, dataset, input_type,
                                             file_type, spatial_transform=spatial_transform,
                                             temporal_transform=temporal_transform, supply_video_path=None,rank=args.rank)
        if args.rank == 0:
            Log.info("source data loaded, sum:{}".format(len(train_dst.data)))
        vali_dst = get_validation_eegandir_data(video_query_path, source_eeg_path, annotation_path, dataset, input_type,
                                             file_type, spatial_transform=spatial_transform,
                                             temporal_transform=temporal_transform,rank=args.rank)

        if args.rank == 0:
            Log.info("target data loaded, sum:{}".format(len(train_dst.data)))
        # test_files = load_folds_data_shhs(target_eeg_path)
        # test_dst, test_count = data_generator_np_OnlyShhs(test_files)
        #test_dst = LoadDataset_index_shhs(target_eeg_path)
        database_dst = get_training_eegandir_data(video_database_path, source_eeg_path, annotation_path, dataset, input_type,
                                                file_type, spatial_transform=spatial_transform,
                                                temporal_transform=temporal_transform, rank=args.rank)
        # test_dst=[1]
        # if args.rank == 0:
        #     Log.info("test data loaded, sum:{}".format(len(test_dst)))

        train_loader = data.DataLoader(
            train_dst, batch_size=args.batch_size,
            num_workers=8,
            pin_memory=True,
            #sampler=torch.utils.data.distributed.DistributedSampler(dataset=train_dst),
            sampler=None,
            shuffle=True,
            drop_last=True)  # drop_last=True to ignore single-image batches.

        query_loader = data.DataLoader(
            vali_dst,
            batch_size=args.batch_size,
            num_workers=8,
            pin_memory=True,
            # sampler=torch.utils.data.distributed.DistributedSampler(dataset=vali_dst)
            drop_last=False
        )

        # test_loader = data.DataLoader(
        #     test_dst,
        #     batch_size=args.batch_size,
        #     num_workers=8,
        #     pin_memory=True,
        #     # sampler=torch.utils.data.distributed.DistributedSampler(dataset=test_dst)
        #     drop_last=False
        # )

        db_loader = data.DataLoader(
            database_dst, batch_size=args.batch_size,
            num_workers=8,
            pin_memory=True,
            # sampler=torch.utils.data.distributed.DistributedSampler(dataset=train_dst),
            sampler=None,
            #shuffle=True,
            drop_last=True)  # drop_last=True to ignore single-image batches.
        # test_loader = vali_loader
    if args.rank == 0:
        Log.info("Dataset: %s, Train set: %d, query set: %d, DB set %d" %
              (args.dataset, len(train_dst), len(vali_dst), len(train_dst)))

    return train_loader, query_loader, db_loader