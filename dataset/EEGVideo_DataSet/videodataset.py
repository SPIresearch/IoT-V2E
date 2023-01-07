import json
from pathlib import Path
import os
import torch
import torch.utils.data as data
from utils.logger import Logger as Log
from random import randrange
import numpy as np
from .loader import VideoLoader
from .loader import EEGClipLoader
import torchvision.transforms as transforms

def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_database(data, subset, root_path, video_path_formatter):
    video_ids = []
    video_paths = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            video_ids.append(key)
            annotations.append(value['annotations'])
            if 'video_path' in value:
                video_paths.append(Path(value['video_path']))
            else:
                label = value['annotations']['label']
                video_paths.append(video_path_formatter(root_path, label, key))

    return video_ids, video_paths, annotations


class VideoDataset(data.Dataset):

    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 video_loader=None,
                 video_path_formatter=(lambda root_path, label, video_id:
                                       root_path / label / video_id),
                 image_name_formatter=lambda x: f'image_{x:05d}.jpg',
                 target_type='label'):
        self.data, self.class_names = self.__make_dataset(
            root_path, annotation_path, subset, video_path_formatter)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform

        if video_loader is None:
            self.loader = VideoLoader(image_name_formatter)
        else:
            self.loader = video_loader

        self.target_type = target_type

    def __make_dataset(self, root_path, annotation_path, subset,
                       video_path_formatter):
        with annotation_path.open('r') as f:
            data = json.load(f)
        video_ids, video_paths, annotations = get_database(
            data, subset, root_path, video_path_formatter)
        class_to_idx = get_class_labels(data)
        idx_to_class = {}
        for name, label in class_to_idx.items():
            idx_to_class[label] = name

        n_videos = len(video_ids)
        dataset = []
        for i in range(n_videos):
            if i % (n_videos // 5) == 0:
                print('dataset loading [{}/{}]'.format(i, len(video_ids)))

            if 'label' in annotations[i]:
                label = annotations[i]['label']
                label_id = class_to_idx[label]
            else:
                label = 'test'
                label_id = -1

            video_path = video_paths[i]
            if not video_path.exists():
                continue

            segment = annotations[i]['segment']
            if segment[1] == 1:
                continue

            frame_indices = list(range(segment[0], segment[1]))
            sample = {
                'video': video_path,
                'segment': segment,
                'frame_indices': frame_indices,
                'video_id': video_ids[i],
                'label': label_id
            }
            dataset.append(sample)

        return dataset, idx_to_class

    def __loading(self, path, frame_indices):
        clip = self.loader(path, frame_indices)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        return clip

    def __getitem__(self, index):
        path = self.data[index]['video']
        if isinstance(self.target_type, list):
            target = [self.data[index][t] for t in self.target_type]
        else:
            target = self.data[index][self.target_type]

        frame_indices = self.data[index]['frame_indices']
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)

        clip = self.__loading(path, frame_indices)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return clip, target

    def __len__(self):
        return len(self.data)

###############################3333

def get_eeg_database(data, subset, root_path, eeg_path, video_path_formatter, eeg_path_formatter,supply_video_path):
    video_ids = []
    video_paths = []
    eeg_paths = []
    annotations = []

    ## filter the classes
    # eeg_classes = sorted(eeg_path.iterdir())
    # eeg_classes = [x.name for x in eeg_classes]
    eeg_classes = ['W', 'N1', 'N2', 'N3', 'R']

    for key, value in data['database'].items():
        audio_file = eeg_path + '/' + key[0:8] + '/' + 'EEG_' + key + '.npy'
        # 如果存在对应的脑电数据
        if os.path.isfile(audio_file):
            # remove classes without audio
            if value['annotations']['label'] in eeg_classes:
                this_subset = value['subset']
                if this_subset == subset:
                    video_ids.append(key)
                    annotations.append(value['annotations'])
                    if 'video_path' in value:
                        video_paths.append(Path(value['video_path']))
                    else:
                        label = value['annotations']['label']
                        temp_path = video_path_formatter(root_path, label, key)
                        if not os.path.exists(temp_path):
                            temp_path = video_path_formatter(supply_video_path, label, key)
                        video_paths.append(temp_path)

                    ### eeg signal
                    # audio_file = eeg_path_formatter(eeg_path, label, key)
                    eeg_paths.append(audio_file)

    return video_ids, video_paths, eeg_paths, annotations















#####################################################
class EEGVideoDataset(data.Dataset):

    def __init__(self,
                 root_path,
                 eeg_path,
                 annotation_path,
                 subset,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 video_loader=None,
                 video_path_formatter=(lambda root_path, label, video_id:
                                       root_path / label / video_id),
                 eeg_path_formatter=(lambda eeg_path, label, video_id:
                                       eeg_path / label / video_id),
                 image_name_formatter=lambda x: f'image_{x:05d}.jpg',
                 supply_video_path=None,
                 target_type='label',
                 rank=None):
        self.data, self.class_names, self.n_videos = self.__make_eeg_dataset(
            root_path, eeg_path, annotation_path, subset, video_path_formatter, eeg_path_formatter,supply_video_path,rank=rank)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform

        if video_loader is None:
            self.loader = VideoLoader(image_name_formatter)
        else:
            self.loader = video_loader
        self.eeg_loader = EEGClipLoader()
        self.target_type = target_type

    def __make_eeg_dataset(self, root_path, eeg_path, annotation_path, subset,
                       video_path_formatter, audio_path_formatter,supply_video_path,rank=None):
        # with annotation_path.open('r') as f:
        with open(annotation_path,'r') as f:
            data = json.load(f)
        video_ids, video_paths, eeg_paths, annotations = get_eeg_database(
            data, subset, root_path, eeg_path, video_path_formatter, audio_path_formatter,supply_video_path)

        class_to_idx = get_class_labels(data)
        idx_to_class = {}
        for name, label in class_to_idx.items():
            idx_to_class[label] = name

        n_videos = len(video_ids)
        dataset = []
        for i in range(n_videos):
            if i % 1000 == 0 and rank == 0:
                Log.info(('dataset loading [{}/{}]'.format(i, len(video_ids))))

            if 'label' in annotations[i]:
                label = annotations[i]['label']
                label_id = class_to_idx[label]
            else:
                label = 'test'
                label_id = -1

            video_path = video_paths[i]
            # if not video_path.exists():
            if not os.path.exists(video_path):
                continue

            segment = annotations[i]['segment']
            if segment[1] == 1:
                continue

            frame_indices = list(range(segment[0], segment[1]))
            # if i == 1:
            #     print(segment[0])
            #     print(segment[1])
            #     print(segment)

            sample = {
                'video': video_path,
                'eeg': str(eeg_paths[i]),
                'segment': segment,
                'frame_indices': frame_indices,
                'video_id': video_ids[i],
                'label': label_id,
                "i_in_all":i
            }
            dataset.append(sample)
        return dataset, idx_to_class, n_videos

    def __loading(self, path, frame_indices, eeg_filename):

        clip = self.loader(path, frame_indices)

        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        else:
            transf = transforms.ToTensor()
            clip = [transf(img) for img in clip]

        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        eeg = self.eeg_loader(eeg_filename)
        # eeg feature should be 1X512 dimension
        # sampled one features from the all the temporal eeg features
        if eeg is None:
            eeg_dim = 3840
            eeg = np.zeros(eeg_dim, dtype=np.float32)  # AudioCNN14embed512
            print("eeg error")

        # if eeg is not None and len(eeg.shape) > 1:
        #     # eeg = np.mean(eeg, axis=0)
        #     ind = randrange(eeg.shape[0])
        #     eeg = eeg[ind]
        return clip, eeg

    def __getitem__(self, index):
        path = self.data[index]['video']
        eeg_filename = self.data[index]['eeg']

        if isinstance(self.target_type, list):
            target = [self.data[index][t] for t in self.target_type]
        else:
            target = self.data[index][self.target_type]

        frame_indices = self.data[index]['frame_indices']
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)

        # print(frame_indices)
        # print('\n')

        clip, eeg = self.__loading(path, frame_indices, eeg_filename)

        eeg = eeg.astype(np.float32)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return eeg, clip, target, index

    def __len__(self):
        return len(self.data)
