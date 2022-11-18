import torch
import albumentations
from torch.utils.data import DataLoader, Dataset 
import pytorch_lightning as pl

import os
import numpy as np


class Crop(object):

    def __init__(self, cropped_shape=None, random_crop=False):
        self.cropped_shape = cropped_shape
        if cropped_shape is not None:
            mel_num, spec_len = cropped_shape
            if random_crop:
                self.cropper = albumentations.RandomCrop
            else:
                self.cropper = albumentations.CenterCrop
            self.preprocessor = albumentations.Compose([self.cropper(mel_num, spec_len)])
        else:
            self.preprocessor = lambda **kwargs: kwargs

    def __call__(self, item):
        item['input'] = self.preprocessor(image=item['input'])['image']
        return item


class VASSpecs(torch.utils.data.Dataset):

    def __init__(self, split, spec_dir_path, mel_num=None, spec_len=None, spec_crop_len=None,
                 random_crop=None, crop_coord=None, for_which_class=None):
        super().__init__()
        self.split = split
        self.spec_dir_path = spec_dir_path
        # fixing split_path in here because of compatibility with vggsound which hangles it in vggishish
        self.split_path = f'./data/vas_{split}.txt'
        self.feat_suffix = '_mel.npy'

        if not os.path.exists(self.split_path):
            print(f'split does not exist in {self.split_path}..')

        full_dataset = open(self.split_path).read().splitlines()
        # ['baby/video_00000', ..., 'dog/video_00000', ...]
        if for_which_class:
            self.dataset = [v for v in full_dataset if v.startswith(for_which_class)]
        else:
            self.dataset = full_dataset

        unique_classes = sorted(list(set([cls_vid.split('/')[0] for cls_vid in self.dataset])))
        self.label2target = {label: target for target, label in enumerate(unique_classes)}

        self.transforms = Crop([mel_num, spec_crop_len], random_crop)

    def __getitem__(self, idx):
        item = {}

        cls, vid = self.dataset[idx].split('/')
        spec_path = os.path.join(self.spec_dir_path.replace('*', cls), f'{vid}{self.feat_suffix}')

        spec = np.load(spec_path)
        item['input'] = spec
        item['file_path_'] = spec_path

        item['label'] = cls
        item['target'] = self.label2target[cls]

        if self.transforms is not None:
            item = self.transforms(item)

        # specvqgan expects `image` and `file_path_` keys in the item
        # it also expects inputs in [-1, 1] but specs are in [0, 1]
        item['image'] = 2 * item['input'] - 1
        item.pop('input')

        return item

    def __len__(self):
        return len(self.dataset)



class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, spec_dir_path, num_workers=None,  mel_num = None, spec_len = None, spec_crop_len = None, random_crop = None):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else batch_size*2

        self.spec_dir_path = spec_dir_path
        self.mel_num = mel_num
        self.spec_len = spec_len
        self.spec_crop_len = spec_crop_len
        self.random_crop = random_crop

    def setup(self, stage=None):
        self.train_dataset = VASSpecs( 'train', 
                         spec_dir_path = self.spec_dir_path, 
                         mel_num = self.mel_num, 
                         spec_len = self.spec_len, 
                         spec_crop_len = self.spec_crop_len, 
                         random_crop = self.random_crop)
        self.val_dataset = VASSpecs( 'valid', 
                         spec_dir_path = self.spec_dir_path, 
                         mel_num = self.mel_num, 
                         spec_len = self.spec_len, 
                         spec_crop_len = self.spec_crop_len, 
                         random_crop = self.random_crop)
        # self.test_dataset = VASSpecs( 'test', 
        #                  spec_dir_path = self.spec_dir_path, 
        #                  mel_num = self.mel_num, 
        #                  spec_len = self.spec_len, 
        #                  spec_crop_len = self.spec_crop_len, 
        #                  random_crop = self.random_crop)



    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=self.worker_init_fn,
                          shuffle=True)
        
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=self.worker_init_fn)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=self.worker_init_fn)

    @staticmethod
    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)