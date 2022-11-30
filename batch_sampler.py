import random
import numpy as np
import re
import glob
import joblib
from utils.dataset import *

from torch.utils import data as data
from torch.utils.data import DataLoader


import hydra 
from hydra.utils import to_absolute_path as abs_path


class SequentialSampler(data.Sampler):
    def __init__(self, imgs_dir, wsis, shuffle=False):
        self.imgs_dir = imgs_dir
        self.sub_classes = [0, 1, 2]
        self.shuffle = shuffle
        self.wsis = wsis
        self.dataset = self.get_files(wsis)
        self.lt = []
        self.length = 0
        for wsi in self.wsis:
            pl = [data for data in self.dataset if wsi in data]
            self.lt.append(list(range(self.length, self.length + len(pl), 1)))
            self.length += len(pl)

    def __len__(self):
        return len(self.lt)

    def __iter__(self):
        if self.shuffle:
            [random.shuffle(i) for i in self.lt]
            return iter(random.sample(self.lt, len(self.lt)))
        else:
            return iter(self.lt)
            # yield self.lt

    def get_files(self, wsis):
        re_pattern = re.compile('|'.join([f"/{i}/" for i in self.sub_classes]))
        files_list = []
        for wsi in wsis:
            files_list.extend(
                [
                    p for p in natsorted(glob.glob(self.imgs_dir + f"*/{wsi}_*/*.png", recursive=True))
                    if bool(re_pattern.search(p))
                ]
            )
        return files_list


@hydra.main(config_path='config', config_name='config_lightning')
def main(cfg):
    train_wsis = joblib.load('/home/asanomi/デスクトップ/WSI_PL/liu/cv0_train_wsi.jb')
    sampler = SequentialSampler(
                                '/home/asanomi/MNISTdata/202203_chemotherapy/mnt1_LEV2/',
                                train_wsis, 
                                False
                                )
    transform = {'Resize': False, 'HFlip': True, 'VFlip': True}
    fold = 0
    train_wsis = joblib.load(abs_path('liu' + f'/cv{fold}_train_wsi.jb'))
    valid_wsis = joblib.load(abs_path('liu' + f'/cv{fold}_valid_wsi.jb'))
    test_wsis = joblib.load(abs_path('liu' + f'/cv{fold}_test_wsi.jb'))

    # train dataloader 
    dataset = WSIDataset(
        imgs_dir='/home/asanomi/MNISTdata/202203_chemotherapy/mnt1_LEV2/',
        train_wsis=train_wsis,
        valid_wsis=valid_wsis,
        test_wsis=test_wsis,
        classes=[0, 1, 2],
        shape=[256, 256],
        transform=transform
    )
    train_set, _, _ = dataset.get()
    train_loader = DataLoader(train_set, 
                              batch_sampler=sampler,
                              num_workers=1, 
                              shuffle=False,
                              )
    x = 0

    for batch in train_loader:
        ft, labels, name = batch['img'], batch['label'], batch['name']
        ft, labels = ft[0:1000], labels[0:1000]
        print(np.unique(name))

if __name__ == '__main__':
    main()