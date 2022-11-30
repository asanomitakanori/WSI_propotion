from pathlib import Path
import cv2
import numpy as np
import torch
import random

from scipy.ndimage import rotate

class WSI_loader(object):
    def __init__(self, data_path, level):
        data_path = Path(data_path)
        self.imgs = sorted(data_path.glob(f"size_level{level}/input/*"))
        self.gts = sorted(data_path.glob(f"size_level{level}/gt/*.npy"))
        self.height = 5000
        self.width = 5000
        assert len(self.imgs) ==  len(self.gts), \
            f'Imgs num {len(self.imgs)} is different from gts num {len(self.gts)}'

    def __len__(self):
        return len(self.imgs)

    @classmethod
    def flip_and_rotate(cls, img, gt, seed):
        img = rotate(img, 90 * (seed % 4))
        gt = rotate(gt, 90 * (seed % 4))

        if seed > 3:
            img = np.fliplr(img).copy()
            gt = np.fliplr(gt).copy()

        return img, gt

    def __getitem__(self, data_id):
        img = cv2.imread(str(self.imgs[data_id]))
        gt = np.load(str(self.gts[data_id]))
        if img.max() > 1:
            img = img / 255  
        if gt.max() > 1:
            gt = gt / 255  

        # random crop
        seed1 = []
        if self.random_crop==True:
            tmp = gt[0:img.shape[0] - self.height, 0:img.shape[1] - self.width, :]
            tumor_bed, no_label, residual, background = tmp[..., 0], tmp[..., 1], tmp[..., 2], tmp[..., 3]
            lt = [tumor_bed, no_label, residual, background]
            while len(seed1) == 0:
                seed1, seed2 = np.where(lt[np.random.randint(0, len(lt))]>0)
            seed = np.random.randint(0, len(seed1))
            seed1, seed2 = seed1[seed], seed2[seed]
            img = img[seed1:seed1 + self.height, seed2:seed2 + self.width]
            gt = gt[seed1:seed1 + self.height, seed2:seed2 + self.width]

        # random flip and rotate
        # seed = random.randrange(8)
        # img, gt = self.flip_and_rotate(img, gt, seed)

        img = img.transpose((2, 0, 1))
        gt = gt.transpose((2, 0, 1))

        return {
            'img': torch.from_numpy(img).type(torch.FloatTensor),
            'gt': torch.from_numpy(gt).type(torch.FloatTensor),
        }
