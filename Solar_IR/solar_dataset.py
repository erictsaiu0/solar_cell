import random

from torch.utils.data import Dataset
import cv2
import numpy as np
import os
import torch

class Solar_Dataset(Dataset):
    def __init__(self, src_path, transforms_input, transform_label):
        self.src_path = src_path
        self.transforms_input = transforms_input
        self.transform_label = transform_label

        self.img_path_ls = []
        self.lbl_path_ls = []
        for dp, dn, fn in os.walk(self.src_path):
            for f in fn:
                if 'img.png' in f:
                    self.img_path_ls.append(os.path.join(dp, f))
                elif 'label.png' in f:
                    self.lbl_path_ls.append(os.path.join(dp, f))

    def __len__(self):
        return len(self.img_path_ls)

    def __getitem__(self, idx):
        img_path = self.img_path_ls[idx]
        lbl_path = self.lbl_path_ls[idx]

        def img_read(path, is_label=False):
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = np.array(image)
            if is_label:
                image = np.where(image > 0, 1, 0)
                image = image.astype('float32')
            return image

        img = img_read(img_path)
        lbl = img_read(lbl_path, is_label=True)

        seed = np.random.randint(1e5)
        if self.transforms_input is not None:
            torch.manual_seed(seed)
            img = self.transforms_input(img)
            torch.manual_seed(seed)
            lbl = self.transform_label(lbl)

        return img, lbl


class Solar_Dataset_wo_label(Dataset):
    def __init__(self, src_path, transforms):
        self.src_path = src_path
        self.transforms = transforms

        self.img_path_ls = []
        for dp, dn, fn in os.walk(self.src_path):
            if 'Zoon' in dp:
            # if 'mask_save' in dp:
                for f in fn:
                    if 'masked' not in f:
                        self.img_path_ls.append(os.path.join(dp, f))

    def __len__(self):
        return len(self.img_path_ls)

    def __getitem__(self, idx):
        img_path = self.img_path_ls[idx]

        def img_read(path):
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = np.array(image)
            return image

        img = img_read(img_path)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, img_path
