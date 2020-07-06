from typing import List
from pathlib import Path

from skimage.io import imread as gif_imread

from torch.utils.data import Dataset
import warnings
from catalyst import utils
from catalyst.data import ListDataset

import cv2
import os
import numpy as np
from PIL import Image
from random import shuffle, randint
import inspect
import torch
from catalyst import utils


class NaturalData(Dataset):
    """Dataset for segmentation tasks
    Returns a dict with ``image``, ``mask`` and ``filename`` keys
    """

    def __init__(
        self, images: List[Path], masks: List[Path] = None, transforms=None,
            label_mapping = None, data_root = None, classes = 19, 
    ) -> None:
        """
        Args:
            images (List[Path]): list of paths to the images
            masks (List[Path]): list of paths to the masks
                (names must be the same as in images)
            transforms: optional dict transforms
        """
        self.images = images
        self.masks = masks
        self.transforms = transforms
        # self.label_mapping = label_mapping
        # self.class_values = {v for k, v in label_mapping.items()
        #                      } if label_mapping != None else None
        self.data_root = data_root
        self.classes = classes

    def __len__(self) -> int:
        """Length of the dataset"""
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        """Main method"""

        image_path, mask_path = str(self.images[idx]), str(self.masks[idx])
        if self.data_root is not None:
            image_path = os.path.join(self.data_root, image_path)
            mask_path = os.path.join(self.data_root, mask_path)

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        image = np.float32(image)
        # image = utils.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W
        # mask = utils.gif_imread(mask_path)
        # mask[mask==255] = 0

        result = {'image': image, 'mask' : mask}
        if self.transforms:
            try:
                result = self.transforms(result) # ablu.transform 可以对2d mask 进行操作
            except:
                result = self.transforms(force_apply = True, **result)
            # add 1 channel to binary mask to be compatible with prediction
            result['mask'] = result['mask'].long()
            # mask_ont_hot = torch.zeros(class_num).scatter_(1, label, 1)
            # mask_one_hot = torch.stack([result['mask'] == i for i in range(self.classes)], dim = 0)  #.unsqueeze(0)
            # result['mask'] = mask_one_hot.float()

        result['filename'] = str(image_path)
        # print_range(result['image'].numpy(), result['mask'].numpy())
        return result

get_s_ix = lambda fn: int(fn.stem.split(os.sep)[0])

pil_load = lambda fn: np.array(Image.open(fn))


def print_range(image, mask):
    print('    \timage\tmask')
    print('size\t%s\t%s'%(image.shape, mask.shape))
    print('range\t%s\t%s' %((np.min(image), np.max(image)), (np.min(mask), np.max(mask))))