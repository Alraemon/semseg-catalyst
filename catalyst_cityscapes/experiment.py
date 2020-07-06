from collections import OrderedDict
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from catalyst.dl import ConfigExperiment
from .dataset import NaturalData


def _img_list2dataset(img_list_fp:str, root = None):
    image_list, mask_list = [], []

    for line in open(str(img_list_fp), 'r'):
        _img_fp, _mask_fp = line.strip().split()
        _img_fp = Path(root)/_img_fp if root else Path(_img_fp)
        _mask_fp = Path(root)/_mask_fp if root else Path(_mask_fp)
        
        image_list.append(Path(_img_fp))
        mask_list.append(Path(_mask_fp))
    return image_list, mask_list


class Experiment(ConfigExperiment):
    import warnings
    warnings.filterwarnings("ignore")

    def get_datasets(
        self,
        stage: str,
        train_set: str,
        valid_set: str,
        label_mapping: dict = None,
        data_root: str = None,
        classes: int = 19,
        # image_size: int,
        # is_aug: bool,
        **kwargs
    ):

        datasets = OrderedDict()
        for mode, img_list_fp in zip(
            ["train", "valid"], [train_set, valid_set]
        ):
            # if mode == 'train'
            _image_fps, _mask_fps = _img_list2dataset(img_list_fp)

            tfms_fn = self.get_transforms(stage=stage, dataset=mode)

            datasets[mode] = NaturalData(
                images= _image_fps,
                masks= _mask_fps,
                label_mapping = label_mapping,
                transforms=tfms_fn,
                data_root = data_root,
                classes = classes 

            )

        return datasets
