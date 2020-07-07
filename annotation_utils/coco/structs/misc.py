from __future__ import annotations
from typing import List
from logger import logger
from common_utils.check_utils import check_type_from_list, check_type
from common_utils.common_types.bbox import BBox
from common_utils.common_types.segmentation import Polygon
from common_utils.common_types.keypoint import Keypoint2D
from .objects import COCO_Category

class KeypointGroup:
    def __init__(
        self, bound_obj, coco_cat: COCO_Category,
        kpt_list: List[Keypoint2D]=None, kpt_label_list: List[str]=None,
        img_filename: str=None
    ):
        check_type(bound_obj, valid_type_list=[BBox, Polygon])
        self.bound_obj = bound_obj
        check_type(coco_cat, valid_type_list=[COCO_Category])
        self.coco_cat = coco_cat
        if kpt_list is not None:
            check_type_from_list(kpt_list, valid_type_list=[Keypoint2D])
            self.kpt_list = kpt_list
        else:
            self.kpt_list = []
        self.kpt_label_list = []
        self.img_filename = img_filename
        if kpt_list is not None or kpt_label_list is not None:
            if kpt_list is None or kpt_label_list is None:
                logger.error(f'Must provide both kpt_list and kpt_label_list, or neither.')
                logger.error(f'kpt_list: {kpt_list}\nkpt_label_list: {kpt_label_list}')
                logger.error(f'Ground truth labels: {self.coco_cat.keypoints}')
                if self.img_filename is not None:
                    logger.error(f'Image filename: {self.img_filename}')
                raise Exception
            if len(kpt_list) != len(kpt_label_list):
                logger.error(f'len(kpt_list) == {len(kpt_list)} != {len(kpt_label_list)} == len(kpt_label_list)')
                raise Exception
            for kpt, label in zip(kpt_list, kpt_label_list):
                self.register(kpt=kpt, label=label)
        self.postponed_kpt_list = []
        self.postponed_kpt_label_list = []
    
    def register(self, kpt: Keypoint2D, label: str, strict: bool=True):
        if label in self.kpt_label_list:
            logger.error(f"Keypoint label '{label}' already exists in self.kpt_list.")
            logger.error(f'Currently registered labels: {self.kpt_label_list}')
            logger.error(f'Ground truth labels: {self.coco_cat.keypoints}')
            if self.img_filename is not None:
                logger.error(f'Image filename: {self.img_filename}')
            raise Exception
        if label not in self.coco_cat.keypoints:
            if not strict:
                # raise NotImplementedError
                self.postponed_kpt_list.append(kpt)
                self.postponed_kpt_label_list.append(label)
            else:
                logger.error(f"Label {label} couldn't be found in corresponding category's keypoint list.")
                logger.error(f'coco_cat.keypoints: {self.coco_cat.keypoints}')
                raise Exception
        else:
            self.kpt_list.append(kpt)
            self.kpt_label_list.append(label)