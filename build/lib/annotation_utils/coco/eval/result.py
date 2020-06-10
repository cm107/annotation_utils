from __future__ import annotations
from logger import logger
from common_utils.base.basic import BasicLoadableObject
from .bbox import BBoxResultHandler
from .keypoint import KeypointResultHandler

class COCO_Results(BasicLoadableObject):
    def __init__(
        self, bbox_results: BBoxResultHandler=None, kpt_results: KeypointResultHandler=None,
        gt_dataset_path: str=None
    ):
        super().__init__()
        self.bbox_results = bbox_results if bbox_results is not None else BBoxResultHandler()
        self.kpt_results = kpt_results if kpt_results is not None else KeypointResultHandler()
        self.gt_dataset_path = gt_dataset_path

    def to_dict(self) -> dict:
        result_dict = {}
        if len(self.bbox_results) > 0:
            result_dict['bbox_results'] = self.bbox_results.to_dict_list()
        if len(self.kpt_results) > 0:
            result_dict['kpt_results'] = self.kpt_results.to_dict_list()
        if self.gt_dataset_path is not None:
            result_dict['gt_dataset_path'] = self.gt_dataset_path
        return result_dict
    
    @classmethod
    def from_dict(self, item_dict: dict) -> COCO_Results:
        if not ('bbox_results' in item_dict or 'kpt_results' in item_dict):
            logger.warning(f'There are no results to load from item_dict.')
        return COCO_Results(
            bbox_results=BBoxResultHandler.from_dict_list(item_dict['bbox_results']) if 'bbox_results' in item_dict else None,
            kpt_results=KeypointResultHandler.from_dict_list(item_dict['kpt_results']) if 'kpt_results' in item_dict else None,
            gt_dataset_path=item_dict['gt_dataset_path'] if 'gt_dataset_path' in item_dict else None
        )
    