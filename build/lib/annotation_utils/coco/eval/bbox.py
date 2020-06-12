from __future__ import annotations
from typing import List
from common_utils.base.basic import BasicLoadableObject, BasicLoadableHandler, BasicHandler
from common_utils.common_types.bbox import BBox
from common_utils.check_utils import check_required_keys

class BBoxResult(BasicLoadableObject):
    def __init__(self, image_id: int, category_id: int, bbox: BBox, score: float):
        super().__init__()
        self.image_id = image_id
        self.category_id = category_id
        self.bbox = bbox
        self.score = score
    
    def to_dict(self) -> dict:
        return {
            'image_id': self.image_id,
            'category_id': self.category_id,
            'bbox': self.bbox.to_list(output_format='pminsize'),
            'score': self.score
        }

    @classmethod
    def from_dict(self, item_dict: dict) -> BBoxResult:
        check_required_keys(
            item_dict,
            required_keys=['image_id', 'category_id', 'bbox', 'score']
        )
        return BBoxResult(
            image_id=item_dict['image_id'],
            category_id=item_dict['category_id'],
            bbox=BBox.from_list(bbox=item_dict['bbox'], input_format='pminsize'),
            score=item_dict['score']
        )

class BBoxResultHandler(
    BasicLoadableHandler['BBoxResultHandler', 'BBoxResult'],
    BasicHandler['BBoxResultHandler', 'BBoxResult']
):
    def __init__(self, bbox_results: List[BBoxResult]=None):
        super().__init__(obj_type=BBoxResult, obj_list=bbox_results)
        self.bbox_results = self.obj_list
    
    @classmethod
    def from_dict_list(self, dict_list: List[dict]) -> BBoxResultHandler:
        return BBoxResultHandler(
            bbox_results=[BBoxResult.from_dict(item_dict) for item_dict in dict_list]
        )