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
    
    def to_labeled(self, model_name: str=None, test_name: str=None) -> LabeledBBoxResult:
        return LabeledBBoxResult(
            coco_result=self,
            model_name=model_name,
            test_name=test_name
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

    def to_labeled(self, model_name: str=None, test_name: str=None) -> LabeledBBoxResultHandler:
        return LabeledBBoxResultHandler([result.to_labeled(model_name=model_name, test_name=test_name) for result in self])

class LabeledBBoxResult(BasicLoadableObject['LabeledBBoxResult']):
    def __init__(self, coco_result: BBoxResult, test_name: str=None, model_name: str=None):
        super().__init__()
        self.coco_result = coco_result
        self.test_name = test_name
        self.model_name = model_name
    
    def to_dict(self) -> dict:
        return {
            'coco_result': self.coco_result.to_dict(),
            'test_name': self.test_name,
            'model_name': self.model_name
        }
    
    @classmethod
    def from_dict(cls, item_dict: dict) -> LabeledBBoxResult:
        return LabeledBBoxResult(
            coco_result=BBoxResult.from_dict(item_dict['coco_result']),
            test_name=item_dict['test_name'],
            model_name=item_dict['model_name']
        )

class LabeledBBoxResultHandler(
    BasicLoadableHandler['LabeledBBoxResultHandler', 'LabeledBBoxResult'],
    BasicHandler['LabeledBBoxResultHandler', 'LabeledBBoxResult']
):
    def __init__(self, results: List[LabeledBBoxResult]=None):
        super().__init__(obj_type=LabeledBBoxResult, obj_list=results)
        self.results = self.obj_list

    @classmethod
    def from_dict_list(cls, dict_list: List[dict]) -> LabeledBBoxResult:
        return LabeledBBoxResultHandler([LabeledBBoxResult.from_dict(item_dict) for item_dict in dict_list])