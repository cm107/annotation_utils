from __future__ import annotations
from typing import List
from logger import logger
from common_utils.base.basic import BasicLoadableObject, BasicLoadableHandler, BasicHandler
from common_utils.common_types.keypoint import Keypoint2D_List
from common_utils.check_utils import check_required_keys

class KeypointResult(BasicLoadableObject):
    def __init__(self, image_id: int, category_id: int, keypoints: Keypoint2D_List, score: float=None, score_list: List[float]=None):
        super().__init__()
        self.image_id = image_id
        self.category_id = category_id
        self.keypoints = keypoints

        if score is None and score_list is None:
            logger.error(f'Need to provide either score or score_list.')
            raise Exception

        self.score = score
        self.score_list = score_list

    def to_dict(self) -> dict:
        result_dict =  {
            'image_id': self.image_id,
            'category_id': self.category_id,
            'keypoints': self.keypoints.to_list(demarcation=False)
        }
        if self.score is not None:
            result_dict['score'] = self.score
        if self.score_list is not None:
            result_dict['score_list'] = self.score_list
        return result_dict

    @classmethod
    def from_dict(self, item_dict: dict) -> KeypointResult:
        check_required_keys(
            item_dict,
            required_keys=['image_id', 'category_id', 'keypoints']
        )
        if 'score' not in item_dict and 'score_list' not in item_dict:
            logger.error(f"Need to provide a dictionary with either a 'score' key or a 'score_list' key.")
            raise Exception
        return KeypointResult(
            image_id=item_dict['image_id'],
            category_id=item_dict['category_id'],
            keypoints=Keypoint2D_List.from_list(item_dict['keypoints'], demarcation=False),
            score=item_dict['score']
        )

class KeypointResultHandler(
    BasicLoadableHandler['KeypointResultHandler', 'KeypointResult'],
    BasicHandler['KeypointResultHandler', 'KeypointResult']
):
    def __init__(self, kpt_results: List[KeypointResult]=None):
        super().__init__(obj_type=KeypointResult, obj_list=kpt_results)
        self.kpt_results = self.obj_list
    
    @classmethod
    def from_dict_list(self, dict_list: List[dict]) -> KeypointResultHandler:
        return KeypointResultHandler(
            kpt_results=[KeypointResult.from_dict(item_dict) for item_dict in dict_list]
        )