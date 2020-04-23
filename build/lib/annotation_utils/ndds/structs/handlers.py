from __future__ import annotations
import json
from typing import List
from common_utils.check_utils import check_file_exists
from ...base.structs import BaseStructHandler
from .objects import NDDS_Annotation_Object

class NDDS_Annotation_Object_Handler(BaseStructHandler['NDDS_Annotation_Object_Handler', 'BaseStructObject']):
    def __init__(self, ndds_obj_list: List[NDDS_Annotation_Object]=None):
        super().__init__(obj_type=NDDS_Annotation_Object, obj_list=ndds_obj_list)
        self.objects = self.obj_list

    @classmethod
    def from_dict_list(cls, dict_list: List[dict]) -> NDDS_Annotation_Object_Handler:
        return NDDS_Annotation_Object_Handler(
            ndds_obj_list=[NDDS_Annotation_Object.from_dict(obj_dict) for obj_dict in dict_list]
        )

    @classmethod
    def load_from_path(cls, json_path: str) -> NDDS_Annotation_Object_Handler:
        check_file_exists(json_path)
        json_data = json.load(open(json_path, 'r'))
        return NDDS_Annotation_Object_Handler.from_dict_list(json_data)