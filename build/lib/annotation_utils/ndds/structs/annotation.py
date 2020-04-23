from __future__ import annotations
import json
from common_utils.check_utils import check_file_exists, check_required_keys

from ...base.structs import BaseStructObject
from .handlers import NDDS_Annotation_Object_Handler
from .objects import CameraData

class NDDS_Annotation(BaseStructObject['NDDS_Annotation']):
    def __init__(
        self, camera_data: CameraData, objects: NDDS_Annotation_Object_Handler=None
    ):
        super().__init__()
        self.camera_data = camera_data
        self.objects = objects if objects is not None else NDDS_Annotation_Object_Handler()

    def __str__(self):
        return f"NDDS_Annotation({self.to_dict()})"

    def to_dict(self) -> dict:
        return {
            'camera_data': self.camera_data.to_dict(),
            'objects': self.objects.to_dict_list()
        }

    @classmethod
    def from_dict(self, ann_dict: dict) -> NDDS_Annotation:
        check_required_keys(
            ann_dict,
            required_keys=[
                'camera_data', 'objects'
            ]
        )
        return NDDS_Annotation(
            camera_data=CameraData.from_dict(ann_dict['camera_data']),
            objects=NDDS_Annotation_Object_Handler.from_dict_list(ann_dict['objects'])
        )
    
    @classmethod
    def load_from_path(cls, json_path: str) -> NDDS_Annotation:
        check_file_exists(json_path)
        json_dict = json.load(open(json_path, 'r'))
        return NDDS_Annotation.from_dict(json_dict)