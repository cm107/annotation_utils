from __future__ import annotations
from common_utils.check_utils import check_required_keys
from common_utils.base.basic import BasicLoadableObject
from .handlers import NDDS_Annotation_Object_Handler
from .objects import CameraData

class NDDS_Annotation(BasicLoadableObject['NDDS_Annotation']):
    def __init__(
        self, camera_data: CameraData, objects: NDDS_Annotation_Object_Handler=None
    ):
        super().__init__()
        self.camera_data = camera_data
        self.objects = objects if objects is not None else NDDS_Annotation_Object_Handler()
    
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
