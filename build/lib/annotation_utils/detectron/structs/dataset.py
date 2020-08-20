from __future__ import annotations
from typing import List

from common_utils.common_types.bbox import BBox
from common_utils.common_types.segmentation import Segmentation
from common_utils.common_types.keypoint import Keypoint2D_List
from common_utils.check_utils import check_required_keys
from logger import logger

from common_utils.base.basic import BasicLoadableObject, BasicLoadableHandler, BasicHandler

try:
    from detectron2.structures.boxes import BoxMode
except ImportError:
    logger.error(f'detectron2 needs to be installed before using annotation_utils/detectron2')
    logger.error(f'Refer to https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md')
    raise ImportError

class Detectron2_Annotation(BasicLoadableObject['Detectron2_Annotation']):
    def __init__(
        self,
        iscrowd: int, bbox: BBox, keypoints: Keypoint2D_List, category_id: int,
        segmentation: Segmentation, bbox_mode: BoxMode=BoxMode.XYWH_ABS
    ):
        super().__init__()
        self.iscrowd = iscrowd
        self.bbox = bbox
        self.keypoints = keypoints
        self.category_id = category_id
        self.segmentation = segmentation
        self.bbox_mode = bbox_mode

    def to_dict(self) -> dict:
        if self.bbox_mode == BoxMode.XYWH_ABS:
            bbox = self.bbox.to_list(output_format='pminsize')
        elif self.bbox_mode == BoxMode.XYXY_ABS:
            bbox = self.bbox.to_list(output_format='pminpmax')
        else:
            raise NotImplementedError
        return {
            'iscrowd': self.iscrowd,
            'bbox': bbox,
            'keypoints': self.keypoints.to_list(demarcation=True),
            'category_id': self.category_id,
            'segmentation': self.segmentation.to_list(demarcation=False),
            'bbox_mode': self.bbox_mode
        }

    @classmethod
    def from_dict(cls, ann_dict: dict) -> Detectron2_Annotation:
        check_required_keys(ann_dict, required_keys=['iscrowd', 'bbox', 'category_id', 'bbox_mode'])
        
        if ann_dict['bbox_mode'] == BoxMode.XYWH_ABS:
            bbox = BBox.from_list(ann_dict['bbox'], input_format='pminsize')
        elif ann_dict['bbox_mode'] == BoxMode.XYXY_ABS:
            bbox = BBox.from_list(ann_dict['bbox'], input_format='pminpmax')
        else:
            raise NotImplementedError
        if 'keypoints' in ann_dict:
            keypoints = Keypoint2D_List.from_list(value_list=ann_dict['keypoints'], demarcation=False)
        else:
            keypoints = Keypoint2D_List()

        return Detectron2_Annotation(
            iscrowd=ann_dict['iscrowd'],
            bbox=bbox,
            keypoints=keypoints,
            category_id=ann_dict['category_id'],
            segmentation=Segmentation.from_list(points_list=ann_dict['segmentation'] if 'segmentation' in ann_dict else [], demarcation=False),
            bbox_mode=ann_dict['bbox_mode']
        )

class Detectron2_Annotation_List(
    BasicLoadableHandler['Detectron2_Annotation_List', 'Detectron2_Annotation'],
    BasicHandler['Detectron2_Annotation_List', 'Detectron2_Annotation']
):
    def __init__(self, ann_list: List[Detectron2_Annotation]=None):
        super().__init__(obj_type=Detectron2_Annotation, obj_list=ann_list)
        self.ann_list = self.obj_list

    @classmethod
    def from_dict_list(cls, dict_list: List[dict]) -> Detectron2_Annotation_List:
        return Detectron2_Annotation_List(ann_list=[Detectron2_Annotation.from_dict(dict_item) for dict_item in dict_list])

class Detectron2_Annotation_Dict(BasicLoadableObject['Detectron2_Annotation_Dict']):
    def __init__(
        self,
        file_name: str, height: int, width: int, image_id: int, annotations: Detectron2_Annotation_List=None
    ):
        super().__init__()
        self.file_name = file_name
        self.height = height
        self.width = width
        self.image_id = image_id
        self.annotations = annotations if annotations is not None else Detectron2_Annotation_List()

    @classmethod
    def from_dict(cls, ann_dict: dict) -> Detectron2_Annotation_Dict:
        check_required_keys(
            ann_dict,
            required_keys=['file_name', 'height', 'width', 'image_id', 'annotations']
        )
        return Detectron2_Annotation_Dict(
            file_name=ann_dict['file_name'],
            height=ann_dict['height'],
            width=ann_dict['width'],
            image_id=ann_dict['image_id'],
            annotations=Detectron2_Annotation_List.from_dict_list(dict_list=ann_dict['annotations'])
        )

class Detectron2_Annotation_Dict_List(
    BasicLoadableHandler['Detectron2_Annotation_Dict_List', 'Detectron2_Annotation_Dict'],
    BasicHandler['Detectron2_Annotation_Dict_List', 'Detectron2_Annotation_Dict']
):
    def __init__(self, ann_dict_list: List[Detectron2_Annotation_Dict]=None):
        super().__init__(obj_type=Detectron2_Annotation_Dict, obj_list=ann_dict_list)
        self.ann_dict_list = self.obj_list

    @classmethod
    def from_dict_list(cls, ann_dict_list: List[dict]) -> Detectron2_Annotation_Dict_List:
        return Detectron2_Annotation_Dict_List(ann_dict_list=[Detectron2_Annotation_Dict.from_dict(ann_dict) for ann_dict in ann_dict_list])
