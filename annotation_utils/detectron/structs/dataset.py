from __future__ import annotations
from typing import List

from common_utils.common_types.bbox import BBox
from common_utils.common_types.segmentation import Segmentation
from common_utils.check_utils import check_type
from common_utils.common_types.keypoint import Keypoint2D_List
from common_utils.check_utils import check_required_keys
from logger import logger

try:
    from detectron2.structures.boxes import BoxMode
except ImportError:
    logger.error(f'detectron2 needs to be installed before using annotation_utils/detectron2')
    logger.error(f'Refer to https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md')
    raise ImportError

class Detectron2_Annotation:
    def __init__(
        self,
        iscrowd: int, bbox: BBox, keypoints: Keypoint2D_List, category_id: int,
        segmentation: Segmentation, bbox_mode: BoxMode=BoxMode.XYWH_ABS
    ):
        self.iscrowd = iscrowd
        self.bbox = bbox
        self.keypoints = keypoints
        self.category_id = category_id
        self.segmentation = segmentation
        self.bbox_mode = bbox_mode

    def __str__(self) -> str:
        return str(self.__dict__)

    def __repr__(self) -> str:
        return self.__str__()

    def copy(self) -> Detectron2_Annotation:
        return Detectron2_Annotation(
            iscrowd=self.iscrowd,
            bbox=self.bbox,
            keypoints=self.keypoints,
            category_id=self.category_id,
            segmentation=self.segmentation,
            bbox_mode=self.bbox_mode
        )

    def to_dict(self) -> dict:
        if self.bbox_mode != BoxMode.XYWH_ABS:
            raise NotImplementedError
        return {
            'iscrowd': self.iscrowd,
            'bbox': [self.bbox.xmin, self.bbox.ymin, self.bbox.xmax-self.bbox.xmin, self.bbox.ymax-self.bbox.ymin],
            'keypoints': self.keypoints.to_list(demarcation=False),
            'category_id': self.category_id,
            'segmentation': self.segmentation.to_list(demarcation=False),
            'bbox_mode': self.bbox_mode
        }

    @classmethod
    def from_dict(cls, ann_dict: dict) -> Detectron2_Annotation:
        check_required_keys(ann_dict, required_keys=['iscrowd', 'bbox', 'keypoints', 'category_id', 'bbox_mode'])
        
        if ann_dict['bbox_mode'] == BoxMode.XYWH_ABS:
            bbox_xmin, bbox_ymin, bbox_w, bbox_h = ann_dict['bbox']
            bbox = BBox(xmin=bbox_xmin, ymin=bbox_ymin, xmax=bbox_xmin+bbox_w, ymax=bbox_ymin+bbox_h)
        elif  ann_dict['bbox_mode'] == BoxMode.XYXY_ABS:
            bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = ann_dict['bbox']
            bbox = BBox(xmin=bbox_xmin, ymin=bbox_ymin, xmax=bbox_xmax, ymax=bbox_ymax)
        else:
            raise NotImplementedError
            
        return Detectron2_Annotation(
            iscrowd=ann_dict['iscrowd'],
            bbox=bbox,
            keypoints=Keypoint2D_List.from_list(value_list=ann_dict['keypoints'], demarcation=False),
            category_id=ann_dict['category_id'],
            segmentation=Segmentation.from_list(points_list=ann_dict['segmentation'] if 'segmentation' in ann_dict else [], demarcation=False),
            bbox_mode=ann_dict['bbox_mode']
        )

class Detectron2_Annotation_List:
    def __init__(self, ann_list: List[Detectron2_Annotation]):
        self.ann_list = ann_list

    def __str__(self) -> str:
        return str(self.to_dict_list())

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self) -> int:
        return len(self.ann_list)

    def __getitem__(self, idx: int) -> Detectron2_Annotation:
        if len(self.ann_list) == 0:
            logger.yellow(f"self.ann_list:\n{self.ann_list}")
            logger.error(f"Detectron2_Annotation_List is empty.")
            raise IndexError
        elif idx < 0 or idx >= len(self.ann_list):
            logger.error(f"Index out of range: {idx}")
            raise IndexError
        else:
            return self.ann_list[idx]

    def __setitem__(self, idx: int, value: Detectron2_Annotation):
        check_type(value, valid_type_list=[Detectron2_Annotation])
        self.ann_list[idx] = value

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self) -> Detectron2_Annotation:
        if self.n < len(self.ann_list):
            result = self.ann_list[self.n]
            self.n += 1
            return result
        else:
            raise StopIteration

    def copy(self) -> Detectron2_Annotation_List:
        Detectron2_Annotation_List(
            ann_list=self.ann_list.copy()
        )

    def to_dict_list(self) -> List[dict]:
        return [ann.to_dict() for ann in self]

    @classmethod
    def from_dict_list(cls, dict_list: List[dict]) -> Detectron2_Annotation_List:
        return Detectron2_Annotation_List(ann_list=[Detectron2_Annotation.from_dict(dict_item) for dict_item in dict_list])

class Detectron2_Annotation_Dict:
    def __init__(
        self,
        file_name: str, height: int, width: int, image_id: int, annotations: Detectron2_Annotation_List
    ):
        self.file_name = file_name
        self.height = height
        self.width = width
        self.image_id = image_id
        self.annotations = annotations

    def __str__(self) -> str:
        return str(self.__dict__)

    def __repr__(self) -> str:
        return self.__str__()

    def copy(self) -> Detectron2_Annotation_Dict:
        return Detectron2_Annotation_Dict(
            file_name=self.file_name,
            height=self.height,
            width=self.width,
            image_id=self.image_id,
            annotations=self.annotations.copy()
        )

    def to_dict(self) -> dict:
        return {
            'file_name': self.file_name,
            'height': self.height,
            'width': self.width,
            'image_id': self.image_id,
            'annotations': self.annotations.to_dict_list()
        }

    @classmethod
    def from_dict(cls, ann_dict: dict) -> Detectron2_Annotation_Dict:
        return Detectron2_Annotation_Dict(
            file_name=ann_dict['file_name'],
            height=ann_dict['height'],
            width=ann_dict['width'],
            image_id=ann_dict['image_id'],
            annotations=Detectron2_Annotation_List.from_dict_list(dict_list=ann_dict['annotations'])
        )

class Detectron2_Annotation_Dict_List:
    def __init__(self, ann_dict_list: List[Detectron2_Annotation_Dict]):
        self.ann_dict_list = ann_dict_list

    def __str__(self) -> str:
        return str(self.to_dict_list())

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self) -> int:
        return len(self.ann_dict_list)

    def __getitem__(self, idx: int) -> Detectron2_Annotation_Dict:
        if len(self.ann_dict_list) == 0:
            logger.error(f"Detectron2_Annotation_Dict_List is empty.")
            raise IndexError
        elif idx < 0 or idx >= len(self.ann_dict_list):
            logger.error(f"Index out of range: {idx}")
            raise IndexError
        else:
            return self.ann_dict_list[idx]

    def __setitem__(self, idx: int, value: Detectron2_Annotation_Dict):
        check_type(value, valid_type_list=[Detectron2_Annotation_Dict])
        self.ann_dict_list[idx] = value

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self) -> Detectron2_Annotation_Dict:
        if self.n < len(self.ann_dict_list):
            result = self.ann_dict_list[self.n]
            self.n += 1
            return result
        else:
            raise StopIteration

    def copy(self) -> Detectron2_Annotation_Dict_List:
        return Detectron2_Annotation_Dict_List(
            ann_dict_list=self.ann_dict_list.copy()
        )

    def to_dict_list(self) -> List[dict]:
        return [ann_dict.to_dict() for ann_dict in self]

    @classmethod
    def from_dict_list(cls, ann_dict_list: List[dict]) -> Detectron2_Annotation_Dict_List:
        return Detectron2_Annotation_Dict_List(ann_dict_list=[Detectron2_Annotation_Dict.from_dict(ann_dict) for ann_dict in ann_dict_list])
