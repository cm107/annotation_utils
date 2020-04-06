from __future__ import annotations

import cv2
from typing import List
import json

from logger import logger
from common_utils.time_utils import get_present_year, get_present_time_Ymd, \
    get_ctime
from common_utils.user_utils import get_username
from common_utils.check_utils import check_required_keys, check_file_exists, check_type_from_list
from common_utils.path_utils import get_filename
from common_utils.file_utils import file_exists
from common_utils.common_types.keypoint import Keypoint2D_List, Keypoint3D_List
from common_utils.common_types.bbox import BBox
from common_utils.common_types.segmentation import Segmentation

from ...camera import Camera
from ....base import BaseStructObject

class COCO_Info(BaseStructObject['COCO_Info']):
    def __init__(
        self,
        description: str,
        url: str="https://github.com/cm107/annotation_utils",
        version: str="1.0",
        year: str=get_present_year(),
        contributor: str=get_username(),
        date_created: str=get_present_time_Ymd()
    ):
        super().__init__()
        self.description = description
        self.url = url
        self.version = version
        self.year = year
        self.contributor = contributor
        self.date_created = date_created

    def __str__(self):
        print_str = ""
        print_str += f"description:\n\t{self.description}\n"
        print_str += f"url:\n\t{self.url}\n"
        print_str += f"version:\n\t{self.version}\n"
        print_str += f"year:\n\t{self.year}\n"
        print_str += f"contributor:\n\t{self.contributor}\n"
        print_str += f"date_created:\n\t{self.date_created}\n"
        return print_str

    @classmethod
    def from_dict(cls, info_dict: dict) -> COCO_Info:
        check_required_keys(
            info_dict,
            required_keys=['description', 'url', 'version', 'year', 'contributor', 'date_created']
        )
        return COCO_Info(
            description=info_dict['description'],
            url=info_dict['url'],
            version=info_dict['version'],
            year=info_dict['year'],
            contributor=info_dict['contributor'],
            date_created=info_dict['date_created']
        )

    @classmethod
    def load_from_path(cls, json_path: str) -> COCO_Info:
        check_file_exists(json_path)
        json_dict = json.load(open(json_path, 'r'))
        return COCO_Info.from_dict(json_dict)

class COCO_License(BaseStructObject['COCO_License']):
    def __init__(self, url: str, id: int, name: str):
        self.url = url
        self.id = id
        self.name = name

    def __str__(self):
        return f"url: {self.url}, id: {self.id}, name: {self.name}"

    def is_equal_to(self, other: COCO_License, exclude_id: bool=True) -> bool:
        result = True
        result = result and self.url == other.url
        result = result and self.name == other.name
        if not exclude_id:
            result = result and self.id == other.id
        return result

    @classmethod
    def from_dict(cls, license_dict: dict) -> COCO_License:
        check_required_keys(
            license_dict,
            required_keys=['url', 'id', 'name']
        )
        return COCO_License(
            url=license_dict['url'],
            id=license_dict['id'],
            name=license_dict['name']
        )

    @classmethod
    def load_from_path(cls, json_path: str) -> COCO_License:
        check_file_exists(json_path)
        json_dict = json.load(open(json_path, 'r'))
        return COCO_License.from_dict(json_dict)

class COCO_Image(BaseStructObject['COCO_License']):
    def __init__(
        self, license_id: int, file_name: str, coco_url: str,
        height: int, width: int, date_captured: str, flickr_url: str, id: int
    ):
        self.license_id = license_id
        self.file_name = file_name
        self.coco_url = coco_url
        self.height = height
        self.width = width
        self.date_captured = date_captured
        self.flickr_url = flickr_url
        self.id = id

    def __str__(self):
        print_str = "========================\n"
        print_str += f"license_id:\n\t{self.license_id}\n"
        print_str += f"file_name:\n\t{self.file_name}\n"
        print_str += f"coco_url:\n\t{self.coco_url}\n"
        print_str += f"height:\n\t{self.height}\n"
        print_str += f"width:\n\t{self.width}\n"
        print_str += f"date_captured:\n\t{self.date_captured}\n"
        print_str += f"flickr_url:\n\t{self.flickr_url}\n"
        print_str += f"id:\n\t{self.id}\n"
        return print_str

    def is_equal_to(
        self, other: COCO_Image,
        exclude_id: bool=True, exclude_date_captured: bool=False
    ) -> bool:
        result = True
        result = result and self.file_name == other.file_name
        result = result and self.coco_url == other.coco_url
        result = result and self.height == other.height
        result = result and self.width == other.width
        result = result and self.flickr_url == other.flickr_url
        if not exclude_date_captured:
            result = result and self.date_captured == other.date_captured
        if not exclude_id:
            result = result and self.id == other.id
            result = result and self.license_id == other.license_id
        return result

    def to_dict(self) -> dict:
        return {
            'license': self.license_id,
            'file_name': self.file_name,
            'coco_url': self.coco_url,
            'height': self.height,
            'width': self.width,
            'date_captured': self.date_captured,
            'flickr_url': self.flickr_url,
            'id': self.id
        }

    @classmethod
    def from_dict(cls, image_dict: dict) -> COCO_Image:
        check_required_keys(
            image_dict,
            required_keys=[
                'license', 'file_name', 'coco_url',
                'height', 'width', 'date_captured',
                'id'
            ]
        )
        return COCO_Image(
            license_id=image_dict['license'],
            file_name=image_dict['file_name'],
            coco_url=image_dict['coco_url'],
            height=image_dict['height'],
            width=image_dict['width'],
            date_captured=image_dict['date_captured'],
            flickr_url=image_dict['flickr_url'] if 'flickr_url' in image_dict else None,
            id=image_dict['id'],
        )

    @classmethod
    def from_img_path(self, img_path: str, license_id: int, image_id: int) -> COCO_Image:
        check_file_exists(img_path)
        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]
        return COCO_Image(
            license_id=license_id,
            file_name=get_filename(img_path),
            coco_url=img_path,
            height=img_h,
            width=img_w,
            date_captured=get_ctime(img_path),
            flickr_url=None,
            id=image_id
        )

    @classmethod
    def load_from_path(cls, json_path: str) -> COCO_Image:
        check_file_exists(json_path)
        json_dict = json.load(open(json_path, 'r'))
        return COCO_Image.from_dict(json_dict)

class COCO_Annotation(BaseStructObject['COCO_License']):
    def __init__(
        self,
        segmentation: Segmentation, num_keypoints: int, area: float, iscrowd: int,
        keypoints: Keypoint2D_List, image_id: int, bbox: BBox, category_id: int,
        id: int,
        keypoints_3d: Keypoint3D_List=None, camera: Camera=None
    ):
        self.segmentation = segmentation
        self.num_keypoints = num_keypoints
        self.area = area
        self.iscrowd = iscrowd
        self.keypoints = keypoints
        self.image_id = image_id
        self.bbox = bbox
        self.category_id = category_id
        self.id = id
        self.keypoints_3d = keypoints_3d
        self.camera = camera

    def __str__(self) -> str:
        print_str = 'COCO_Annotation'
        indent = 1
        print_str += '\n'
        print_str += '\t'*indent + f'segmentation: {self.segmentation}'
        print_str += '\n'
        print_str += '\t'*indent + f'num_keypoints: {self.num_keypoints}'
        print_str += '\n'
        print_str += '\t'*indent + f'area: {self.area}'
        print_str += '\n'
        print_str += '\t'*indent + f'iscrowd: {self.iscrowd}'
        print_str += '\n'
        print_str += '\t'*indent + f'keypoints: {self.keypoints}'
        print_str += '\n'
        print_str += '\t'*indent + f'image_id: {self.image_id}'
        print_str += '\n'
        print_str += '\t'*indent + f'bbox: {self.bbox}'
        print_str += '\n'
        print_str += '\t'*indent + f'category_id: {self.category_id}'
        print_str += '\n'
        print_str += '\t'*indent + f'id: {self.id}'
        print_str += '\n'
        print_str += '\t'*indent + f'keypoints_3d: {self.keypoints_3d}'
        print_str += '\n'
        print_str += '\t'*indent + f'camera: {self.camera}'
        return print_str

    @classmethod
    def simple_constructor(
        cls,
        image_id: int, category_id: int, id: int,
        segmentation: Segmentation=Segmentation([]),
        bbox: BBox=None,
        keypoints: Keypoint2D_List=Keypoint2D_List([]),
        iscrowd: int=0,
        keypoints_3d: Keypoint3D_List=None, camera: Camera=None
    ) -> COCO_Annotation:
        if bbox is None:
            if len(segmentation) > 0:
                seg_bbox_list = segmentation.to_bbox()
                seg_bbox_xmin = min([seg_bbox.xmin for seg_bbox in seg_bbox_list])
                seg_bbox_ymin = min([seg_bbox.ymin for seg_bbox in seg_bbox_list])
                seg_bbox_xmax = max([seg_bbox.xmax for seg_bbox in seg_bbox_list])
                seg_bbox_ymax = max([seg_bbox.ymax for seg_bbox in seg_bbox_list])
                result_bbox = BBox(xmin=seg_bbox_xmin, ymin=seg_bbox_ymin, xmax=seg_bbox_xmax, ymax=seg_bbox_ymax)
            else:
                logger.error(f'Need to specify either segmentation or bbox')
                raise Exception
        else:
            result_bbox = bbox
        
        return COCO_Annotation(
            segmentation=segmentation,
            num_keypoints=len(keypoints),
            area=result_bbox.area(),
            iscrowd=iscrowd,
            keypoints=keypoints,
            image_id=image_id,
            bbox=result_bbox,
            category_id=category_id,
            id=id,
            keypoints_3d=keypoints_3d,
            camera=camera
        )

    def to_dict(self) -> dict:
        data_dict = {
            'segmentation': self.segmentation.to_list(demarcation=False),
            'num_keypoints': self.num_keypoints,
            'area': self.area,
            'iscrowd': self.iscrowd,
            'keypoints': self.keypoints.to_list(demarcation=False),
            'image_id': self.image_id,
            'bbox': self.bbox.to_list(output_format='pminsize'),
            'category_id': self.category_id,
            'id': self.id
        }
        if self.keypoints_3d is not None:
            data_dict['keypoints_3d'] = self.keypoints_3d.to_list(demarcation=False)
        if self.camera is not None:
            data_dict['camera_params'] = self.camera.to_dict()
        return data_dict

    @classmethod
    def from_dict(cls, ann_dict: dict) -> COCO_Annotation:
        check_required_keys(
            ann_dict,
            required_keys=[
                'segmentation', 'num_keypoints', 'area',
                'iscrowd', 'keypoints', 'image_id',
                'bbox', 'category_id', 'id'
            ]
        )
        return COCO_Annotation(
            segmentation=Segmentation.from_list(ann_dict['segmentation'], demarcation=False),
            num_keypoints=ann_dict['num_keypoints'],
            area=ann_dict['area'],
            iscrowd=ann_dict['iscrowd'],
            keypoints=Keypoint2D_List.from_list(ann_dict['keypoints'], demarcation=False),
            image_id=ann_dict['image_id'],
            bbox=BBox.from_list(ann_dict['bbox'], input_format='pminsize'),
            category_id=ann_dict['category_id'],
            id=ann_dict['id'],
            keypoints_3d=Keypoint3D_List.from_list(ann_dict['keypoints_3d'], demarcation=False) if 'keypoints_3d' in ann_dict else None,
            camera=Camera.from_dict(ann_dict['camera_params']) if 'camera_params' in ann_dict else None
        )

    @classmethod
    def load_from_path(cls, json_path: str) -> COCO_Annotation:
        check_file_exists(json_path)
        json_dict = json.load(open(json_path, 'r'))
        return COCO_Annotation.from_dict(json_dict)

class COCO_Category(BaseStructObject['COCO_License']):
    def __init__(
        self, supercategory: str, id: int, name: str, keypoints: List[str], skeleton: List[list]
    ):
        self.supercategory = supercategory
        self.id = id
        self.name = name
        self.keypoints = keypoints
        self.skeleton = skeleton

    def __str__(self):
        print_str = "========================\n"
        print_str += f"supercategory:\n\t{self.supercategory}\n"
        print_str += f"id:\n\t{self.id}\n"
        print_str += f"name:\n\t{self.name}\n"
        print_str += f"keypoints:\n\t{self.keypoints}\n"
        print_str += f"skeleton:\n\t{self.skeleton}\n"
        return print_str

    def is_equal_to(
        self, other: COCO_Category,
        exclude_id: bool=True
    ) -> bool:
        result = True
        result = result and self.supercategory == other.supercategory
        result = result and self.name == other.name
        result = result and self.keypoints == other.keypoints
        result = result and self.skeleton == other.skeleton
        if not exclude_id:
            result = result and self.id == other.id
        return result

    @classmethod
    def from_dict(cls, category_dict: dict) -> COCO_Category:
        check_required_keys(
            category_dict,
            required_keys=[
                'supercategory', 'id', 'name',
                'keypoints', 'skeleton'
            ]
        )
        return COCO_Category(
            supercategory=category_dict['supercategory'],
            id=category_dict['id'],
            name=category_dict['name'],
            keypoints=category_dict['keypoints'],
            skeleton=category_dict['skeleton'],
        )

    @classmethod
    def from_label_skeleton(
        cls, supercategory: str, name: str, id: int,
        label_skeleton: List[list], skeleton_idx_offset: int=0
    ) -> COCO_Category:
        label_list = []
        for [start_label, end_label] in label_skeleton:
            check_type_from_list([start_label, end_label], valid_type_list=[str])
            if start_label not in label_list:
                label_list.append(start_label)
            if end_label not in label_list:
                label_list.append(end_label)
        label_list.sort()
        int_skeleton = []
        for [start_label, end_label] in label_skeleton:
            start_idx = label_list.index(start_label) + skeleton_idx_offset
            end_idx = label_list.index(end_label) + skeleton_idx_offset
            int_skeleton.append([start_idx, end_idx])
        return COCO_Category(
            supercategory=supercategory,
            id=id,
            name=name,
            keypoints=label_list,
            skeleton=int_skeleton
        )

    def get_label_skeleton(self, skeleton_idx_offset: int=0) -> list:
        str_skeleton = []
        for int_bone in self.skeleton:
            bone_start = self.keypoints[int_bone[0]-skeleton_idx_offset] 
            bone_end = self.keypoints[int_bone[1]-skeleton_idx_offset]
            str_bone = [bone_start, bone_end]
            str_skeleton.append(str_bone)
        return str_skeleton

    @classmethod
    def load_from_path(cls, json_path: str) -> COCO_Category:
        check_file_exists(json_path)
        json_dict = json.load(open(json_path, 'r'))
        return COCO_Category.from_dict(json_dict)