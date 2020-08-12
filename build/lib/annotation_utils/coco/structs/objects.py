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

from ..camera import Camera
# from ...base import BaseStructObject
from common_utils.base.basic import BasicLoadableIdObject, BasicLoadableObject

class COCO_Info(BasicLoadableObject['COCO_Info']):
    def __init__(
        self,
        description: str="Dataset Created Using annotation_utils",
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

class COCO_License(BasicLoadableIdObject['COCO_License']):
    def __init__(self, url: str, id: int, name: str):
        super().__init__(id=id)
        self.url = url
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

class COCO_Image(BasicLoadableIdObject['COCO_Image']):
    def __init__(
        self, license_id: int, file_name: str, coco_url: str,
        height: int, width: int, date_captured: str, flickr_url: str, id: int
    ):
        super().__init__(id=id)
        self.license_id = license_id
        self.file_name = file_name
        self.coco_url = coco_url
        self.height = height
        self.width = width
        self.date_captured = date_captured
        self.flickr_url = flickr_url

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

class COCO_Annotation(BasicLoadableIdObject['COCO_Annotation']):
    def __init__(
        self,
        id: int, category_id: int, image_id: int, # Standard Required
        segmentation: Segmentation=None, # Standard Optional
        bbox: BBox=None, area: float=None,
        keypoints: Keypoint2D_List=None, num_keypoints: int=None,
        iscrowd: int=0,
        keypoints_3d: Keypoint3D_List=None, camera: Camera=None # Custom Optional
    ):
        super().__init__(id=id)
        # Standard Required
        self.category_id = category_id
        self.image_id = image_id

        # Standard Optional
        self.segmentation = segmentation if segmentation is not None else Segmentation([])
        if bbox is not None:
            self.bbox = bbox
        else:
            if len(self.segmentation) > 0:
                self.bbox = self.segmentation.to_bbox()
            else:
                logger.error(f'A COCO_Annotation needs to be given either a bbox or a non-empty segmentation at the very least to make a valid annotation.')
                logger.error(f'id: {id}, category_id: {category_id}, image_id: {image_id}')
                raise Exception
        self.area = area
        self.keypoints = keypoints if keypoints is not None else Keypoint2D_List([])
        self.num_keypoints = num_keypoints if num_keypoints is not None else len(self.keypoints)
        self.iscrowd = iscrowd

        # Custom Optional
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

    def to_dict(self, strict: bool=True) -> dict:
        if strict:
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
        else:
            data_dict = {
                'bbox': self.bbox.to_list(output_format='pminsize'),
                'area': self.area,
                'iscrowd': self.iscrowd,
                'image_id': self.image_id,
                'category_id': self.category_id,
                'id': self.id
            }
            if len(self.segmentation) > 0:
                data_dict['segmentation'] = self.segmentation.to_list(demarcation=False)
            if len(self.keypoints) > 0:
                data_dict['keypoints'] = self.keypoints.to_list(demarcation=False)
                data_dict['num_keypoints'] = self.num_keypoints

            if self.keypoints_3d is not None:
                data_dict['keypoints_3d'] = self.keypoints_3d.to_list(demarcation=False)
            if self.camera is not None:
                data_dict['camera_params'] = self.camera.to_dict()
            return data_dict

    def save_to_path(self, save_path: str, overwrite: bool=False, strict: bool=True):
        if file_exists(save_path) and not overwrite:
            logger.error(f'File already exists at save_path: {save_path}')
            raise Exception
        json_dict = self.to_dict(strict=strict)
        json.dump(json_dict, open(save_path, 'w'), indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, ann_dict: dict, strict: bool=True) -> COCO_Annotation:
        if strict:
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
        else:
            check_required_keys(
                ann_dict,
                required_keys=[
                    'id', 'category_id', 'image_id'
                ]
            )
            return COCO_Annotation(
                segmentation=Segmentation.from_list(ann_dict['segmentation'], demarcation=False) if 'segmentation' in ann_dict else None,
                num_keypoints=ann_dict['num_keypoints'] if 'num_keypoints' in ann_dict else None,
                area=ann_dict['area'] if 'area' in ann_dict else None,
                iscrowd=ann_dict['iscrowd'] if 'iscrowd' in ann_dict else None,
                keypoints=Keypoint2D_List.from_list(ann_dict['keypoints'], demarcation=False) if 'keypoints' in ann_dict else None,
                image_id=ann_dict['image_id'],
                bbox=BBox.from_list(ann_dict['bbox'], input_format='pminsize') if 'bbox' in ann_dict else None,
                category_id=ann_dict['category_id'],
                id=ann_dict['id'],
                keypoints_3d=Keypoint3D_List.from_list(ann_dict['keypoints_3d'], demarcation=False) if 'keypoints_3d' in ann_dict else None,
                camera=Camera.from_dict(ann_dict['camera_params']) if 'camera_params' in ann_dict else None
            )

    @classmethod
    def load_from_path(cls, json_path: str, strict: bool=True) -> COCO_Annotation:
        check_file_exists(json_path)
        json_dict = json.load(open(json_path, 'r'))
        return COCO_Annotation.from_dict(ann_dict=json_dict, strict=strict)

class COCO_Category(BasicLoadableIdObject['COCO_Category']):
    def __init__(
        self, id: int, supercategory: str=None, name: str=None, keypoints: List[str]=None, skeleton: List[list]=None
    ):
        super().__init__(id=id)

        # Standard Optional
        if supercategory is None and name is None:
            logger.error(f'Need to provide COCO_Category with either supercategory or name at the very least, but both are None.')
            logger.error(f'id: {id}')
            raise Exception
        supercategory0 = str(supercategory) if type(supercategory) in [int, float] else supercategory
        name0 = str(name) if type(name) in [int, float] else name
        self.supercategory = supercategory0 if supercategory0 is not None else name0
        self.name = name0 if name0 is not None else supercategory0
        self.keypoints = keypoints if keypoints is not None else []
        self.skeleton = skeleton if skeleton is not None else []

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

    def to_dict(self, strict: bool=True) -> dict:
        if strict:
            return self.__dict__
        else:
            result_dict = {
                'id': self.id,
                'name': self.name
            }
            if self.supercategory != self.name:
                result_dict['supercategory'] = self.supercategory
            if len(self.keypoints) > 0:
                result_dict['keypoints'] = self.keypoints
            if len(self.skeleton) > 0:
                result_dict['skeleton'] = self.skeleton
            return result_dict

    def save_to_path(self: T, save_path: str, overwrite: bool=False, strict: bool=True):
        if file_exists(save_path) and not overwrite:
            logger.error(f'File already exists at save_path: {save_path}')
            raise Exception
        json_dict = self.to_dict(strict=strict)
        json.dump(json_dict, open(save_path, 'w'), indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, category_dict: dict, strict: bool=True) -> COCO_Category:
        if strict:
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
        else:
            check_required_keys(
                category_dict,
                required_keys=[
                    'id'
                ]
            )
            return COCO_Category(
                id=category_dict['id'],
                supercategory=category_dict['supercategory'] if 'supercategory' in category_dict else None,
                name=category_dict['name'] if 'name' in category_dict else None,
                keypoints=category_dict['keypoints'] if 'keypoints' in category_dict else None,
                skeleton=category_dict['skeleton'] if 'skeleton' in category_dict else None
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
    def load_from_path(cls, json_path: str, strict: bool=True) -> COCO_Category:
        check_file_exists(json_path)
        json_dict = json.load(open(json_path, 'r'))
        return COCO_Category.from_dict(json_dict, strict=strict)