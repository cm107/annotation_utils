from __future__ import annotations

from typing import List
import json
import operator
import random

from logger import logger
from common_utils.check_utils import check_type, check_type_from_list, check_file_exists
from common_utils.path_utils import get_extension_from_filename
from common_utils.file_utils import file_exists

from .objects import COCO_License, COCO_Image, COCO_Annotation, COCO_Category
from ....base import BaseStructHandler

class COCO_License_Handler(BaseStructHandler['COCO_License_Handler', 'COCO_License']):
    def __init__(self, license_list: List[COCO_License]=None):
        super().__init__(obj_type=COCO_License, obj_list=license_list)
        self.license_list = self.obj_list

    def get_license_from_id(self, id: int) -> COCO_License:
        """ TODO: Delete all references to this method. """
        logger.warning(f'get_license_from_id is decapricated.')
        logger.warning(f'Please use get_obj_from_id instead.')
        return self.get_obj_from_id(id=id)

    @classmethod
    def from_dict_list(cls, dict_list: List[dict]) -> COCO_License_Handler:
        return COCO_License_Handler(
            license_list=[COCO_License.from_dict(license_dict) for license_dict in dict_list]
        )

    @classmethod
    def load_from_path(cls, json_path: str) -> COCO_License_Handler:
        check_file_exists(json_path)
        json_data = json.load(open(json_path, 'r'))
        return COCO_License_Handler.from_dict_list(json_data)

class COCO_Image_Handler(BaseStructHandler['COCO_Image_Handler', 'COCO_Image']):
    def __init__(self, image_list: List[COCO_Image]=None):
        super().__init__(obj_type=COCO_Image, obj_list=image_list)
        self.image_list = self.obj_list

    def get_image_from_id(self, id: int) -> COCO_Image:
        """ TODO: Delete all references to this method. """
        logger.warning(f'get_image_from_id is decapricated.')
        logger.warning(f'Please use get_obj_from_id instead.')
        return self.get_obj_from_id(id=id)

    def get_images_from_file_name(self, file_name: str) -> List[COCO_Image]:
        return [coco_image for coco_image in self if file_name == coco_image.file_name]

    def get_images_from_coco_url(self, coco_url: str) -> List[COCO_Image]:
        return [coco_image for coco_image in self if coco_url == coco_image.coco_url]

    def get_images_from_flickr_url(self, flickr_url: str) -> List[COCO_Image]:
        return [coco_image for coco_image in self if flickr_url == coco_image.flickr_url]

    def get_extensions(self) -> List[str]:
        extension_list = []
        for coco_image in self:
            extension = get_extension_from_filename(coco_image.file_name)
            if extension not in extension_list:
                extension_list.append(extension)
        return extension_list

    def get_images_from_imgIds(self, imgIds: list) -> List[COCO_Image]:		
	        return [x for x in self if x.id in imgIds]

    @classmethod
    def from_dict_list(cls, dict_list: List[dict]) -> COCO_Image_Handler:
        return COCO_Image_Handler(
            image_list=[COCO_Image.from_dict(image_dict) for image_dict in dict_list]
        )

    @classmethod
    def load_from_path(cls, json_path: str) -> COCO_Image_Handler:
        check_file_exists(json_path)
        json_data = json.load(open(json_path, 'r'))
        return COCO_Image_Handler.from_dict_list(json_data)

class COCO_Annotation_Handler(BaseStructHandler['COCO_Annotation_Handler', 'COCO_Annotation']):
    def __init__(self, annotation_list: List[COCO_Annotation]=None):
        super().__init__(obj_type=COCO_Annotation, obj_list=annotation_list)
        self.annotation_list = self.obj_list

    def get_annotation_from_id(self, id: int) -> COCO_Annotation:
        """ TODO: Delete all references to this method. """
        logger.warning(f'get_annotation_from_id is decapricated.')
        logger.warning(f'Please use get_obj_from_id instead.')
        return self.get_obj_from_id(id=id)

    def get_annotations_from_annIds(self, annIds: list) -> List[COCO_Annotation]:		
        return [ann for ann in self if ann.id in annIds]		
        
    def get_annotations_from_imgIds(self, imgIds: list) -> List[COCO_Annotation]:		
        return [ann for ann in self if ann.image_id in imgIds]

    def to_dict_list(self, strict: bool=True) -> List[dict]:
        return [item.to_dict(strict=strict) for item in self]

    def save_to_path(self, save_path: str, overwrite: bool=False, strict: bool=True):
        if file_exists(save_path) and not overwrite:
            logger.error(f'File already exists at save_path: {save_path}')
            raise Exception
        json_data = self.to_dict_list(strict=strict)
        json.dump(json_data, open(save_path, 'w'), indent=2, ensure_ascii=False)

    @classmethod
    def from_dict_list(cls, dict_list: List[dict], strict: bool=True) -> COCO_Annotation_Handler:
        return COCO_Annotation_Handler(
            annotation_list=[COCO_Annotation.from_dict(ann_dict, strict=strict) for ann_dict in dict_list]
        )

    @classmethod
    def load_from_path(cls, json_path: str, strict: bool=True) -> COCO_Annotation_Handler:
        check_file_exists(json_path)
        json_data = json.load(open(json_path, 'r'))
        return COCO_Annotation_Handler.from_dict_list(json_data, strict=strict)

class COCO_Category_Handler(BaseStructHandler['COCO_Category_Handler', 'COCO_Category']):
    def __init__(self, category_list: List[COCO_Category]=None):
        super().__init__(obj_type=COCO_Category, obj_list=category_list)
        self.category_list = self.obj_list

    def get_category_from_id(self, id: int) -> COCO_Category:
        """ TODO: Delete all references to this method. """
        logger.warning(f'get_category_from_id is decapricated.')
        logger.warning(f'Please use get_obj_from_id instead.')
        return self.get_obj_from_id(id=id)

    def get_categories_from_name(self, name: str) -> List[COCO_Category]:		
        return [cat for cat in self if cat.name == name]

    def get_unique_category_from_name(self, name: str) -> COCO_Category:
        found_categories = self.get_categories_from_name(name)
        if len(found_categories) == 0:
            logger.error(f"Couldn't find any categories by the name: {name}")
            raise Exception
        elif len(found_categories) > 1:
            logger.error(f"Found {len(found_categories)} categories with the name {name}")
            logger.error(f"Found Categories:")
            for category in found_categories:
                logger.error(category)
            raise Exception
        return found_categories[0]

    def get_skeleton_from_name(self, name: str) -> (list, list):
        unique_category = self.get_unique_category_from_name(name)
        skeleton = unique_category.skeleton
        label_skeleton = unique_category.get_label_skeleton()
        return skeleton, label_skeleton

    def to_dict_list(self, strict: bool=True) -> List[dict]:
        return [item.to_dict(strict=strict) for item in self]

    def save_to_path(self, save_path: str, overwrite: bool=False, strict: bool=True):
        if file_exists(save_path) and not overwrite:
            logger.error(f'File already exists at save_path: {save_path}')
            raise Exception
        json_data = self.to_dict_list(strict=strict)
        json.dump(json_data, open(save_path, 'w'), indent=2, ensure_ascii=False)

    @classmethod
    def from_dict_list(cls, dict_list: List[dict], strict: bool=True) -> COCO_Category_Handler:
        return COCO_Category_Handler(
            category_list=[COCO_Category.from_dict(cat_dict, strict=strict) for cat_dict in dict_list]
        )

    @classmethod
    def load_from_path(cls, json_path: str, strict: bool=True) -> COCO_Category_Handler:
        check_file_exists(json_path)
        json_data = json.load(open(json_path, 'r'))
        return COCO_Category_Handler.from_dict_list(json_data, strict=strict)