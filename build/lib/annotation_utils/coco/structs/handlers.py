from __future__ import annotations

from typing import List
import json
import operator
import random

from logger import logger
from common_utils.check_utils import check_type, check_type_from_list, \
    check_file_exists, check_value_from_list
from common_utils.path_utils import get_extension_from_filename
from common_utils.file_utils import file_exists

from .objects import COCO_License, COCO_Image, COCO_Annotation, COCO_Category
from ...base import BaseStructHandler

class COCO_License_Handler(BaseStructHandler['COCO_License_Handler', 'COCO_License']):
    def __init__(self, license_list: List[COCO_License]=None):
        super().__init__(obj_type=COCO_License, obj_list=license_list)
        self.license_list = self.obj_list

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

    def remove(self, id_list: List[int], verbose: bool=False):
        # TODO: Create a base class that inherits from BaseStruct that requires an id class parameter
        # This method could be added to the base handler of the resulting object class.
        if len(id_list) > 0:
            idx_list = list(range(len(self)))
            idx_list.reverse()
            for idx in idx_list:
                if self[idx].id in id_list:
                    del self[idx]
                    if verbose:
                        logger.info(f'Deleted License Id: {idx}')

    def remove_if_no_imgs(self, img_handler: COCO_Image_Handler, id_list: List[int]=None, verbose: bool=False):
        rm_license_id_list = []
        if id_list is not None:
            for license_id in id_list:
                imgs = img_handler.get_images_from_licenseIds([license_id])
                if len(imgs) == 0:
                    rm_license_id_list.append(license_id)
        else:
            for coco_license in self:
                imgs = img_handler.get_images_from_licenseIds([coco_license.id])
                if len(imgs) == 0:
                    rm_license_id_list.append(license_id)
        self.remove(rm_license_id_list, verbose=verbose)

class COCO_Image_Handler(BaseStructHandler['COCO_Image_Handler', 'COCO_Image']):
    def __init__(self, image_list: List[COCO_Image]=None):
        super().__init__(obj_type=COCO_Image, obj_list=image_list)
        self.image_list = self.obj_list

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

    def get_images_from_licenseIds(self, licenseIds: List[int]) -> List[COCO_Image]:
        return [x for x in self if x.license_id in licenseIds]

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

    def remove(self, id_list: List[int], verbose: bool=False):
        # TODO: Create a base class that inherits from BaseStruct that requires an id class parameter
        # This method could be added to the base handler of the resulting object class.
        if len(id_list) > 0:
            idx_list = list(range(len(self)))
            idx_list.reverse()
            for idx in idx_list:
                if self[idx].id in id_list:
                    del self[idx]
                    if verbose:
                        logger.info(f'Deleted Image Id: {idx}')
    
    def remove_if_no_anns(self, ann_handler: COCO_Annotation_Handler, license_handler: COCO_License_Handler=None, id_list: List[int]=None, verbose: bool=False):
        """Removes all of the COCO_Image objects in the handler that do not have any corresponding annotations.
        
        Arguments:
            ann_handler {COCO_Annotation_Handler} -- [Reference COCO Annotation Handler]
            license_handler {COCO_License_Handler} -- [
                The COCO License Handler that you would like to update according to image removals.
                If None, licenses will not be updated.
            ]
        
        Keyword Arguments:
            id_list {List[int]} -- [
                The image IDs that you would like to check.
                If None, all images are checked.
            ] (default: {None})
        """
        rm_image_id_list = []
        if id_list is not None:
            for image_id in id_list:
                anns = ann_handler.get_annotations_from_imgIds([image_id])
                if len(anns) == 0:
                    rm_image_id_list.append(image_id)
        else:
            for coco_image in self:
                anns = ann_handler.get_annotations_from_imgIds([coco_image.id])
                if len(anns) == 0:
                    rm_image_id_list.append(coco_image.id)
        self.remove(rm_image_id_list, verbose=verbose)
        if license_handler is not None:
            pending_license_id_list = [license.id for license in license_handler]
            license_handler.remove_if_no_imgs(img_handler=self, id_list=pending_license_id_list, verbose=verbose)

class COCO_Annotation_Handler(BaseStructHandler['COCO_Annotation_Handler', 'COCO_Annotation']):
    def __init__(self, annotation_list: List[COCO_Annotation]=None):
        super().__init__(obj_type=COCO_Annotation, obj_list=annotation_list)
        self.annotation_list = self.obj_list

    def get_annotations_from_annIds(self, annIds: list) -> List[COCO_Annotation]:		
        return [ann for ann in self if ann.id in annIds]		
        
    def get_annotations_from_imgIds(self, imgIds: list) -> List[COCO_Annotation]:		
        return [ann for ann in self if ann.image_id in imgIds]
    
    def get_annotations_from_catIds(self, catIds: list) -> List[COCO_Annotation]:
        return [ann for ann in self if ann.category_id in catIds]

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

    def remove(self, id_list: List[int], verbose: bool=False):
        # TODO: Create a base class that inherits from BaseStruct that requires an id class parameter
        # This method could be added to the base handler of the resulting object class.
        if len(id_list) > 0:
            idx_list = list(range(len(self)))
            idx_list.reverse()
            for idx in idx_list:
                if self[idx].id in id_list:
                    del self[idx]
                    if verbose:
                        logger.info(f'Deleted Annotation Id: {idx}')
    
    def remove_if_no_categories(
        self, cat_handler: COCO_Category_Handler,
        img_handler: COCO_Image_Handler=None, license_handler: COCO_License_Handler=None, id_list: List[int]=None, verbose: bool=False
    ):
        rm_ann_id_list = []
        pending_img_id_list = []
        existing_cat_id_list = [cat.id for cat in cat_handler]

        if id_list is not None:
            anns = self.get_annotations_from_annIds(id_list)
            for ann in anns:
                if ann.category_id not in existing_cat_id_list:
                    rm_ann_id_list.append(ann.id)
                    pending_img_id_list.append(ann.image_id)
        else:
            for ann in self:
                if ann.category_id not in existing_cat_id_list:
                    rm_ann_id_list.append(ann.id)
                    pending_img_id_list.append(ann.image_id)
        
        self.remove(rm_ann_id_list, verbose=verbose)
        if img_handler is not None:
            img_handler.remove_if_no_anns(ann_handler=self, license_handler=license_handler, id_list=pending_img_id_list, verbose=verbose)

class COCO_Category_Handler(BaseStructHandler['COCO_Category_Handler', 'COCO_Category']):
    def __init__(self, category_list: List[COCO_Category]=None):
        super().__init__(obj_type=COCO_Category, obj_list=category_list)
        self.category_list = self.obj_list

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
    
    def remove(self, id_list: List[int], verbose: bool=False):
        # TODO: Create a base class that inherits from BaseStruct that requires an id class parameter
        # This method could be added to the base handler of the resulting object class.
        if len(id_list) > 0:
            idx_list = list(range(len(self)))
            idx_list.reverse()
            for idx in idx_list:
                if self[idx].id in id_list:
                    del self[idx]
                    if verbose:
                        logger.info(f'Deleted Category Id: {idx}')
    
    def remove_by_name(
        self, names: List[str],
        ann_handler: COCO_Annotation_Handler=None, img_handler: COCO_Image_Handler=None, license_handler: COCO_License_Handler=None,
        verbose: bool=False
    ):
        existing_category_names = [coco_cat.name for coco_cat in self]
        check_value_from_list(names, valid_value_list=existing_category_names)
        rm_ids = [coco_cat.id for coco_cat in self if coco_cat.name in names]
        self.remove(rm_ids, verbose=verbose)
        if ann_handler is not None:
            ann_handler.remove_if_no_categories(
                cat_handler=self,
                img_handler=img_handler,
                license_handler=license_handler,
                id_list=None,
                verbose=verbose
            )