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

class COCO_License_Handler:
    def __init__(self, license_list: List[COCO_License]=None):
        if license_list is not None:
            check_type_from_list(license_list, valid_type_list=[COCO_License])
        self.license_list = license_list if license_list is not None else []

    def __str__(self):
        print_str = ""
        for coco_license in self.license_list:
            print_str += f"{coco_license}\n"

        return print_str

    def __repr__(self):
        return self.__str__()

    def __len__(self) -> int:
        return len(self.license_list)

    def __getitem__(self, idx: int) -> COCO_License:
        if type(idx) is int:
            if len(self.license_list) == 0:
                logger.error(f"COCO_License_Handler is empty.")
                raise IndexError
            elif idx < 0 or idx >= len(self.license_list):
                logger.error(f"Index out of range: {idx}")
                raise IndexError
            else:
                return self.license_list[idx]
        elif type(idx) is slice:
            return self.license_list[idx.start:idx.stop:idx.step]
        else:
            logger.error(f'Expected int or slice. Got type(idx)={type(idx)}')
            raise TypeError

    def __setitem__(self, idx: int, value: COCO_License):
        check_type(value, valid_type_list=[COCO_License])
        if type(idx) is int:
            self.license_list[idx] = value
        elif type(idx) is slice:
            self.license_list[idx.start:idx.stop:idx.step] = value
        else:
            logger.error(f'Expected int or slice. Got type(idx)={type(idx)}')
            raise TypeError

    def __delitem__(self, idx):
        if type(idx) is int:
            if len(self.license_list) == 0:
                logger.error(f"COCO_License_Handler is empty.")
                raise IndexError
            elif idx < 0 or idx >= len(self.license_list):
                logger.error(f"Index out of range: {idx}")
                raise IndexError
            else:
                del self.license_list[idx]
        elif type(idx) is slice:
            del self.license_list[idx.start:idx.stop:idx.step]
        else:
            logger.error(f'Expected int or slice. Got type(idx)={type(idx)}')
            raise TypeError

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self) -> COCO_License:
        if self.n < len(self.license_list):
            result = self.license_list[self.n]
            self.n += 1
            return result
        else:
            raise StopIteration

    def copy(self) -> COCO_License_Handler:
        return COCO_License_Handler(license_list=self.license_list.copy())

    def append(self, item: COCO_License):
        check_type(item, valid_type_list=[COCO_License])
        self.license_list.append(item)

    def sort(self, attr_name: str, reverse: bool=False):
        if not hasattr(COCO_License, attr_name):
            logger.error(f"COCO_License class has not attribute: '{attr_name}'")
            if len(self) > 0:
                attr_list = list(self.license_list[0].__dict__.keys())    
                logger.error(f'Possible attribute names:')
                for name in attr_list:
                    logger.error(f'\t{name}')
                raise Exception
        self.license_list.sort(key=operator.attrgetter(attr_name), reverse=reverse)

    def shuffle(self):
        random.shuffle(self.license_list)

    def get_license_from_id(self, id: int) -> COCO_License:
        license_id_list = []
        for coco_license in self:
            if id == coco_license.id:
                return coco_license
            else:
                license_id_list.append(coco_license.id)
        license_id_list.sort()
        logger.error(f"Couldn't find coco_license with id={id}")
        logger.error(f"Possible ids: {license_id_list}")
        raise Exception

    def to_dict_list(self) -> List[dict]:
        return [item.to_dict() for item in self]

    @classmethod
    def from_dict_list(cls, dict_list: List[dict]) -> COCO_License_Handler:
        return COCO_License_Handler(
            license_list=[COCO_License.from_dict(license_dict) for license_dict in dict_list]
        )

    def save_to_path(self, save_path: str, overwrite: bool=False):
        if file_exists(save_path) and not overwrite:
            logger.error(f'File already exists at save_path: {save_path}')
            raise Exception
        json_data = self.to_dict_list()
        json.dump(json_data, open(save_path, 'w'), indent=2, ensure_ascii=False)

    @classmethod
    def load_from_path(cls, json_path: str) -> COCO_License_Handler:
        check_file_exists(json_path)
        json_data = json.load(open(json_path, 'r'))
        return COCO_License_Handler.from_dict_list(json_data)

class COCO_Image_Handler:
    def __init__(self, image_list: List[COCO_Image]=None):
        if image_list is not None:
            check_type_from_list(image_list, valid_type_list=[COCO_Image])
        self.image_list = image_list if image_list is not None else []

    def __str__(self):
        print_str = ""
        for image in self.image_list:
            print_str += f"{image}\n"

        return print_str

    def __repr__(self):
        return self.__str__()

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, idx: int) -> COCO_Image:
        if type(idx) is int:
            if len(self.image_list) == 0:
                logger.error(f"COCO_Image_Handler is empty.")
                raise IndexError
            elif idx < 0 or idx >= len(self.image_list):
                logger.error(f"Index out of range: {idx}")
                raise IndexError
            else:
                return self.image_list[idx]
        elif type(idx) is slice:
            return self.image_list[idx.start:idx.stop:idx.step]
        else:
            logger.error(f'Expected int or slice. Got type(idx)={type(idx)}')
            raise TypeError

    def __setitem__(self, idx: int, value: COCO_Image):
        check_type(value, valid_type_list=[COCO_Image])
        if type(idx) is int:
            self.image_list[idx] = value
        elif type(idx) is slice:
            self.image_list[idx.start:idx.stop:idx.step] = value
        else:
            logger.error(f'Expected int or slice. Got type(idx)={type(idx)}')
            raise TypeError

    def __delitem__(self, idx: int):
        if type(idx) is int:
            if len(self.image_list) == 0:
                logger.error(f"COCO_Image_Handler is empty.")
                raise IndexError
            elif idx < 0 or idx >= len(self.image_list):
                logger.error(f"Index out of range: {idx}")
                raise IndexError
            else:
                del self.image_list[idx]
        elif type(idx) is slice:
            del self.image_list[idx.start:idx.stop:idx.step]
        else:
            logger.error(f'Expected int or slice. Got type(idx)={type(idx)}')
            raise TypeError

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self) -> COCO_Image:
        if self.n < len(self.image_list):
            result = self.image_list[self.n]
            self.n += 1
            return result
        else:
            raise StopIteration

    def copy(self) -> COCO_Image_Handler:
        return COCO_Image_Handler(
            image_list=self.image_list.copy()
        )

    def append(self, item: COCO_Image):
        check_type(item, valid_type_list=[COCO_Image])
        self.image_list.append(item)

    def sort(self, attr_name: str, reverse: bool=False):
        if len(self) > 0:
            if not hasattr(self.image_list[0], attr_name):
                logger.error(f"COCO_Image class has not attribute: '{attr_name}'")
                attr_list = list(self.image_list[0].__dict__.keys())    
                logger.error(f'Possible attribute names:')
                for name in attr_list:
                    logger.error(f'\t{name}')
                raise Exception
            self.image_list.sort(key=operator.attrgetter(attr_name), reverse=reverse)
        else:
            logger.error(f'COCO_Image_Handler is empty')
            raise Exception

    def shuffle(self):
        random.shuffle(self.image_list)

    def get_image_from_id(self, id: int) -> COCO_Image:
        image_id_list = []
        for coco_image in self:
            if id == coco_image.id:
                return coco_image
            else:
                image_id_list.append(coco_image.id)
        image_id_list.sort()
        logger.error(f"Couldn't find coco_image with id={id}")
        logger.error(f"Possible ids: {image_id_list}")
        raise Exception

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

    def to_dict_list(self) -> List[dict]:
        return [item.to_dict() for item in self]

    @classmethod
    def from_dict_list(cls, dict_list: List[dict]) -> COCO_Image_Handler:
        return COCO_Image_Handler(
            image_list=[COCO_Image.from_dict(image_dict) for image_dict in dict_list]
        )

    def save_to_path(self, save_path: str, overwrite: bool=False):
        if file_exists(save_path) and not overwrite:
            logger.error(f'File already exists at save_path: {save_path}')
            raise Exception
        json_data = self.to_dict_list()
        json.dump(json_data, open(save_path, 'w'), indent=2, ensure_ascii=False)

    @classmethod
    def load_from_path(cls, json_path: str) -> COCO_Image_Handler:
        check_file_exists(json_path)
        json_data = json.load(open(json_path, 'r'))
        return COCO_Image_Handler.from_dict_list(json_data)

class COCO_Annotation_Handler:
    def __init__(self, annotation_list: List[COCO_Annotation]=None):
        if annotation_list is not None:
            check_type_from_list(annotation_list, valid_type_list=[COCO_Annotation])
        self.annotation_list = annotation_list if annotation_list is not None else []

    def __str__(self):
        print_str = ""
        for annotation in self.annotation_list:
            print_str += f"{annotation}\n"

        return print_str

    def __repr__(self):
        return self.__str__()

    def __len__(self) -> int:
        return len(self.annotation_list)

    def __getitem__(self, idx: int) -> COCO_Annotation:
        if type(idx) is int:
            if len(self.annotation_list) == 0:
                logger.error(f"COCO_Annotation_Handler is empty.")
                raise IndexError
            elif idx < 0 or idx >= len(self.annotation_list):
                logger.error(f"Index out of range: {idx}")
                raise IndexError
            else:
                return self.annotation_list[idx]
        elif type(idx) is slice:
            return self.annotation_list[idx.start:idx.stop:idx.step]
        else:
            logger.error(f'Expected int or slice. Got type(idx)={type(idx)}')
            raise TypeError

    def __setitem__(self, idx: int, value: COCO_Annotation):
        check_type(value, valid_type_list=[COCO_Annotation])
        if type(idx) is int:
            self.annotation_list[idx] = value
        elif type(idx) is slice:
            self.annotation_list[idx.start:idx.stop:idx.step] = value
        else:
            logger.error(f'Expected int or slice. Got type(idx)={type(idx)}')
            raise TypeError

    def __delitem__(self, idx: int):
        if type(idx) is int:
            if len(self.annotation_list) == 0:
                logger.error(f"COCO_Annotation_Handler is empty.")
                raise IndexError
            elif idx < 0 or idx >= len(self.annotation_list):
                logger.error(f"Index out of range: {idx}")
                raise IndexError
            else:
                del self.annotation_list[idx]
        elif type(idx) is slice:
            del self.annotation_list[idx.start:idx.stop:idx.step]
        else:
            logger.error(f'Expected int or slice. Got type(idx)={type(idx)}')
            raise TypeError

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self) -> COCO_Annotation:
        if self.n < len(self.annotation_list):
            result = self.annotation_list[self.n]
            self.n += 1
            return result
        else:
            raise StopIteration

    def copy(self) -> COCO_Annotation_Handler:
        return COCO_Annotation_Handler(
            annotation_list=self.annotation_list.copy()
        )

    def append(self, item: COCO_Annotation):
        check_type(item, valid_type_list=[COCO_Annotation])
        self.annotation_list.append(item)

    def sort(self, attr_name: str, reverse: bool=False):
        if not hasattr(COCO_Annotation, attr_name):
            logger.error(f"COCO_Annotation class has not attribute: '{attr_name}'")
            if len(self) > 0:
                attr_list = list(self.annotation_list[0].__dict__.keys())    
                logger.error(f'Possible attribute names:')
                for name in attr_list:
                    logger.error(f'\t{name}')
                raise Exception
        self.annotation_list.sort(key=operator.attrgetter(attr_name), reverse=reverse)

    def shuffle(self):
        random.shuffle(self.annotation_list)

    def get_annotation_from_id(self, id: int) -> COCO_Annotation:
        annotation_id_list = []
        for coco_annotation in self:
            if id == coco_annotation.id:
                return coco_annotation
            else:
                annotation_id_list.append(coco_annotation.id)
        annotation_id_list.sort()
        logger.error(f"Couldn't find coco_annotation with id={id}")
        logger.error(f"Possible ids: {annotation_id_list}")
        raise Exception

    def get_annotations_from_annIds(self, annIds: list) -> List[COCO_Annotation]:		
        return [ann for ann in self if ann.id in annIds]		
        
    def get_annotations_from_imgIds(self, imgIds: list) -> List[COCO_Annotation]:		
        return [ann for ann in self if ann.image_id in imgIds]

    def to_dict_list(self) -> List[dict]:
        return [item.to_dict() for item in self]

    @classmethod
    def from_dict_list(cls, dict_list: List[dict]) -> COCO_Annotation_Handler:
        return COCO_Annotation_Handler(
            annotation_list=[COCO_Annotation.from_dict(ann_dict) for ann_dict in dict_list]
        )

    def save_to_path(self, save_path: str, overwrite: bool=False):
        if file_exists(save_path) and not overwrite:
            logger.error(f'File already exists at save_path: {save_path}')
            raise Exception
        json_data = self.to_dict_list()
        json.dump(json_data, open(save_path, 'w'), indent=2, ensure_ascii=False)

    @classmethod
    def load_from_path(cls, json_path: str) -> COCO_Annotation_Handler:
        check_file_exists(json_path)
        json_data = json.load(open(json_path, 'r'))
        return COCO_Annotation_Handler.from_dict_list(json_data)

class COCO_Category_Handler:
    def __init__(self, category_list: List[COCO_Category]=None):
        if category_list is not None:
            check_type_from_list(category_list, valid_type_list=[COCO_Category])
            self.category_list = category_list
        else:
            self.category_list = []

    def __str__(self):
        print_str = ""
        for category in self.category_list:
            print_str += f"{category}\n"

        return print_str

    def __repr__(self):
        return self.__str__()

    def __len__(self) -> int:
        return len(self.category_list)

    def __getitem__(self, idx: int) -> COCO_Category:
        if type(idx) is int:
            if len(self.category_list) == 0:
                logger.error(f"COCO_Category_Handler is empty.")
                raise IndexError
            elif idx < 0 or idx >= len(self.category_list):
                logger.error(f"Index out of range: {idx}")
                raise IndexError
            else:
                return self.category_list[idx]
        elif type(idx) is slice:
            return self.category_list[idx.start:idx.stop:idx.step]
        else:
            logger.error(f'Expected int or slice. Got type(idx)={type(idx)}')
            raise TypeError

    def __setitem__(self, idx: int, value: COCO_Category):
        check_type(value, valid_type_list=[COCO_Category])
        if type(idx) is int:
            self.category_list[idx] = value
        elif type(idx) is slice:
            self.category_list[idx.start:idx.stop:idx.step] = value
        else:
            logger.error(f'Expected int or slice. Got type(idx)={type(idx)}')
            raise TypeError

    def __delitem__(self, idx: int):
        if type(idx) is int:
            if len(self.category_list) == 0:
                logger.error(f"COCO_Category_Handler is empty.")
                raise IndexError
            elif idx < 0 or idx >= len(self.category_list):
                logger.error(f"Index out of range: {idx}")
                raise IndexError
            else:
                del self.category_list[idx]
        elif type(idx) is slice:
            del self.category_list[idx.start:idx.stop:idx.step]
        else:
            logger.error(f'Expected int or slice. Got type(idx)={type(idx)}')
            raise TypeError

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self) -> COCO_Category:
        if self.n < len(self.category_list):
            result = self.category_list[self.n]
            self.n += 1
            return result
        else:
            raise StopIteration

    def copy(self) -> COCO_Category_Handler:
        return COCO_Category_Handler(
            category_list=self.category_list.copy()
        )

    def append(self, item: COCO_Category):
        check_type(item, valid_type_list=[COCO_Category])
        self.category_list.append(item)

    def sort(self, attr_name: str, reverse: bool=False):
        if not hasattr(COCO_Category, attr_name):
            logger.error(f"COCO_Category class has not attribute: '{attr_name}'")
            if len(self) > 0:
                attr_list = list(self.category_list[0].__dict__.keys())    
                logger.error(f'Possible attribute names:')
                for name in attr_list:
                    logger.error(f'\t{name}')
                raise Exception
        self.category_list.sort(key=operator.attrgetter(attr_name), reverse=reverse)

    def shuffle(self):
        random.shuffle(self.category_list)

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

    def get_category_from_id(self, id: int) -> COCO_Category:
        category_id_list = []
        for coco_category in self.category_list:
            if id == coco_category.id:
                return coco_category
            else:
                category_id_list.append(coco_category.id)
        category_id_list.sort()
        logger.error(f"Couldn't find coco_category with id={id}")
        logger.error(f"Possible ids: {category_id_list}")
        raise Exception

    def get_skeleton_from_name(self, name: str) -> (list, list):
        unique_category = self.get_unique_category_from_name(name)
        skeleton = unique_category.skeleton
        label_skeleton = unique_category.get_label_skeleton()
        return skeleton, label_skeleton

    def to_dict_list(self) -> List[dict]:
        return [item.to_dict() for item in self]

    @classmethod
    def from_dict_list(cls, dict_list: List[dict]) -> COCO_Category_Handler:
        return COCO_Category_Handler(
            category_list=[COCO_Category.from_dict(cat_dict) for cat_dict in dict_list]
        )

    def save_to_path(self, save_path: str, overwrite: bool=False):
        if file_exists(save_path) and not overwrite:
            logger.error(f'File already exists at save_path: {save_path}')
            raise Exception
        json_data = self.to_dict_list()
        json.dump(json_data, open(save_path, 'w'), indent=2, ensure_ascii=False)

    @classmethod
    def load_from_path(cls, json_path: str) -> COCO_Category_Handler:
        check_file_exists(json_path)
        json_data = json.load(open(json_path, 'r'))
        return COCO_Category_Handler.from_dict_list(json_data)