from __future__ import annotations
from typing import List
import labelme
import json
from tqdm import tqdm
import operator

from logger import logger
from common_utils.common_types.point import Point2D_List
from common_utils.check_utils import check_required_keys, check_type, check_file_exists, check_value, \
    check_dir_exists
from common_utils.path_utils import get_all_files_of_extension, get_rootname_from_path, get_filename, \
    get_dirpath_from_filepath
from common_utils.file_utils import delete_all_files_in_dir, make_dir_if_not_exists, file_exists, copy_file

# TODO: Inherit from basic

class LabelmeShape:
    def __init__(
        self,
        label: str, points: Point2D_List, shape_type: str,
        group_id: int=None, flags: dict={}
    ):
        self._check_valid(shape_type=shape_type, points=points)
        self.label = label
        self.points = points
        self.shape_type = shape_type
        self.group_id = group_id
        self.flags = flags

    @staticmethod
    def _check_valid(shape_type: str, points: Point2D_List):
        check_value(shape_type, valid_value_list=['polygon', 'rectangle', 'circle', 'line', 'point', 'linestrip'])
        if shape_type == 'polygon':
            if len(points) < 3:
                logger.error(f'Labelme polygon requires at least 3 points.')
                raise Exception
        elif shape_type == 'rectangle':
            if len(points) != 2:
                logger.error(f'Labelme rectangle requires exactly 2 points.')
                raise Exception
        elif shape_type == 'circle':
            if len(points) != 2:
                logger.error(f'Labelme circle requires exactly 2 points.')
                raise Exception
        elif shape_type == 'line':
            if len(points) != 2:
                logger.error(f'Labelme line requires exactly 2 points.')
                raise Exception
        elif shape_type == 'point':
            if len(points) != 1:
                logger.error(f'Labelme point requires exactly 1 points.')
                raise Exception
        elif shape_type == 'linestrip':
            if len(points) < 2:
                logger.error(f'Labelme linestrip requires at least 2 points.')
                raise Exception

    def to_dict(self) -> dict:
        return {
            'label': self.label,
            'points': self.points.to_list(demarcation=True),
            'shape_type': self.shape_type,
            'group_id': self.group_id,
            'flags': self.flags
        }

    @classmethod
    def from_dict(cls, shape_dict: dict) -> LabelmeShape:
        check_required_keys(shape_dict, required_keys=['label', 'points', 'shape_type', 'flags'])
        return LabelmeShape(
            label=shape_dict['label'],
            points=Point2D_List.from_list(value_list=shape_dict['points'], demarcation=True),
            shape_type=shape_dict['shape_type'],
            group_id=shape_dict['group_id'] if 'group_id' in shape_dict else 0,
            flags=shape_dict['flags']
        )

class LabelmeShapeHandler:
    def __init__(self, shape_list: List[LabelmeShape]=None):
        self.shape_list = shape_list if shape_list is not None else []

    def __len__(self) -> int:
        return len(self.shape_list)

    def __getitem__(self, idx: int) -> LabelmeShape:
        if len(self.shape_list) == 0:
            logger.error(f"LabelmeShapeHandler is empty.")
            raise IndexError
        elif idx < 0 or idx >= len(self.shape_list):
            logger.error(f"Index out of range: {idx}")
            raise IndexError
        else:
            return self.shape_list[idx]

    def __setitem__(self, idx: int, value: LabelmeShape):
        check_type(value, valid_type_list=[LabelmeShape])
        self.shape_list[idx] = value

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self) -> LabelmeShape:
        if self.n < len(self.shape_list):
            result = self.shape_list[self.n]
            self.n += 1
            return result
        else:
            raise StopIteration

    def append(self, shape: LabelmeShape):
        check_type(shape, valid_type_list=[LabelmeShape])
        self.shape_list.append(shape)

    def to_dict_list(self) -> List[dict]:
        return [shape.to_dict() for shape in self]

    @classmethod
    def from_dict_list(cls, dict_list: List[dict]) -> LabelmeShapeHandler:
        return LabelmeShapeHandler(
            shape_list=[LabelmeShape.from_dict(shape_dict) for shape_dict in dict_list]
        )

class LabelmeAnnotation:
    def __init__(
        self,
        img_path: str, img_h: int, img_w: int,
        version: str=labelme.__version__, flags: dict={},
        shapes: LabelmeShapeHandler=None,
        img_data: str=None
    ):
        self.img_path = img_path
        self.img_h, self.img_w = img_h, img_w
        self.version = version
        self.flags = flags
        self.shapes = shapes if shapes is not None else LabelmeShapeHandler()
        self.img_data = img_data

    def to_dict(self) -> dict:
        return {
            'version': self.version,
            'flags': self.flags,
            'shapes': self.shapes.to_dict_list(),
            'imagePath': self.img_path,
            'imageData': self.img_data,
            'imageHeight': self.img_h,
            'imageWidth': self.img_w
        }
    
    @classmethod
    def from_dict(cls, labelme_dict: dict) -> LabelmeAnnotation:
        check_required_keys(labelme_dict, required_keys=['version', 'flags', 'shapes', 'imagePath', 'imageData', 'imageHeight', 'imageWidth'])
        return LabelmeAnnotation(
            version=labelme_dict['version'],
            flags=labelme_dict['flags'],
            shapes=LabelmeShapeHandler.from_dict_list(labelme_dict['shapes']),
            img_path=labelme_dict['imagePath'],
            img_data=labelme_dict['imageData'],
            img_h=labelme_dict['imageHeight'],
            img_w=labelme_dict['imageWidth']
        )

    def save_to_path(self, save_path: str, overwrite: bool=False, img_path: str=None):
        if file_exists(save_path) and not overwrite:
            logger.error(f'File already exists at save_path: {save_path}')
            raise Exception
        if img_path is not None:
            import os
            self.img_path = os.path.relpath(path=img_path, start=get_dirpath_from_filepath(save_path))
        json_dict = self.to_dict()
        json.dump(json_dict, open(save_path, 'w'), indent=2, ensure_ascii=False)

    @classmethod
    def load_from_path(cls, json_path: str) -> LabelmeAnnotation:
        check_file_exists(json_path)
        json_dict = json.load(open(json_path, 'r'))
        return LabelmeAnnotation.from_dict(json_dict)

class LabelmeAnnotationHandler:
    def __init__(self, labelme_ann_list: List[LabelmeAnnotation]=None):
        self.labelme_ann_list = labelme_ann_list if labelme_ann_list is not None else []

    def __len__(self) -> int:
        return len(self.labelme_ann_list)

    def __getitem__(self, idx: int) -> LabelmeAnnotation:
        if len(self.labelme_ann_list) == 0:
            logger.error(f"LabelmeAnnotationHandler is empty.")
            raise IndexError
        elif idx < 0 or idx >= len(self.labelme_ann_list):
            logger.error(f"Index out of range: {idx}")
            raise IndexError
        else:
            return self.labelme_ann_list[idx]

    def __setitem__(self, idx: int, value: LabelmeAnnotation):
        check_type(value, valid_type_list=[LabelmeAnnotation])
        self.labelme_ann_list[idx] = value

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self) -> LabelmeAnnotation:
        if self.n < len(self.labelme_ann_list):
            result = self.labelme_ann_list[self.n]
            self.n += 1
            return result
        else:
            raise StopIteration

    def append(self, ann: LabelmeAnnotation):
        check_type(ann, valid_type_list=[LabelmeAnnotation])
        self.labelme_ann_list.append(ann)

    def erase_image_data(self):
        for ann in self:
            ann.img_data = None

    def _check_paths_valid(self, src_img_dir: str):
        check_dir_exists(src_img_dir)
        img_filename_list = []
        duplicate_img_filename_list = []
        for ann in self:
            img_filename = get_filename(ann.img_path)
            if img_filename not in img_filename_list:
                img_filename_list.append(ann.img_path)
            else:
                duplicate_img_filename_list.append(ann.img_path)
            img_path = f'{src_img_dir}/{img_filename}'
            check_file_exists(img_path)
        if len(duplicate_img_filename_list) > 0:
            logger.error(f'Found the following duplicate image filenames in LabelmeAnnotationHandler:\n{duplicate_img_filename_list}')
            raise Exception

    def save_to_dir(self, json_save_dir: str, src_img_dir: str, overwrite: bool=False, dst_img_dir: str=None):
        self._check_paths_valid(src_img_dir=src_img_dir)
        make_dir_if_not_exists(json_save_dir)
        delete_all_files_in_dir(json_save_dir, ask_permission=not overwrite)
        if dst_img_dir is not None:
            make_dir_if_not_exists(dst_img_dir)
            delete_all_files_in_dir(dst_img_dir, ask_permission=not overwrite)

        for ann in tqdm(self, total=len(self), unit='ann', leave=True):
            save_path = f'{json_save_dir}/{get_rootname_from_path(ann.img_path)}.json'
            src_img_path = f'{src_img_dir}/{get_filename(ann.img_path)}'
            if dst_img_dir is not None:
                dst_img_path = f'{dst_img_dir}/{get_filename(ann.img_path)}'
                copy_file(src_path=src_img_path, dest_path=dst_img_path, silent=True)
                ann.save_to_path(save_path=save_path, img_path=dst_img_path)
            else:
                ann.save_to_path(save_path=save_path, img_path=src_img_path)

    @classmethod
    def load_from_pathlist(cls, json_path_list: list) -> LabelmeAnnotationHandler:
        return LabelmeAnnotationHandler(
            labelme_ann_list=[LabelmeAnnotation.load_from_path(json_path) for json_path in json_path_list]
        )

    @classmethod
    def load_from_dir(cls, load_dir: str) -> LabelmeAnnotationHandler:
        check_dir_exists(load_dir)
        json_path_list = get_all_files_of_extension(dir_path=load_dir, extension='json')
        return cls.load_from_pathlist(json_path_list)

    def sort(self, attr_name: str, reverse: bool=False):
        if len(self) > 0:
            attr_list = list(self.labelme_ann_list[0].__dict__.keys())    
            if attr_name not in attr_list:
                logger.error(f"{LabelmeAnnotation.__name__} class has not attribute: '{attr_name}'")
                logger.error(f'Possible attribute names:')
                for name in attr_list:
                    logger.error(f'\t{name}')
                raise Exception

            self.labelme_ann_list.sort(key=operator.attrgetter(attr_name), reverse=reverse)
        else:
            logger.error(f"Cannot sort. {type(self).__name__} is empty.")
            raise Exception