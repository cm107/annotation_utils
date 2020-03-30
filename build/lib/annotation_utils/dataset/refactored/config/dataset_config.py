from __future__ import annotations
from typing import List
import yaml
import json
import numpy as np

from logger import logger
from common_utils.file_utils import file_exists
from common_utils.path_utils import find_longest_container_dir, \
    find_shortest_common_rel_path, get_extension_from_path, \
    get_possible_rel_paths, rel_to_abs_path, get_filename
from common_utils.check_utils import check_required_keys, check_type, \
    check_type_from_list, check_list_length, check_file_exists, \
    check_dir_exists, check_value

from ....coco.refactored.structs.base import BaseStructObject, BaseStructHandler

class Path:
    def __init__(self, path_str: str):
        self.path_str = path_str

    def __str__(self) -> str:
        return self.path_str

    def __repr__(self) -> str:
        return self.__str__()

    def __key(self) -> tuple:
        return tuple([self.__class__] + list(self.__dict__.values()))

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other) -> bool:
        if isinstance(other, self.__class__):
            return self.__key() == other.__key()
        return NotImplemented

    def __len__(self) -> int:
        return len(self.split())

    def __getitem__(self, idx: int) -> Path:
        if type(idx) is int:
            if len(self) == 0:
                logger.error(f"{type(self).__name__} is empty.")
                raise IndexError
            elif idx >= len(self) or idx < -len(self):
                logger.error(f"Index out of range: {idx}")
                raise IndexError
            else:
                return Path(self.split()[idx])
        elif type(idx) is slice:
            return Path.from_split(self.split()[idx.start:idx.stop:idx.step])
        else:
            logger.error(f'Expected int or slice. Got type(idx)={type(idx)}')
            raise TypeError

    def __setitem__(self, idx: int, value: Path):
        check_type(value, valid_type_list=[Path, str])
        if type(value) is str:
            value0 = Path(value)
        else:
            value0 = value
        path_parts = self.split()
        if type(idx) is int:
            path_parts[idx] = value.path_str
        elif type(idx) is slice:
            path_parts[idx.start:idx.stop:idx.step] = value.split()
        else:
            logger.error(f'Expected int or slice. Got type(idx)={type(idx)}')
            raise TypeError

    def __delitem__(self, idx):
        if type(idx) is int:
            if len(self) == 0:
                logger.error(f"{type(self).__name__} is empty.")
                raise IndexError
            elif idx >= len(self) or idx < -len(self):
                logger.error(f"Index out of range: {idx}")
                raise IndexError
            else:
                path_parts = self.split()
                del path_parts[idx]
                result = Path.from_split(path_parts)
                self.path_str = result.to_str()
        elif type(idx) is slice:
            path_parts = self.split()
            del path_parts[idx.start:idx.stop:idx.step]
            result = Path.from_split(path_parts)
            self.path_str = result.to_str()
        else:
            logger.error(f'Expected int or slice. Got type(idx)={type(idx)}')
            raise TypeError

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self) -> Path:
        if self.n < len(self):
            result = Path(self.split()[self.n])
            self.n += 1
            return result
        else:
            raise StopIteration

    def __add__(self, other: Path) -> Path:
        return Path.from_split(self.split() + other.split()).prune_slashes()

    @classmethod
    def buffer(cls, path: Path) -> Path:
        return path

    def copy(self) -> Path:
        return Path(self.path_str)

    def split(self) -> List[str]:
        result = self.path_str.split('/')
        result = [result_part for result_part in result]
        return result

    @classmethod
    def from_split(self, str_list: List[str]) -> Path:
        return Path('/'.join(str_list))

    def head(self) -> Path:
        return self[0]

    def no_head(self) -> Path:
        return self[1:]

    def tail(self) -> Path:
        return self[-1]

    def no_tail(self) -> Path:
        return self[:-1]

    def to_str(self) -> str:
        return self.path_str

    def pop_head(self) -> Path:
        result = self.head()
        self.path_str = self.no_head().to_str()
        return result

    def pop_tail(self) -> Path:
        result = self.tail()
        self.path_str = self.no_tail().to_str()
        return result

    def push_head(self, path_part: Path):
        check_type(path_part, valid_type_list=[Path, str])
        if type(path_part) is str:
            path_part0 = Path(path_part)
        else:
            path_part0 = path_part
        return Path.from_split([path_part0.path_str] + self.split())

    def push_tail(self, path_part: Path):
        check_type(path_part, valid_type_list=[Path, str])
        if type(path_part) is str:
            path_part0 = Path(path_part)
        else:
            path_part0 = path_part
        return Path.from_split(self.split() + [path_part0.path_str])

    def get_extension(self) -> str:
        if '.' in self.tail().path_str:
            return self.tail().path_str.split('.')[-1]
        else:
            return ''

    def has_extension(self) -> bool:
        return self.get_extension() != ''

    def abs(self) -> Path:
        return Path(rel_to_abs_path(self.to_str()))

    def prune_slashes(self) -> Path:
        path_str = self.to_str()
        while '//' in path_str:
            path_str = path_str.replace('//', '/')
        return Path(path_str)

    @classmethod
    def get_unique_paths(cls, paths: List[Path]) -> List[Path]:
        path_str_list = [path.path_str for path in paths]
        unique_path_str_list = list(dict.fromkeys(path_str_list))
        unique_paths = [Path(unique_path_str) for unique_path_str in unique_path_str_list]
        return unique_paths

    @classmethod
    def has_common_head(cls, paths: List[Path]) -> bool:
        if len(paths) > 1:
            path_len_list = [len(path) for path in paths]
            if 0 in path_len_list:
                return False
            path_heads = [path.head() for path in paths]
            unique_path_heads = cls.get_unique_paths(path_heads)
            if len(unique_path_heads) == 1:
                return True
            else:
                return False
        elif len(paths) == 1 and len(paths[0]) > 0 and paths[0] != Path(''):
            return True
        else:
            return False

    @classmethod
    def has_common_tail(cls, paths: List[Path]) -> bool:
        if len(paths) > 1:
            path_len_list = [len(path) for path in paths]
            if 0 in path_len_list:
                return False
            path_tails = [path.tail() for path in paths]
            unique_path_tails = cls.get_unique_paths(path_tails)
            if len(unique_path_tails) == 1:
                return True
            else:
                return False
        elif len(paths) == 1 and len(paths[0]) > 0 and paths[0] != Path(''):
            return True
        else:
            return False

    @classmethod
    def get_common_head(cls, paths: List[Path]) -> (bool, Path):
        if len(paths) > 1:
            path_len_list = [len(path) for path in paths]
            if 0 in path_len_list:
                return False, None
            path_heads = [path.head() for path in paths]
            unique_path_heads = cls.get_unique_paths(path_heads)
            if len(unique_path_heads) == 1 and unique_path_heads[0] != Path(''):
                return True, unique_path_heads[0]
            else:
                return False, None
        elif len(paths) == 1 and len(paths[0]) > 0 and paths[0].head() != Path(''):
            return True, paths[0].head()
        else:
            return False, None

    @classmethod
    def get_common_tail(cls, paths: List[Path]) -> (bool, Path):
        if len(paths) > 1:
            path_len_list = [len(path) for path in paths]
            if 0 in path_len_list:
                return False, None
            path_tails = [path.tail() for path in paths]
            unique_path_tails = cls.get_unique_paths(path_tails)
            if len(unique_path_tails) == 1 and unique_path_tails[0] != Path(''):
                return True, unique_path_tails[0]
            else:
                return False, None
        elif len(paths) == 1 and len(paths[0]) > 0 and paths[0].tail() != Path(''):
            return True, paths[0].tail()
        else:
            return False, None

    def replace(self, old: Path, new: Path) -> Path:
        check_type_from_list([old, new], valid_type_list=[Path, str])
        if type(old) is str:
            old_path = Path(old)
        else:
            old_path = old
        if type(new) is str:
            new_path = Path(new)
        else:
            new_path = new
        result = self.to_str()
        result = result.replace(old_path.to_str(), new_path.to_str())
        while '//' in result:
            result = result.replace('//', '/')
        return Path(result)

    def possible_rel_paths(self) -> List[Path]:
        return [Path.from_split(self.split()[i:]) for i in range(len(self))]

    def possible_container_dirs(self) -> List[Path]:
        result = [
            Path.from_split(self.split()[:i]) if self.has_extension() else Path.from_split(self.split()[:i+1]) \
                for i in range(len(self))
        ]
        result = [path for path in result if path.path_str != '']
        return result

    @classmethod
    def del_duplicates(cls, path_list: List[Path]) -> List[Path]:
        path_str_list = list(dict.fromkeys([path.path_str for path in path_list]))
        return [Path(path_str) for path_str in path_str_list]

    @classmethod
    def get_common_container_dirs(cls, path_list: List[Path]) -> List[Path]:
        possible_container_dirs_sets = []
        for path in path_list:
            possible_container_dirs_sets.append(set([path0.to_str() for path0 in path.possible_container_dirs()]))
        common_container_dirs = list(set.intersection(*possible_container_dirs_sets))
        return [Path(common_container_dir) for common_container_dir in common_container_dirs]

    @classmethod
    def get_longest_container_dir(cls, path_list: List[Path]) -> Path:
        common_container_dirs = Path.get_common_container_dirs(path_list)
        longest_container_dir = None
        for common_container_dir in common_container_dirs:
            if longest_container_dir is None or len(common_container_dir) > len(longest_container_dir):
                longest_container_dir = common_container_dir
        return longest_container_dir

    @staticmethod
    def _root_src_tail2dst_head(src_path: Path, dst_path: Path) -> bool:
        check_type(src_path, valid_type_list=[Path])
        check_type(dst_path, valid_type_list=[Path])
        if len(src_path) > 0:
            tail = src_path.pop_tail()
            if tail != Path(''):
                dst_path.path_str = (tail + dst_path).to_str()
                success = True
            else:
                success = False
        else:
            success = False
        return success

    @classmethod
    def _root_src_head2dst_tail(cls, src_path: Path, dst_path: Path) -> bool:
        check_type(src_path, valid_type_list=[Path])
        check_type(dst_path, valid_type_list=[Path])
        if len(src_path) > 0:
            head = src_path.pop_head()
            if head != Path(''):
                dst_path.path_str = (dst_path + head).to_str()
                success = True
            else:
                success = False
        else:
            success = False
        return success

    @classmethod
    def tail2head(cls, src_obj: List[Path], dst_obj: List[Path]) -> bool:
        # TODO: Assertion test
        check_type_from_list([src_obj, dst_obj], valid_type_list=[Path, list])
        if type(src_obj) is list:
            check_type_from_list(src_obj, valid_type_list=[Path])
        if type(dst_obj) is list:
            check_type_from_list(dst_obj, valid_type_list=[Path])
        
        if type(src_obj) is Path:
            if type(dst_obj) is Path:
                success = cls._root_src_tail2dst_head(src_obj, dst_obj)
            elif type(dst_obj) is list:
                if len(src_obj) > 0:
                    tail = src_obj.pop_tail()
                    for dst_path in dst_obj:
                        dst_path.path_str = (tail + dst_path).to_str()
                    success = True
                else:
                    success = False
            else:
                raise Exception
        elif type(src_obj) is list:
            has_common_tail, common_tail = cls.get_common_tail(src_obj)
            if has_common_tail:
                if type(dst_obj) is Path:
                    dst_obj.path_str = (common_tail + dst_obj).to_str()
                    for src_path in src_obj:
                        src_path.path_str = src_path.no_tail().to_str()
                    success = True
                elif type(dst_obj) is list:
                    if len(src_obj) != len(dst_obj):
                        logger.error(f'len(src_obj) == {len(src_obj)} != {len(dst_obj)} == len(dst_obj)')
                        raise Exception
                    for dst_path in dst_obj:
                        dst_path.path_str = (common_tail + dst_path).to_str()
                    for src_path in src_obj:
                        src_path.path_str = src_path.no_tail().to_str()
                    success = True
                else:
                    raise Exception
            else:
                success = False
        else:
            raise Exception
        return success

    @classmethod
    def head2tail(cls, src_obj: List[Path], dst_obj: List[Path]) -> bool:
        check_type_from_list([src_obj, dst_obj], valid_type_list=[Path, list])
        if type(src_obj) is list:
            check_type_from_list(src_obj, valid_type_list=[Path])
        if type(dst_obj) is list:
            check_type_from_list(dst_obj, valid_type_list=[Path])
        
        if type(src_obj) is Path:
            if type(dst_obj) is Path:
                success = cls._root_src_head2dst_tail(src_obj, dst_obj)
            elif type(dst_obj) is list:
                if len(src_obj) > 0:
                    head = src_obj.pop_head()
                    for dst_path in dst_obj:
                        dst_path.path_str = (dst_path + head).to_str()
                    success = True
                else:
                    success = False
            else:
                raise Exception
        elif type(src_obj) is list:
            has_common_head, common_head = cls.get_common_head(src_obj)
            if has_common_head:
                if type(dst_obj) is Path:
                    dst_obj.path_str = (dst_obj + common_head).to_str()
                    for src_path in src_obj:
                        src_path.path_str = src_path.no_head().to_str()
                    success = True
                elif type(dst_obj) is list:
                    if len(src_obj) != len(dst_obj):
                        logger.error(f'len(src_obj) == {len(src_obj)} != {len(dst_obj)} == len(dst_obj)')
                        raise Exception
                    for dst_path in dst_obj:
                        dst_path.path_str = (dst_path + common_head).to_str()
                    for src_path in src_obj:
                        src_path.path_str = src_path.no_head().to_str()
                    success = True
                else:
                    raise Exception
            else:
                success = False
        else:
            raise Exception
        return success

class DatasetConfig(BaseStructObject['DatasetConfig']):
    def __init__(self, img_dir: str, ann_path: str, ann_format: str='coco'):
        super().__init__()
        self.img_dir = img_dir
        self.ann_path = ann_path
        self.ann_format = ann_format

    def __str__(self) -> str:
        return str(self.__dict__)

class DatasetConfigCollection(BaseStructHandler['DatasetConfigCollection', 'DatasetConfig']):
    def __init__(self, dataset_config_list: List[DatasetConfig]=None):
        super().__init__(obj_type=DatasetConfig, obj_list=dataset_config_list)
        self.dataset_config_list = self.obj_list

    def to_dict0(self) -> dict:
        # TODO: Work In Progress
        img_dir_list = [Path(config.img_dir).abs() for config in self.dataset_config_list]
        ann_path_list = [Path(config.ann_path).abs() for config in self.dataset_config_list]
        ann_format_list = [config.ann_format for config in self.dataset_config_list]
        img_container_dir = Path.get_longest_container_dir(img_dir_list)
        ann_container_dir = Path.get_longest_container_dir(ann_path_list)
        collection_dir = Path.get_longest_container_dir([img_container_dir, ann_container_dir])
        dataset_names = [
            img_dir.replace(f'{collection_dir.path_str}/', '') \
                if img_dir != collection_dir else Path('') \
                for img_dir in img_dir_list
        ]
        
        # rel_img_dir_list = [None] * len(dataset_names)
        rel_img_dir_list = [Path('') * len(dataset_names)]
        
        logger.purple(f'Before dataset_names: {dataset_names}')
        logger.purple(f'Before rel_img_dir_list: {rel_img_dir_list}')

        while True:
            logger.blue(f'Flag0')
            # Adjust dataset_names tails
            tail_moved = Path.tail2head(dataset_names, rel_img_dir_list)

            if not tail_moved:
                break
        
        while True:
            logger.blue(f'Flag1')
            # Adjust dataset_names heads
            head_moved = Path.head2tail(dataset_names, collection_dir)

            if not head_moved:
                break

        logger.purple(f'After dataset_names: {dataset_names}')
        logger.purple(f'After rel_img_dir_list: {rel_img_dir_list}')

        rel_img_dir_list = [rel_img_dir if rel_img_dir is not None else Path('') for rel_img_dir in rel_img_dir_list]
        rel_ann_path_list = [
            ann_path.replace(f'{collection_dir}/{dataset_name}/', '') \
                if dataset_name != Path('') else ann_path.replace(f'{collection_dir}/', '') \
                for ann_path, dataset_name in zip(ann_path_list, dataset_names)
        ]

        dataset_names = [dataset_name.path_str if dataset_name.path_str != '' else '.' for dataset_name in dataset_names]
        rel_img_dir = rel_img_dir_list[0].path_str if len(list(dict.fromkeys(rel_img_dir_list))) == 1 else [rel_img_dir.path_str for rel_img_dir in rel_img_dir_list]
        if type(rel_img_dir) is str:
            rel_img_dir = rel_img_dir if rel_img_dir != '' else '.'
        elif type(rel_img_dir) is list:
            rel_img_dir = [dir_path if dir_path != '' else '.' for dir_path in rel_img_dir]
        else:
            raise Exception
        rel_ann_path = rel_ann_path_list[0].path_str if len(list(dict.fromkeys(rel_ann_path_list))) == 1 else [rel_ann_path.path_str for rel_ann_path in rel_ann_path_list]
        ann_format = ann_format_list[0] if len(list(dict.fromkeys(ann_format_list))) == 1 else ann_format_list

        return {
            'collection_dir': collection_dir.path_str,
            'dataset_names': dataset_names,
            'dataset_specific': {
                'img_dir': rel_img_dir,
                'ann_path': rel_ann_path,
                'ann_format': ann_format
            }
        }

    def to_dict(self) -> dict:
        # TODO: May need some fixing
        img_dir_list = [Path(config.img_dir).abs() for config in self.dataset_config_list]
        ann_path_list = [Path(config.ann_path).abs() for config in self.dataset_config_list]
        ann_format_list = [config.ann_format for config in self.dataset_config_list]
        img_container_dir = Path.get_longest_container_dir(img_dir_list)
        ann_container_dir = Path.get_longest_container_dir(ann_path_list)
        collection_dir = Path.get_longest_container_dir([img_container_dir, ann_container_dir])
        dataset_names = [
            img_dir.replace(f'{collection_dir.path_str}/', '') \
                if img_dir != collection_dir else Path('') \
                for img_dir in img_dir_list
        ]
        
        rel_img_dir_list = [None] * len(dataset_names)
        
        done = False
        while not done:
            # Adjust dataset_names tails
            uniform_tail = None
            for i in range(len(dataset_names)):
                if len(dataset_names[i]) == 0:
                    continue
                for j in range(i+1, len(dataset_names)):
                    if len(dataset_names[j]) == 0:
                        continue
                    if dataset_names[i][-1] != dataset_names[j][-1]:
                        uniform_tail = False
                        break
                    elif uniform_tail is None:
                        uniform_tail = True
                if uniform_tail is not None and not uniform_tail:
                    break
            if uniform_tail:
                for i in range(len(dataset_names)):
                    if rel_img_dir_list[i] is None:
                        rel_img_dir_list[i] = dataset_names[i][-1]
                    else:
                        rel_img_dir_list[i] = dataset_names[i][-1] + rel_img_dir_list[i]
                    del dataset_names[i][-1]

            # Adjust dataset_names heads
            uniform_head = None
            for i in range(len(dataset_names)):
                if len(dataset_names[i]) == 0:
                    continue
                for j in range(i+1, len(dataset_names)):
                    if len(dataset_names[j]) == 0:
                        continue
                    if dataset_names[i][0] != dataset_names[j][0]:
                        uniform_head = False
                        break
                    elif uniform_head is None:
                        uniform_head = True
                if uniform_head is not None and not uniform_head:
                    break
            if uniform_head:
                collection_dir = collection_dir + dataset_names[0][0]
                for i in range(len(dataset_names)):
                    del dataset_names[i][0]
            
            if not uniform_tail and not uniform_head:
                done = True

        rel_img_dir_list = [rel_img_dir if rel_img_dir is not None else Path('') for rel_img_dir in rel_img_dir_list]
        rel_ann_path_list = [
            ann_path.replace(f'{collection_dir}/{dataset_name}/', '') \
                if dataset_name != Path('') else ann_path.replace(f'{collection_dir}/', '') \
                for ann_path, dataset_name in zip(ann_path_list, dataset_names)
        ]

        dataset_names = [dataset_name.path_str if dataset_name.path_str != '' else '.' for dataset_name in dataset_names]
        rel_img_dir = rel_img_dir_list[0].path_str if len(list(dict.fromkeys(rel_img_dir_list))) == 1 else [rel_img_dir.path_str for rel_img_dir in rel_img_dir_list]
        if type(rel_img_dir) is str:
            rel_img_dir = rel_img_dir if rel_img_dir != '' else '.'
        elif type(rel_img_dir) is list:
            rel_img_dir = [dir_path if dir_path != '' else '.' for dir_path in rel_img_dir]
        else:
            raise Exception
        rel_ann_path = rel_ann_path_list[0].path_str if len(list(dict.fromkeys(rel_ann_path_list))) == 1 else [rel_ann_path.path_str for rel_ann_path in rel_ann_path_list]
        ann_format = ann_format_list[0] if len(list(dict.fromkeys(ann_format_list))) == 1 else ann_format_list

        return {
            'collection_dir': collection_dir.path_str,
            'dataset_names': dataset_names,
            'dataset_specific': {
                'img_dir': rel_img_dir,
                'ann_path': rel_ann_path,
                'ann_format': ann_format
            }
        }

    @classmethod
    def from_dict(cls, collection_dict: dict, check_paths: bool=True) -> DatasetConfigCollection:
        check_required_keys(
            collection_dict,
            required_keys=[
                'collection_dir', 'dataset_names', 'dataset_specific'
            ]
        )
        collection_dir = collection_dict['collection_dir']
        check_type(collection_dir, valid_type_list=[str])
        dataset_names = collection_dict['dataset_names']
        check_type(dataset_names, valid_type_list=[list])
        check_type_from_list(dataset_names, valid_type_list=[str])
        dataset_specific = collection_dict['dataset_specific']
        check_type(dataset_specific, valid_type_list=[dict])
        check_required_keys(
            dataset_specific,
            required_keys=[
                'img_dir', 'ann_path', 'ann_format'
            ]
        )
        img_dir = dataset_specific['img_dir']
        check_type(img_dir, valid_type_list=[str, list])
        if type(img_dir) is list:
            check_type_from_list(img_dir, valid_type_list=[str])
            check_list_length(img_dir, correct_length=len(dataset_names))
        ann_path = dataset_specific['ann_path']
        check_type(ann_path, valid_type_list=[str, list])
        if type(ann_path) is list:
            check_type_from_list(ann_path, valid_type_list=[str])
            check_list_length(ann_path, correct_length=len(dataset_names))
        ann_format = dataset_specific['ann_format']
        check_type(ann_format, valid_type_list=[str, list])
        if type(ann_format) is list:
            check_type_from_list(ann_format, valid_type_list=[str])
            check_list_length(ann_format, correct_length=len(dataset_names))

        dataset_config_list = []
        for i in range(len(dataset_names)):
            if type(img_dir) is str:
                img_dir0 = img_dir
            elif type(img_dir) is list:
                if i >= len(img_dir):
                    raise IndexError
                img_dir0 = img_dir[i]
            else:
                raise Exception

            if type(ann_path) is str:
                ann_path0 = ann_path
            elif type(ann_path) is list:
                if i >= len(ann_path):
                    raise IndexError
                ann_path0 = ann_path[i]
            else:
                raise Exception

            if type(ann_format) is str:
                ann_format0 = ann_format
            elif type(ann_format) is list:
                if i >= len(ann_format):
                    raise IndexError
                ann_format0 = ann_format[i]
            else:
                raise Exception
            img_dir1 = rel_to_abs_path(f'{collection_dir}/{dataset_names[i]}/{img_dir0}')
            ann_path1 = rel_to_abs_path(f'{collection_dir}/{dataset_names[i]}/{ann_path0}')
            if check_paths:
                check_dir_exists(img_dir1)
                check_file_exists(ann_path1)
            config = DatasetConfig(
                img_dir=img_dir1,
                ann_path=ann_path1,
                ann_format=ann_format0
            )
            dataset_config_list.append(config)
        return DatasetConfigCollection(dataset_config_list=dataset_config_list)

    def save_to_path(self, save_path: str, overwrite: bool=False):
        if file_exists(save_path) and not overwrite:
            logger.error(f'File already exists at: {save_path}')
            logger.error(f'Use overwrite=True to overwrite.')
            raise Exception
        extension = get_extension_from_path(save_path)
        if extension == 'json':
            json.dump(self.to_dict(), open(save_path, 'w'), indent=2, ensure_ascii=False)
        elif extension == 'yaml':
            yaml.dump(self.to_dict(), open(save_path, 'w'), allow_unicode=True)
        else:
            logger.error(f'Invalid file extension encountered: {extension}')
            logger.error(f"Valid file extensions: {['json', 'yaml']}")
            logger.error(f'Path specified: {save_path}')
            raise Exception

    @classmethod
    def load_from_path(cls, path: str) -> DatasetConfigCollection:
        check_file_exists(path)
        extension = get_extension_from_path(path)
        if extension == 'json':
            collection_dict = json.load(open(path, 'r'))
        elif extension == 'yaml':
            collection_dict = yaml.load(open(path, 'r'), Loader=yaml.FullLoader)
        else:
            logger.error(f'Invalid file extension encountered: {extension}')
            logger.error(f'Path specified: {path}')
            raise Exception
        return DatasetConfigCollection.from_dict(collection_dict)

class DatasetConfigCollectionHandler(BaseStructHandler['DatasetConfigCollectionHandler', 'DatasetConfigCollection']):
    def __init__(self, collection_list: List[DatasetConfigCollection]=None):
        super().__init__(obj_type=DatasetConfigCollection, obj_list=collection_list)
        self.collection_list = self.obj_list

    def to_dict_list(self) -> List[dict]:
        return [collection.to_dict() for collection in self]

    @classmethod
    def from_dict_list(cls, collection_dict_list: List[dict]) -> DatasetConfigCollectionHandler:
        return DatasetConfigCollectionHandler(
            collection_list=[DatasetConfigCollection.from_dict(collection_dict) for collection_dict in collection_dict_list]
        )

    def save_to_path(self, save_path: str, overwrite: bool=False):
        if file_exists(save_path) and not overwrite:
            logger.error(f'File already exists at: {save_path}')
            logger.error(f'Use overwrite=True to overwrite.')
            raise Exception
        extension = get_extension_from_path(save_path)
        if extension == 'json':
            json.dump(self.to_dict_list(), open(save_path, 'w'), indent=2, ensure_ascii=False)
        elif extension == 'yaml':
            yaml.dump(self.to_dict_list(), open(save_path, 'w'), allow_unicode=True)
        else:
            logger.error(f'Invalid file extension encountered: {extension}')
            logger.error(f"Valid file extensions: {['json', 'yaml']}")
            logger.error(f'Path specified: {save_path}')
            raise Exception

    @classmethod
    def load_from_path(cls, path: str) -> DatasetConfigCollectionHandler:
        check_file_exists(path)
        extension = get_extension_from_path(path)
        if extension == 'json':
            collection_dict_list = json.load(open(path, 'r'))
        elif extension == 'yaml':
            collection_dict_list = yaml.load(open(path, 'r'), Loader=yaml.FullLoader)
        else:
            logger.error(f'Invalid file extension encountered: {extension}')
            logger.error(f'Path specified: {path}')
            raise Exception
        return DatasetConfigCollectionHandler.from_dict_list(collection_dict_list)