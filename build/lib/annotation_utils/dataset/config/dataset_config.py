from __future__ import annotations
from typing import List
import yaml
import json

from logger import logger
from common_utils.file_utils import file_exists
from common_utils.path_utils import get_extension_from_path, rel_to_abs_path
from common_utils.check_utils import check_required_keys, check_type, \
    check_type_from_list, check_list_length, check_file_exists, \
    check_dir_exists

from ...base import BaseStructObject, BaseStructHandler
from .path import Path

class DatasetConfig(BaseStructObject['DatasetConfig']):
    def __init__(self, img_dir: str, ann_path: str, ann_format: str='coco', tag: str=None):
        super().__init__()
        self.img_dir = img_dir
        self.ann_path = ann_path
        self.ann_format = ann_format
        self.tag = tag

    def __str__(self) -> str:
        return str(self.__dict__)

class DatasetConfigCollection(BaseStructHandler['DatasetConfigCollection', 'DatasetConfig']):
    def __init__(self, dataset_config_list: List[DatasetConfig]=None, tag: str=None):
        super().__init__(obj_type=DatasetConfig, obj_list=dataset_config_list)
        self.dataset_config_list = self.obj_list
        self.tag = tag

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
        if len(self.dataset_config_list) == 0:
            return {}
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
        tag = [config.tag for config in self]
        tag = tag[0] if len(list(dict.fromkeys(tag))) == 1 else tag

        return {
            'collection_dir': collection_dir.path_str,
            'dataset_names': dataset_names,
            'dataset_specific': {
                'img_dir': rel_img_dir,
                'ann_path': rel_ann_path,
                'ann_format': ann_format,
                'tag': tag
            },
            'tag': self.tag
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
        collection_tag = None if 'tag' not in collection_dict else collection_dict['tag']
        check_type(collection_tag, valid_type_list=[type(None), str])
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
        dataset_tag = None if 'tag' not in dataset_specific else dataset_specific['tag']
        check_type(dataset_tag, valid_type_list=[type(None), str, list])
        if type(dataset_tag) is list:
            check_type_from_list(dataset_tag, valid_type_list=[type(None), str])
            check_list_length(dataset_tag, correct_length=len(dataset_names))

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

            if type(dataset_tag) is str or dataset_tag is None:
                dataset_tag0 = dataset_tag
            elif type(dataset_tag) is list:
                if i >= len(dataset_tag):
                    raise IndexError
                dataset_tag0 = dataset_tag[i]
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
                ann_format=ann_format0,
                tag=dataset_tag0
            )
            dataset_config_list.append(config)
        return DatasetConfigCollection(dataset_config_list=dataset_config_list, tag=collection_tag)

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

    def filter_by_tag(self, tags: List[str]=None, new_collection_tag: str=None) -> DatasetConfigCollection:
        target_tags = [None] if tags is None else [tags] if type(tags) is not list else tags
        collection_tag = self.tag if new_collection_tag is None else new_collection_tag
        return DatasetConfigCollection(
            dataset_config_list=[config for config in self if config.tag in target_tags],
            tag=collection_tag
        )

class DatasetConfigCollectionHandler(BaseStructHandler['DatasetConfigCollectionHandler', 'DatasetConfigCollection']):
    def __init__(self, collection_list: List[DatasetConfigCollection]=None):
        collection_list0 = [collection for collection in collection_list if len(collection) > 0] if collection_list is not None else None
        super().__init__(obj_type=DatasetConfigCollection, obj_list=collection_list0)
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

    def filter_by_collection_tag(self, tags: List[str]=None) -> DatasetConfigCollectionHandler:
        target_tags = [None] if tags is None else [tags] if type(tags) is not list else tags
        return DatasetConfigCollectionHandler([collection for collection in self if collection.tag in target_tags])

    def filter_by_dataset_tag(self, tags: List[str]=None) -> DatasetConfigCollectionHandler:
        target_tags = [None] if tags is None else [tags] if type(tags) is not list else tags
        return DatasetConfigCollectionHandler([collection.filter_by_tag(tags=target_tags) for collection in self])