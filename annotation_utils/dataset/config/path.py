import yaml
import json
from logger import logger
from common_utils.check_utils import check_type_from_list, check_type, check_value_from_list, \
    check_file_exists, check_value
from common_utils.path_utils import get_extension_from_path

class DatasetPathConfig:
    def __init__(self):
        self.data = None

        self.valid_extensions = ['json', 'yaml']
        self.main_required_keys = ['collection_dir', 'dataset_names', 'dataset_specific']
        self.specific_required_keys = ['img_dir', 'ann_path', 'ann_format']
        self.valid_ann_formats = ['coco', 'custom']

    @classmethod
    def from_load(self, target, verbose: bool=False):
        config = DatasetPathConfig()
        config.load(target=target, verbose=verbose)
        return config

    def check_valid_config(self, collection_dict_list: list):
        check_type(item=collection_dict_list, valid_type_list=[list])
        for i, collection_dict in enumerate(collection_dict_list):
            check_type(item=collection_dict, valid_type_list=[dict])
            check_value_from_list(
                item_list=list(collection_dict.keys()),
                valid_value_list=self.main_required_keys
            )
            for required_key in self.main_required_keys:
                if required_key not in collection_dict.keys():
                    logger.error(f"collection_dict at index {i} is missing required key: {required_key}")
                    raise Exception
            collection_dir = collection_dict['collection_dir']
            check_type(item=collection_dir, valid_type_list=[str])
            dataset_names = collection_dict['dataset_names']
            check_type(item=dataset_names, valid_type_list=[list])
            check_type_from_list(item_list=dataset_names, valid_type_list=[str])
            dataset_specific = collection_dict['dataset_specific']
            check_type(item=dataset_specific, valid_type_list=[dict])
            check_value_from_list(
                item_list=list(dataset_specific.keys()),
                valid_value_list=self.specific_required_keys
            )
            for required_key in self.specific_required_keys:
                if required_key not in dataset_specific.keys():
                    logger.error(f"dataset_specific at index {i} is missing required key: {required_key}")
                    raise Exception
            img_dir = dataset_specific['img_dir']
            ann_path = dataset_specific['ann_path']
            ann_format = dataset_specific['ann_format']
            check_type_from_list(item_list=[img_dir, ann_path, ann_format], valid_type_list=[str, list])
            if type(img_dir) is list and len(img_dir) != len(dataset_names):
                logger.error(f"Length mismatch at index: {i}")
                logger.error(f"type(img_dir) is list but len(img_dir) == {len(img_dir)} != {len(dataset_names)} == len(dataset_names)")
                raise Exception
            if type(ann_path) is list and len(ann_path) != len(dataset_names):
                logger.error(f"Length mismatch at index: {i}")
                logger.error(f"type(ann_path) is list but len(ann_path) == {len(ann_path)} != {len(dataset_names)} == len(dataset_names)")
                raise Exception
            if type(ann_format) is list and len(ann_format) != len(dataset_names):
                logger.error(f"Length mismatch at index: {i}")
                logger.error(f"type(ann_format) is list but len(ann_format) == {len(ann_format)} != {len(dataset_names)} == len(dataset_names)")
                raise Exception

            if type(ann_format) is str:
                check_value(item=ann_format, valid_value_list=self.valid_ann_formats)
            elif type(ann_format) is list:
                check_value_from_list(item_list=ann_format, valid_value_list=self.valid_ann_formats)
            else:
                raise Exception

    def write_config(self, dest_path: str, verbose: bool=False):
        extension = get_extension_from_path(dest_path)
        if extension == 'yaml':
            yaml.dump(self.data, open(dest_path, 'w'), allow_unicode=True)
        elif extension == 'json':
            json.dump(self.data, open(dest_path, 'w'), indent=2, ensure_ascii=False)
        else:
            logger.error(f"Invalid extension: {extension}")
            logger.error(f"Valid file extensions: {self.valid_extensions}")
            raise Exception
        if verbose:
            logger.good(f"Dataset path config written successfully to:\n{dest_path}")
    
    def load(self, target, verbose: bool=False):
        if type(target) is str:
            extension = get_extension_from_path(target)
            if extension == 'yaml':
                check_file_exists(target)
                loaded_data = yaml.load(open(target, 'r'), Loader=yaml.FullLoader)
            elif extension == 'json':
                check_file_exists(target)
                loaded_data = json.load(open(target, 'r'))
            else:
                logger.error(f"Invalid extension: {extension}")
                logger.error(f"Note that string targets are assumed to be paths.")
                logger.error(f"Valid file extensions: {self.valid_extensions}")
                raise Exception
        elif type(target) is list:
            loaded_data = target
        elif type(target) is dict:
            loaded_data = [target]
        else:
            logger.error(f"Invalid target type: {type(target)}")
            raise Exception
        self.check_valid_config(collection_dict_list=loaded_data)
        self.data = loaded_data
        if verbose:
            logger.good(f"Dataset path config has been loaded successfully.")

    def _get_img_dir_list(self, dataset_paths: list, img_dir) -> list:
        check_type(item=img_dir, valid_type_list=[str, list])
        if type(img_dir) is str:
            img_dir_list = [f"{dataset_path}/{img_dir}" for dataset_path in dataset_paths]
        elif type(img_dir) is list:
            if len(img_dir) == len(dataset_paths):
                check_type_from_list(item_list=img_dir, valid_type_list=[str])
                img_dir_list = [f"{dataset_path}/{img_dir_path}" for dataset_path, img_dir_path in zip(dataset_paths, img_dir)]
            else:
                logger.error(f"type(img_dir) is list but len(img_dir) == {len(img_dir)} != {len(dataset_paths)} == len(dataset_paths)")
                raise Exception
        else:
            raise Exception
        return img_dir_list

    def _get_ann_path_list(self, dataset_paths: list, ann_path) -> list:
        check_type(item=ann_path, valid_type_list=[str, list])
        if type(ann_path) is str:
            ann_path_list = [f"{dataset_path}/{ann_path}" for dataset_path in dataset_paths]
        elif type(ann_path) is list:
            if len(ann_path) == len(dataset_paths):
                check_type_from_list(item_list=ann_path, valid_type_list=[str])
                ann_path_list = [f"{dataset_path}/{ann_path_path}" for dataset_path, ann_path_path in zip(dataset_paths, ann_path)]
            else:
                logger.error(f"type(ann_path) is list but len(ann_path) == {len(ann_path)} != {len(dataset_paths)} == len(dataset_paths)")
                raise Exception
        else:
            raise Exception
        return ann_path_list

    def _get_ann_format_list(self, dataset_paths: list, ann_format) -> list:
        check_type(item=ann_format, valid_type_list=[str, list])
        
        if type(ann_format) is str:
            check_value(item=ann_format, valid_value_list=self.valid_ann_formats)
            ann_format_list = [ann_format] * len(dataset_paths)
        elif type(ann_format) is list:
            check_value_from_list(item_list=ann_format, valid_value_list=self.valid_ann_formats)
            if len(ann_format) == len(dataset_paths):
                check_type_from_list(item_list=ann_format, valid_type_list=[str])
                ann_format_list = ann_format
            else:
                logger.error(f"type(ann_format) is list but len(ann_format) == {len(ann_format)} != {len(dataset_paths)} == len(dataset_paths)")
                raise Exception
        else:
            raise Exception
        return ann_format_list

    def get_paths(self) -> (list, list, list, list):
        if self.data is None:
            logger.error(f"Config hasn't been loaded yet. Please use load() method.")
            raise Exception

        combined_dataset_dir_list = []
        combined_img_dir_list = []
        combined_ann_path_list = []
        combined_ann_format_list = []
        for collection_dict in self.data:
            collection_dir = collection_dict['collection_dir']
            dataset_names = collection_dict['dataset_names']
            img_dir = collection_dict['dataset_specific']['img_dir']
            ann_path = collection_dict['dataset_specific']['ann_path']
            ann_format = collection_dict['dataset_specific']['ann_format']
            dataset_paths = [f'{collection_dir}/{dataset_name}' for dataset_name in dataset_names]
            img_dir_list = self._get_img_dir_list(dataset_paths=dataset_paths, img_dir=img_dir)
            ann_path_list = self._get_ann_path_list(dataset_paths=dataset_paths, ann_path=ann_path)
            ann_format_list = self._get_ann_format_list(dataset_paths=dataset_paths, ann_format=ann_format)
            combined_dataset_dir_list.extend(dataset_paths)
            combined_img_dir_list.extend(img_dir_list)
            combined_ann_path_list.extend(ann_path_list)
            combined_ann_format_list.extend(ann_format_list)

        return combined_dataset_dir_list, combined_img_dir_list, combined_ann_path_list, combined_ann_format_list