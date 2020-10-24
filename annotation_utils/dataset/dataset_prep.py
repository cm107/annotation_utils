import sys
import math
import pandas as pd
from tqdm import tqdm
from typing import cast, List, Dict
from common_utils.path_utils import get_dirnames_in_dir
from common_utils.file_utils import make_dir_if_not_exists, delete_all_files_in_dir, \
    get_dir_contents_len, dir_exists, file_exists
from ..coco.structs import COCO_Dataset
from .config import DatasetConfig, DatasetConfigCollection, \
    DatasetConfigCollectionHandler

def prepare_datasets_from_dir(
    scenario_root_dir: str, dst_root_dir: str, annotation_filename: str='output.json', skip_existing: bool=False,
    val_target_proportion: float=0.05, min_val_size: int=None, max_val_size: int=None,
    orig_config_save: str='orig.yaml', reorganized_config_save: str='dataset_config.yaml', show_pbar: bool=True
):
    """
    Parameters:
        scenario_root_dir - Path to the source root directory containing all of your scenario folders. [Required]
        dst_root_dir - Path to where you would like to save your prepared scenario datasets (split into train and val) [Required]
        annotation_filename - The filename of every annotation file under scenario_root_dir [Default: 'output.json']
        skip_existing - If you terminated dataset preparation midway, you can skip the scenarios that were already made using skip_existing=True. [Default: False]
        val_target_proportion - The proportion of your scenario that you would like to allocate to validation. [Default: 0.05]
        min_val_size - The minimum number of images that you would like to use for validation. [Default: None]
        max_val_size - The maximum number of images that you would like to use for validation. [Default: None]
        orig_config_save - Where you would like to save the dataset configuration representing your scenario_root_dir. [Default: 'orig.yaml]
        reorganized_config_save - Where you would like to save the dataset configuration representing your dst_root_dir. [Default: 'dataset_config.yaml']
        show_pbar - Whether or not you would like to show a progress bar during preparation. [Default: True]

    Description:
        The datasets under each scenario directory will be combined and then split into a train + validation folder.
        The source root directory should have the following structure:
            scenario_root_dir
                scenario0
                    scenario0_part0
                    scenario0_part1
                    scenario0_part2
                    ...
                scenario1
                    scenario1_part0
                    scenario1_part1
                    scenario1_part2
                    ...
                scenario2
                    scenario2_part0
                    scenario2_part1
                    scenario2_part2
                    ...
                ...
        Note that there is no restriction on directory names, so the directory names should be anything.
        This method reads a fixed directory structure regardless of the directory names.
        Also note that it is necessary for the coco annotation file saved in each scenario part directory to have the same filename.
        If you need a more flexible approach for preparing your datasets, please define where your datasets are located in an excel sheet
        and use prepare_datasets_from_excel instead.

        The destination root directory will have the following structure:
            dst_root_dir
                scenario0
                    train
                    val
                scenario1
                    train
                    val
                scenario2
                    train
                    val
                ...
        
        The dataset configuration file saved at reorganized_config_save will reflect the directory structure of dst_root_dir.
        The configuration file representing the directory structure of the scenario_root_dir is saved under orig_config_save.

        Note that orig_config_save and reorganized_config_save do not have to be inside of dst_root_dir.
        On the contrary, it is recommended to not save orig_config_save and reorganized_config_save inside of dst_root_dir.
        It is recommended that you change the path of orig_config_save and reorganized_config_save everytime you make an addition to your datasets.
        This is because you will likely want to keep track of the previous states of your dataset configuration, and you
        may also want to rollback to a previous configuration at any given time.
    """
    make_dir_if_not_exists(dst_root_dir)
    if get_dir_contents_len(dst_root_dir) > 0 and not skip_existing:
        print(f'Directory {dst_root_dir} is not empty.\nAre you sure you want to delete the contents?')
        answer = input('yes/no')
        if answer.lower() in ['yes', 'y']:
            delete_all_files_in_dir(dst_root_dir)
        elif answer.lower() in ['no', 'n']:
            print(f'Terminating program.')
            sys.exit()
        else:
            raise ValueError(f'Invalid answer: {answer}')

    # Gather datasets from source root directory and combine.
    scenario_names = get_dirnames_in_dir(scenario_root_dir)
    scenario_datasets = cast(List[COCO_Dataset], [])
    orig_collection_handler = DatasetConfigCollectionHandler()
    pbar = tqdm(total=len(scenario_names), unit='scenario(s)') if show_pbar else None
    if pbar is not None:
        pbar.set_description('Gathering Scenarios')
    for scenario_name in scenario_names:
        orig_scenario_collection = DatasetConfigCollection(tag=scenario_name)
        src_scenario_dir = f'{scenario_root_dir}/{scenario_name}'
        part_names = get_dirnames_in_dir(src_scenario_dir)
        part_datasets = cast(List[COCO_Dataset], [])
        part_dataset_dirs = cast(List[str], [])
        for part_name in part_names:
            src_part_dir = f'{src_scenario_dir}/{part_name}'
            src_part_ann_path = f'{src_part_dir}/{annotation_filename}'
            part_dataset = COCO_Dataset.load_from_path(
                json_path=src_part_ann_path,
                img_dir=src_part_dir
            )
            part_datasets.append(part_dataset)
            part_dataset_dirs.append(src_part_dir)
            orig_scenario_part_config = DatasetConfig(
                img_dir=src_part_dir,
                ann_path=src_part_ann_path,
                ann_format='coco',
                tag=part_name
            )
            orig_scenario_collection.append(orig_scenario_part_config)
        scenario_dataset = COCO_Dataset.combine(dataset_list=part_datasets, img_dir_list=part_dataset_dirs, show_pbar=False)
        scenario_datasets.append(scenario_dataset)
        orig_collection_handler.append(orig_scenario_collection)
        if pbar is not None:
            pbar.update()
    orig_collection_handler.save_to_path(orig_config_save, overwrite=True)
    pbar.close()
    
    # Split combined scenario datasets into train and val and save them.
    train_collection = DatasetConfigCollection(tag='train')
    val_collection = DatasetConfigCollection(tag='val')
    pbar = tqdm(total=len(scenario_names)) if show_pbar else None
    if pbar is not None:
        pbar.set_description('Splitting Scenarios Into Train/Val')
    for i in range(len(scenario_names)):
        dst_scenario_dir = f'{dst_root_dir}/{scenario_names[i]}'
        if dir_exists(dst_scenario_dir):
            if skip_existing:
                if pbar is not None:
                    pbar.update()
                continue
            else:
                raise FileExistsError(f'Directory already exists: {dst_scenario_dir}')
        else:
            make_dir_if_not_exists(dst_scenario_dir)
        orig_num_images = len(scenario_datasets[i].images)
        assert orig_num_images >= 2, f'{scenario_names[i]} has only {orig_num_images} images, and thus cannot be split into train and val.'
        num_val = int(len(scenario_datasets[i].images) * val_target_proportion)
        num_val = 1 if num_val == 0 else num_val
        num_val = min_val_size if min_val_size is not None and num_val < min_val_size else num_val
        num_val = max_val_size if max_val_size is not None and num_val > max_val_size else num_val
        num_train = orig_num_images - num_val
        train_dataset, val_dataset = scenario_datasets[i].split_into_parts(ratio=[num_train, num_val], shuffle=True)
        
        dst_train_dir = f'{dst_scenario_dir}/train'
        make_dir_if_not_exists(dst_train_dir)
        train_dataset.move_images(
            dst_img_dir=dst_train_dir,
            preserve_filenames=False,
            update_img_paths=True,
            show_pbar=False
        )
        train_ann_path = f'{dst_train_dir}/output.json'
        train_dataset.save_to_path(train_ann_path, overwrite=True)
        train_dataset_config = DatasetConfig(img_dir=dst_train_dir, ann_path=train_ann_path, ann_format='coco', tag=f'{scenario_names[i]}_train')
        train_collection.append(train_dataset_config)

        dst_val_dir = f'{dst_scenario_dir}/val'
        make_dir_if_not_exists(dst_val_dir)
        val_dataset.move_images(
            dst_img_dir=dst_val_dir,
            preserve_filenames=False,
            update_img_paths=True,
            show_pbar=False
        )
        val_ann_path = f'{dst_val_dir}/output.json'
        val_dataset.save_to_path(val_ann_path, overwrite=True)
        val_dataset_config = DatasetConfig(img_dir=dst_val_dir, ann_path=val_ann_path, ann_format='coco', tag=f'{scenario_names[i]}_val')
        val_collection.append(val_dataset_config)
        if pbar is not None:
            pbar.update()
    pbar.close()
    collection_handler = DatasetConfigCollectionHandler([train_collection, val_collection])
    collection_handler.save_to_path(reorganized_config_save, overwrite=True)

def prepare_datasets_from_excel(
    xlsx_path: str, dst_root_dir: str,
    usecols: str='A:L', skiprows: int=None, skipfooter: int=0,
    skip_existing: bool=False,
    val_target_proportion: float=0.05, min_val_size: int=None, max_val_size: int=None,
    orig_config_save: str='orig.yaml', reorganized_config_save: str='dataset_config.yaml',
    show_pbar: bool=True
):
    """
    Parameters:
        xlsx_path - Path to excel sheet that contains all of the information about where your datasets are located.
        dst_root_dir - Path to where you would like to save your prepared scenario datasets (split into train and val)
        usecols - Specify which columns you would like to parse from the excel sheet at xlsx_path. [Default: 'A:L']
        skiprows - Specify the number of rows from the top that you would like to skip when parsing the excel sheet. [Default: None]
        skipfooter - Specify the number of rows from the bottom that you would like to skip when parsing the excel sheet. [Default: 0]
        skip_existing - If you terminated dataset preparation midway, you can skip the scenarios that were already made using skip_existing=True. [Default: False]
        val_target_proportion - The proportion of your scenario that you would like to allocate to validation. [Default: 0.05]
        min_val_size - The minimum number of images that you would like to use for validation. [Default: None]
        max_val_size - The maximum number of images that you would like to use for validation. [Default: None]
        orig_config_save - Where you would like to save the dataset configuration representing your scenario_root_dir. [Default: 'orig.yaml]
        reorganized_config_save - Where you would like to save the dataset configuration representing your dst_root_dir. [Default: 'dataset_config.yaml']
        show_pbar - Whether or not you would like to show a progress bar during preparation. [Default: True]
    
    Description:
        The datasets specified in the excel sheet at xlsx_path will be combined and then split into a train + validation folder.
        Since the absolute paths of both image directories and annotation paths are parsed from the excel sheet, there is no need to place any restrictions
        on where each dataset needs to be located.

        The destination root directory will have the following structure:
            dst_root_dir
                scenario0
                    train
                    val
                scenario1
                    train
                    val
                scenario2
                    train
                    val
                ...

        The dataset configuration file saved at reorganized_config_save will reflect the directory structure of dst_root_dir.
        The configuration file representing the directory structure defined in your excel sheet is saved under orig_config_save.

        Note that orig_config_save and reorganized_config_save do not have to be inside of dst_root_dir.
        On the contrary, it is recommended to not save orig_config_save and reorganized_config_save inside of dst_root_dir.
        It is recommended that you change the path of orig_config_save and reorganized_config_save everytime you make an addition to your datasets.
        This is because you will likely want to keep track of the previous states of your dataset configuration, and you
        may also want to rollback to a previous configuration at any given time.
    """
    # Parse Excel Sheet
    if not file_exists(xlsx_path):
        raise FileNotFoundError(f'File not found: {xlsx_path}')
    data_df = pd.read_excel(xlsx_path, usecols=usecols, skiprows=skiprows, skipfooter=skipfooter)
    data_records = data_df.to_dict(orient='records')

    required_keys = [
        'Scenario Name', 'Dataset Name', 'Image Directory', 'Annotation Path'
    ]
    parsed_keys = list(data_records[0].keys())
    missing_keys = []
    for required_key in required_keys:
        if required_key not in parsed_keys:
            missing_keys.append(required_key)
    if len(missing_keys) > 0:
        raise KeyError(
            f"""
            Couldn't find the following required keys in the given excel sheet:
            missing_keys: {missing_keys}
            required_keys: {required_keys}
            parsed_keys: {parsed_keys}
            xlsx_path: {xlsx_path}

            Please check your excel sheet and script parameters and try again.
            Note: usecols, skiprows, and skipfooter affect which parts of the excel sheet are parsed.
            """
        )

    def is_empty_cell(info_dict: Dict[str, str], key: str, expected_type: type=str) -> bool:
        return not isinstance(info_dict[key], expected_type) and math.isnan(info_dict[key])

    collection_handler = DatasetConfigCollectionHandler()
    current_scenario_name = None
    working_config_list = cast(List[DatasetConfig], [])
    pbar = tqdm(total=len(data_records), unit='item(s)') if show_pbar else None
    if pbar is not None:
        pbar.set_description('Parsing Excel Sheet')
    for info_dict in data_records:
        for required_cell_key in ['Dataset Name', 'Image Directory', 'Annotation Path']:
            if is_empty_cell(info_dict, key=required_cell_key, expected_type=str):
                raise ValueError(
                    f"""
                    Encountered empty cell under {required_cell_key}.
                    Row Dictionary: {info_dict}
                    xlsx_path: {xlsx_path}
                    Please check your excel sheet.
                    """
                )
        assert 'Scenario Name' in info_dict
        scenario_name = info_dict['Scenario Name'] \
            if 'Scenario Name' in info_dict and not is_empty_cell(info_dict, key='Scenario Name', expected_type=str) \
            else None
        dataset_name = info_dict['Dataset Name']
        img_dir = info_dict['Image Directory']
        ann_path = info_dict['Annotation Path']
        if scenario_name is not None:
            if len(working_config_list) > 0:
                collection = DatasetConfigCollection(working_config_list, tag=current_scenario_name)
                collection_handler.append(collection)
                working_config_list = []
            current_scenario_name = scenario_name
        config = DatasetConfig(img_dir=img_dir, ann_path=ann_path, ann_format='coco', tag=dataset_name)
        working_config_list.append(config)
        if pbar is not None:
            pbar.update()
    if len(working_config_list) > 0:
        collection = DatasetConfigCollection(working_config_list, tag=current_scenario_name)
        collection_handler.append(collection)
        working_config_list = []
    if pbar is not None:
        pbar.close()
    collection_handler.save_to_path(orig_config_save, overwrite=True)

    # Combine Datasets
    train_collection = DatasetConfigCollection(tag='train')
    val_collection = DatasetConfigCollection(tag='val')

    make_dir_if_not_exists(dst_root_dir)
    pbar = tqdm(total=len(collection_handler), unit='scenario(s)') if show_pbar else None
    if pbar is not None:
        pbar.set_description('Combining Scenarios')
    for collection in collection_handler:
        scenario_root_dir = f'{dst_root_dir}/{collection.tag}'
        make_dir_if_not_exists(scenario_root_dir)
        scenario_train_dir = f'{scenario_root_dir}/train'
        make_dir_if_not_exists(scenario_train_dir)
        scenario_val_dir = f'{scenario_root_dir}/val'
        make_dir_if_not_exists(scenario_val_dir)

        if (not file_exists(f'{scenario_train_dir}/output.json') or not file_exists(f'{scenario_val_dir}/output.json')) or not skip_existing:
            combined_dataset = COCO_Dataset.combine_from_config(collection, img_sort_attr_name='file_name', show_pbar=False)
            orig_num_images = len(combined_dataset.images)
            assert orig_num_images >= 2, f'{collection.tag} has only {orig_num_images} images, and thus cannot be split into train and val.'
            num_val = int(len(combined_dataset.images) * val_target_proportion)
            num_val = 1 if num_val == 0 else num_val
            num_val = min_val_size if min_val_size is not None and num_val < min_val_size else num_val
            num_val = max_val_size if max_val_size is not None and num_val > max_val_size else num_val
            num_train = orig_num_images - num_val
            train_dataset, val_dataset = combined_dataset.split_into_parts(ratio=[num_train, num_val], shuffle=True)

            train_dataset.move_images(
                dst_img_dir=scenario_train_dir,
                preserve_filenames=False,
                update_img_paths=True,
                overwrite=True,
                show_pbar=False
            )
            train_dataset.save_to_path(f'{scenario_train_dir}/output.json', overwrite=True)
            train_collection.append(DatasetConfig(img_dir=scenario_train_dir, ann_path=f'{scenario_train_dir}/output.json', tag=f'{collection.tag}_train'))

            val_dataset.move_images(
                dst_img_dir=scenario_val_dir,
                preserve_filenames=False,
                update_img_paths=True,
                overwrite=True,
                show_pbar=False
            )
            val_dataset.save_to_path(f'{scenario_val_dir}/output.json', overwrite=True)
            val_collection.append(DatasetConfig(img_dir=scenario_val_dir, ann_path=f'{scenario_val_dir}/output.json', tag=f'{collection.tag}_val'))
        else:
            train_dataset = COCO_Dataset.load_from_path(f'{scenario_train_dir}/output.json', img_dir=f'{scenario_train_dir}')
            train_collection.append(DatasetConfig(img_dir=scenario_train_dir, ann_path=f'{scenario_train_dir}/output.json', tag=f'{collection.tag}_train'))
            val_dataset = COCO_Dataset.load_from_path(f'{scenario_val_dir}/output.json', img_dir=f'{scenario_val_dir}')
            val_collection.append(DatasetConfig(img_dir=scenario_val_dir, ann_path=f'{scenario_val_dir}/output.json', tag=f'{collection.tag}_val'))
        if pbar is not None:
            pbar.update()
    if pbar is not None:
        pbar.close()

    organized_collection_handler = DatasetConfigCollectionHandler([train_collection, val_collection])
    organized_collection_handler.save_to_path(reorganized_config_save, overwrite=True)