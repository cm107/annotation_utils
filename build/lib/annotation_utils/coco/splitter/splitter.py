import random
import numpy as np

from logger import logger

from common_utils.path_utils import get_next_dump_path
from common_utils.file_utils import dir_exists, delete_all_files_in_dir, make_dir_if_not_exists, make_dir, copy_file
from common_utils.path_utils import get_filename, get_dirpath_from_filepath, get_extension_from_path, get_extension_from_filename

from ..annotation import COCO_AnnotationFileParser
from ..structs import \
    COCO_License, COCO_Image, COCO_Annotation, COCO_Category
from ...util.utils.coco import COCO_Field_Buffer, COCO_Mapper_Handler
from ..writer import COCO_Writer

def get_possible_paths(path: str):
    path_parts = path.split('/')
    return ['/'.join(path_parts[i:]) for i in range(len(path_parts))]

def find_img_path(coco_url: str, src_root_dir: str) -> (bool, str):
    found = False
    img_path = None
    src_root_dirname = src_root_dir.split('/')[-1]
    for possible_path in get_possible_paths(path=coco_url):
        if possible_path.split('/')[0] == src_root_dirname:
            found = True
            img_path = f"{src_root_dir}/{'/'.join(possible_path.split('/')[1:])}"
            break
    return found, img_path

def print_id_update(label: str, added_new: bool, old_id: int, new_id: int):
    header = 'ADDED' if added_new else 'LINKED'
    logger.info(f"{label} {header}: {old_id} -> {new_id}")

class COCO_Splitter:
    def __init__(self, ann_path: list, src_root_dir: str, dest_dir: str, ratio: list=[2, 1, 0]):
        # Constructor Params
        self.ann_path = ann_path
        self.src_root_dir = src_root_dir
        self.dest_dir = dest_dir
        self.ratio = ratio

        # Parser Related
        parser = COCO_AnnotationFileParser(annotation_path=ann_path)
        parser.load(verbose=False)

        self.buffer = COCO_Field_Buffer.from_parser(
            parser=parser
        )
        self.train_buffer = COCO_Field_Buffer.from_scratch(
            description=f"Train COCO Dataset Split From {get_filename(path=ann_path)}",
            url=self.buffer.info.url,
            version=self.buffer.info.version
        )
        self.test_buffer = COCO_Field_Buffer.from_scratch(
            description=f"Test COCO Dataset Split From {get_filename(path=ann_path)}",
            url=self.buffer.info.url,
            version=self.buffer.info.version
        )
        self.val_buffer = COCO_Field_Buffer.from_scratch(
            description=f"Validation COCO Dataset Split From {get_filename(path=ann_path)}",
            url=self.buffer.info.url,
            version=self.buffer.info.version
        )

        self.buffer2train_mapper = COCO_Mapper_Handler()
        self.buffer2test_mapper = COCO_Mapper_Handler()
        self.buffer2val_mapper = COCO_Mapper_Handler()

        # Destination Directory Related
        self.train_dir = f"{self.dest_dir}/train"
        self.test_dir = f"{self.dest_dir}/test"
        self.val_dir = f"{self.dest_dir}/val"

        self.target_img_dir = "img"
        self.target_ann_dir = "ann"

    def setup_directories(self, verbose: bool=False):
        make_dir_if_not_exists(self.dest_dir)
        for target_dir in [self.train_dir, self.test_dir, self.val_dir]:
            if dir_exists(target_dir):
                delete_all_files_in_dir(dir_path=target_dir, ask_permission=False, verbose=verbose)
            else:
                make_dir(target_dir)
            img_dir = f"{target_dir}/{self.target_img_dir}"
            ann_dir = f"{target_dir}/{self.target_ann_dir}"
            make_dir(img_dir)
            make_dir(ann_dir)

    def split_image_list(self, image_list: list) -> (list, list, list):
        locations = np.cumsum([val*int(len(image_list)/sum(self.ratio)) for val in self.ratio]) - 1
        start_location = None
        end_location = 0
        count = 0
        sample_lists = []
        while count < len(locations):
            start_location = end_location
            end_location = locations[count]
            count += 1
            sample_lists.append(self.buffer.images.image_list[start_location:end_location])

        train_list, test_list, val_list = sample_lists
        return train_list, test_list, val_list

    def copy_image(self, coco_image: COCO_Image, dest_img_dir: str, new_img_filename: str=None, verbose: bool=False):
        coco_url = coco_image.coco_url
        found, src_img_path = find_img_path(coco_url=coco_url, src_root_dir=self.src_root_dir)
        if not found:
            logger.error(f"Couldn't find any permutation of coco_url={coco_url} under {self.src_root_dir}")
            raise Exception
        if new_img_filename is None:
            img_filename = get_filename(path=coco_url)
            if img_filename != coco_image.file_name:
                logger.error(f"coco_url filename doesn't match coco json file_name")
                logger.error(f"coco_url: {coco_url}")
                logger.error(f"file_name: {coco_image.file_name}")
                raise Exception
        else:
            img_filename = new_img_filename
        dest_img_path = f"{dest_img_dir}/{img_filename}"
        silent = not verbose
        copy_file(src_path=src_img_path, dest_path=dest_img_path, silent=silent)

    def update_images(
        self, target_buffer: COCO_Field_Buffer, target_mapper: COCO_Mapper_Handler,
        coco_image: COCO_Image, unique_key: str, new_img_filename: str=None, verbose: bool=False
    ):
        pending_coco_image = coco_image.copy()
        if new_img_filename is not None:
            coco_url = pending_coco_image.coco_url
            file_name = pending_coco_image.file_name
            coco_url_dirpath = get_dirpath_from_filepath(coco_url)
            coco_url_filename = get_filename(coco_url)
            if file_name != coco_url_filename:
                logger.error(f"file_name == {file_name} != coco_url_filename == {coco_url_filename}")
                raise Exception
            new_coco_url = f"{coco_url_dirpath}/{new_img_filename}"
            new_file_name = new_img_filename
            pending_coco_image.coco_url = new_coco_url
            pending_coco_image.file_name = new_file_name
            
        added_new, old_id, new_id = target_buffer.process_image(
            coco_image=pending_coco_image, id_mapper=target_mapper, unique_key=unique_key
        )
        if verbose:
            print_id_update(
                label=f'{unique_key} Image', added_new=added_new,
                old_id=old_id, new_id=new_id
            )

    def update_licenses(
        self, target_buffer: COCO_Field_Buffer, target_mapper: COCO_Mapper_Handler,
        coco_license: COCO_License, unique_key: str, verbose: bool=False
    ):
        added_new, old_id, new_id = target_buffer.process_license(
            coco_license=coco_license, id_mapper=target_mapper, unique_key=unique_key
        )
        if verbose:
            print_id_update(
                label=f'{unique_key} License', added_new=added_new,
                old_id=old_id, new_id=new_id
            )

    def update_annotations(
        self, target_buffer: COCO_Field_Buffer, target_mapper: COCO_Mapper_Handler,
        coco_annotation: COCO_Annotation, unique_key: str, verbose: bool=False
    ):
        added_new, old_id, new_id = target_buffer.process_annotation(
            coco_annotation=coco_annotation, id_mapper=target_mapper, unique_key=unique_key
        )
        if verbose:
            print_id_update(
                label=f'{unique_key} Annotation', added_new=added_new,
                old_id=old_id, new_id=new_id
            )

    def update_categories(
        self, target_buffer: COCO_Field_Buffer, target_mapper: COCO_Mapper_Handler,
        coco_category: COCO_Category, unique_key: str, verbose: bool=False
    ):
        added_new, old_id, new_id = target_buffer.process_category(
            coco_category=coco_category, id_mapper=target_mapper, unique_key=unique_key
        )
        if verbose:
            print_id_update(
                label=f'{unique_key} Category', added_new=added_new,
                old_id=old_id, new_id=new_id
            )

    def load_split(self, preserve_filenames: bool=False, verbose: bool=False):
        self.setup_directories(verbose=verbose)
        random.shuffle(self.buffer.images.image_list)
        train_image_list, test_image_list, val_image_list = self.split_image_list(image_list=self.buffer.images.image_list)

        for image_list, target_dir, target_buffer, target_mapper, unique_key in \
            zip(
                [train_image_list, test_image_list, val_image_list],
                [self.train_dir, self.test_dir, self.val_dir],
                [self.train_buffer, self.test_buffer, self.val_buffer],
                [self.buffer2train_mapper, self.buffer2test_mapper, self.buffer2val_mapper],
                ['train', 'test', 'val']
            ):
            dest_img_dir = f"{target_dir}/img"
            for coco_image in image_list:
                image_id = coco_image.id
                if not preserve_filenames:
                    img_extension = get_extension_from_filename(coco_image.file_name)
                    new_img_filename = get_filename(get_next_dump_path(dump_dir=dest_img_dir, file_extension=img_extension))
                    self.copy_image(coco_image=coco_image, dest_img_dir=dest_img_dir, new_img_filename=new_img_filename, verbose=verbose)
                else:
                    self.copy_image(coco_image=coco_image, dest_img_dir=dest_img_dir, verbose=verbose)
                coco_license = self.buffer.licenses.get_license_from_id(id=coco_image.license_id)
                self.update_licenses(
                    target_buffer=target_buffer, target_mapper=target_mapper,
                    coco_license=coco_license, unique_key=unique_key, verbose=verbose
                )
                if not preserve_filenames:
                    self.update_images(
                        target_buffer=target_buffer, target_mapper=target_mapper,
                        coco_image=coco_image, unique_key=unique_key, new_img_filename=new_img_filename, verbose=verbose
                    )
                else:
                    self.update_images(
                        target_buffer=target_buffer, target_mapper=target_mapper,
                        coco_image=coco_image, unique_key=unique_key, verbose=verbose
                    )
                coco_annotation_list = self.buffer.annotations.get_annotations_from_imgIds(imgIds=[image_id])
                for coco_annotation in coco_annotation_list:
                    coco_category = self.buffer.categories.get_category_from_id(id=coco_annotation.category_id)
                    self.update_categories(
                        target_buffer=target_buffer, target_mapper=target_mapper,
                        coco_category=coco_category, unique_key=unique_key, verbose=verbose
                    )
                    self.update_annotations(
                        target_buffer=target_buffer, target_mapper=target_mapper,
                        coco_annotation=coco_annotation, unique_key=unique_key, verbose=verbose
                    )
    
    def write_split(self, verbose: bool=False):
        for target_dir, target_buffer, unique_key in \
            zip(
                [self.train_dir, self.test_dir, self.val_dir],
                [self.train_buffer, self.test_buffer, self.val_buffer],
                ['train', 'test', 'val']
            ):
                if verbose:
                    logger.info(f"Writting {unique_key}.json from split...")
                ann_dir = f"{target_dir}/{self.target_ann_dir}"
                output_path = f"{ann_dir}/{unique_key}.json"
                writer = COCO_Writer.from_buffer(buffer=target_buffer, output_path=output_path)
                json_dict = writer.build_json_dict(verbose=verbose)
                writer.write_json_dict(json_dict=json_dict, verbose=verbose)
        if verbose:
            logger.good(f"Writing of split annotations is complete.")