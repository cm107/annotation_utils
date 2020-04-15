from __future__ import annotations
from typing import List
import json
import cv2
import numpy as np
from tqdm import tqdm

from logger import logger
from streamer.recorder import Recorder
from common_utils.check_utils import check_required_keys, check_file_exists, \
    check_dir_exists, check_value, check_type_from_list, check_type, \
    check_value_from_list
from common_utils.file_utils import file_exists, make_dir_if_not_exists, \
    get_dir_contents_len, delete_all_files_in_dir, copy_file
from common_utils.adv_file_utils import get_next_dump_path
from common_utils.path_utils import get_filename, get_dirpath_from_filepath, \
    get_extension_from_path, rel_to_abs_path, find_moved_abs_path, \
    get_extension_from_filename
from common_utils.cv_drawing_utils import \
    cv_simple_image_viewer, SimpleVideoViewer, \
    draw_bbox, draw_keypoints, draw_segmentation, draw_skeleton, \
    draw_text_rows_at_point
from common_utils.common_types.point import Point2D_List
from common_utils.common_types.segmentation import Polygon, Segmentation
from common_utils.common_types.bbox import BBox
from common_utils.common_types.keypoint import Keypoint2D, Keypoint2D_List
from common_utils.time_utils import get_ctime
from common_utils.image_utils import scale_to_max, pad_to_max

from .objects import COCO_Info
from .handlers import COCO_License_Handler, COCO_Image_Handler, \
    COCO_Annotation_Handler, COCO_Category_Handler, \
    COCO_License, COCO_Image, COCO_Annotation, COCO_Category
from .misc import KeypointGroup
from ....labelme.refactored import LabelmeAnnotationHandler, LabelmeAnnotation, LabelmeShapeHandler, LabelmeShape
from ....util.utils.coco import COCO_Mapper_Handler
from ....dataset.refactored.config import DatasetConfigCollectionHandler

class COCO_Dataset:
    """
    This is a class that can be thought of as a COCO dataset manipulation tool.

    info: Represents the 'info' section of the COCO annotation file.
            Basic information about the dataset is stored here.
    licenses: Represents the 'licenses' handler of the COCO annotation file.
                If any of the images are licensed, the license information is stored here.
                If none of the images are licensed, you must provide a one license (e.g. MIT License) for indexing purposes.
    images: Represents the 'images' handler of the COCO annotation file.
            All necessary image-related information is stored here.
    annotations: Represents the 'annotations' handler of the COCO annotation file.
                    All annotation data is stored here.
    categories: Represents the 'categories' handler of the COCO annotation file.
                All labeling conventions are specified here.

    To get started with using this class, you could create an empty COCO_Dataset:

        ```python
        # Call Constructor
        dataset = COCO_Dataset.new(description='This is a new test coco dataset')

        # Add a license
        dataset.licenses.append(COCO_License(...))
        # Add a category
        dataset.categories.append(COCO_Category(...))
        # Add Images
        dataset.images.append(COCO_Image(...))
        # Add Annotations
        dataset.annotations.append(COCO_Annotation(...))
        ...
        
        ```
    
    Or you could load an existing COCO dataset from a json file path:
        ```python
        dataset = COCO_Dataset.load_from_path('/path/to/coco/json/file.json')
        ```

    Once you have a complete COCO dataset, you can easily manipulate the dataset as needed:
        ```python
        # Example: Double The Size Of All Bounding Boxes
        for coco_image in dataset.images:
            for coco_ann in dataset.annotations:
                xmin, ymin, xmax, ymax = coco_ann.bbox.to_list()
                bbox_h, bbox_w = coco_ann.bbox.shape()
                xmin = xmin - 0.5 * bbox_w
                ymin = ymin - 0.5 * bbox_h
                xmax = xmax + 0.5 * bbox_w
                ymax = ymax + 0.5 * bbox_h
                xmin = 0 if xmin < 0 else coco_image.width - 1 if xmin >= coco_image.width else xmin
                ymin = 0 if ymin < 0 else coco_image.height - 1 if ymin >= coco_image.height else ymin
                xmax = 0 if xmax < 0 else coco_image.width - 1 if xmax >= coco_image.width else xmax
                ymax = 0 if ymax < 0 else coco_image.height - 1 if ymax >= coco_image.height else ymax
                coco_ann.bbox = BBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
        ```
    
    Finally, you can save your modified COCO dataset to a file as follows:
        ```python
        dataset.save_to_path('/save/path.json')
        ```

    Refer to the methods in this class for information about other functionality.
    """
    def __init__(
        self, info: COCO_Info, licenses: COCO_License_Handler, images: COCO_Image_Handler,
        annotations: COCO_Annotation_Handler, categories: COCO_Category_Handler
    ):
        self.info = info
        self.licenses = licenses
        self.images = images
        self.annotations = annotations
        self.categories = categories

    @classmethod
    def buffer(cls, coco_dataset: COCO_Dataset) -> COCO_Dataset:
        """
        A buffer that will return the same value, but mark the object as a COCO_Dataset object.
        This can be useful if your IDE doesn't recognize the type of your coco dataset object.
        
        coco_dataset: The object that you would like to send through the buffer.
        """
        return coco_dataset

    def copy(self) -> COCO_Dataset:
        """
        Copies the entirety of the COCO Dataset to a new object, which is located at a different
        location in memory.
        """
        return COCO_Dataset(
            info=self.info.copy(),
            licenses=self.licenses.copy(),
            images=self.images.copy(),
            annotations=self.annotations.copy(),
            categories=self.categories.copy()
        )

    @classmethod
    def new(cls, description: str=None) -> COCO_Dataset:
        """
        Create an empty COCO Dataset.
        description (optional): A description of the new dataset that you are creating.
        """
        coco_info = COCO_Info(description=description) if description is not None else COCO_Info()
        return COCO_Dataset(
            info=coco_info,
            licenses=COCO_License_Handler(),
            images=COCO_Image_Handler(),
            annotations=COCO_Annotation_Handler(),
            categories=COCO_Category_Handler()
        )

    def to_dict(self, strict: bool=True) -> dict:
        """
        Converts the COCO_Dataset object to a dictionary format, which is the standard format of COCO datasets.
        """
        return {
            'info': self.info.to_dict(),
            'licenses': self.licenses.to_dict_list(),
            'images': self.images.to_dict_list(),
            'annotations': self.annotations.to_dict_list(strict=strict),
            'categories': self.categories.to_dict_list(strict=strict)
        }

    @classmethod
    def from_dict(cls, dataset_dict: dict, strict: bool=True) -> COCO_Dataset:
        """
        Converts a coco dataset dictionary (the standard COCO format) to a COCO_Dataset class object.
        """
        check_required_keys(
            dataset_dict,
            required_keys=[
                'info', 'licenses', 'images',
                'annotations', 'categories'
            ]
        )
        return COCO_Dataset(
            info=COCO_Info.from_dict(dataset_dict['info']),
            licenses=COCO_License_Handler.from_dict_list(dataset_dict['licenses']),
            images=COCO_Image_Handler.from_dict_list(dataset_dict['images']),
            annotations=COCO_Annotation_Handler.from_dict_list(dataset_dict['annotations'], strict=strict),
            categories=COCO_Category_Handler.from_dict_list(dataset_dict['categories'], strict=strict)
        )

    def auto_fix_img_paths(self, src_container_dir: str, ignore_old_matches: bool=True):
        """
        Not yet implemented
        """
        raise NotImplementedError
        for coco_image in self.images:
            if not file_exists(coco_image.coco_url) or ignore_old_matches:
                fixed_path = find_moved_abs_path(
                    old_path=coco_image.coco_url, container_dir=src_container_dir,
                    get_first_match=False
                )
                if fixed_path is None:
                    logger.error(f"Couldn't any relative path in {coco_image.coco_url} inside of {src_container_dir}")
                    logger.error(f"Suggestion: Try adjusting src_container_dir to contain all required sources.")
                    raise Exception
                if file_exists(fixed_path):
                    coco_image.coco_url = fixed_path

    def move_images(
        self, dst_img_dir: str,
        preserve_filenames: bool=False, overwrite_duplicates: bool=False, update_img_paths: bool=True, overwrite: bool=False,
        show_pbar: bool=True
    ):
        """
        Combines all image directories specified in the coco_url of each coco image in self.images
        to a single image directory.

        dst_img_dir: The directory where you would like to save the combined image set.
        preserve_filenames: If False, unique filenames will be generated so as to not create a filename conflict.
        overwrite_duplicates: Only applicable when preserve_filenames=True.
                              In the event that two images with the same filename are moved to dst_img_dir from
                              two different source folders, an error will be raised if overwrite_duplicates=False.
                              If overwrite_duplicates=True, the second copy will be overwrite the first copy.
        update_img_paths: If True, all coco_url paths specified in self.images will be updated to reflect the new
                          combined image directory.
        overwrite: If True, all files in dst_img_dir will be deleted before copying images into the folder.
        """
        used_img_dir_list = []
        for coco_image in self.images:
            used_img_dir = get_dirpath_from_filepath(coco_image.coco_url)
            if used_img_dir not in used_img_dir_list:
                check_dir_exists(used_img_dir)
                used_img_dir_list.append(used_img_dir)

        if len(used_img_dir_list) == 0:
            logger.error(f"Couldn't parse used_img_dir_list.")
            logger.error(f"Are the coco_url paths in your dataset's image dictionary correct?")
            raise Exception

        make_dir_if_not_exists(dst_img_dir)
        if get_dir_contents_len(dst_img_dir) > 0:
            if overwrite:
                delete_all_files_in_dir(dst_img_dir, ask_permission=False)
            else:
                logger.error(f'dst_img_dir={dst_img_dir} is not empty.')
                logger.error('Please use overwrite=True if you would like to delete the contents before proceeding.')
                raise Exception

        pbar = tqdm(total=len(self.images), unit='image(s)') if show_pbar else None
        pbar.set_description(f'Moving Images...')
        for coco_image in self.images:
            if not preserve_filenames:
                img_extension = get_extension_from_path(coco_image.coco_url)
                dst_img_path = get_next_dump_path(
                    dump_dir=dst_img_dir,
                    file_extension=img_extension
                )
                dst_img_path = rel_to_abs_path(dst_img_path)
            else:
                img_filename = get_filename(coco_image.coco_url)
                dst_img_path = f'{dst_img_dir}/{img_filename}'
                if file_exists(dst_img_path) and not overwrite_duplicates:
                    logger.error(f'Failed to copy {coco_image.coco_url} to {dst_img_dir}')
                    logger.error(f'{img_filename} already exists in destination directory.')
                    logger.error(f'Hint: In order to use preserve_filenames=True, all filenames in the dataset must be unique.')
                    logger.error(
                        f'Suggestion: Either update the filenames to be unique or use preserve_filenames=False' + \
                        f' in order to automatically assign the destination filename.'
                    )
                    raise Exception
            copy_file(src_path=coco_image.coco_url, dest_path=dst_img_path, silent=True)
            if update_img_paths:
                coco_image.coco_url = dst_img_path
                coco_image.file_name = get_filename(dst_img_path)
            if pbar is not None:
                pbar.update(1)
        if pbar is not None:
            pbar.close()

    def save_to_path(self, save_path: str, overwrite: bool=False, strict: bool=True):
        """
        Save this COCO_Dataset object to a json file in the standard COCO format.
        
        save_path: Path of where you would like to save the dataset.
        overwrite: If True, any existing file that exists at save_path will be overwritten.
        """
        if file_exists(save_path) and not overwrite:
            logger.error(f'File already exists at save_path: {save_path}')
            raise Exception
        json_dict = self.to_dict(strict=strict)
        json.dump(json_dict, open(save_path, 'w'), indent=2, ensure_ascii=False)

    @classmethod
    def load_from_path(cls, json_path: str, img_dir: str=None, check_paths: bool=True, strict: bool=True) -> COCO_Dataset:
        """
        Loads a COCO_Dataset object from a COCO json file.

        json_path: Path to the COCO json file that you would like to load.
        img_dir: If not None, you can specify the image directory location of all images in this dataset.
                 Note: This can only be done if all image files are saved to the same directory.
                 Note: In order to create a dataset that has a unified image directory, use self.move_images
        check_paths: If True, all image paths will be checked as the dataset is loaded.
                     An error will be thrown if the corresponding image files do not exist.
        """
        check_file_exists(json_path)
        json_dict = json.load(open(json_path, 'r'))
        dataset = COCO_Dataset.from_dict(json_dict, strict=strict)
        if img_dir is not None:
            check_dir_exists(img_dir)
            for coco_image in dataset.images:
                coco_image.coco_url = f'{img_dir}/{coco_image.file_name}'
        if check_paths:
            for coco_image in dataset.images:
                check_file_exists(coco_image.coco_url)
        return dataset

    def to_labelme(self, priority: str='seg') -> LabelmeAnnotationHandler:
        """
        Convert a COCO_Dataset object to a LabelmeAnnotationHandler.
        The goal of this method is to convert a COCO formatted dataset to a Labelme formatted dataset.

        priority:
            'seg': Use segmentations to bound keypoints
            'bbox': Use bounding boxes to bound keypoints
        """
        check_value(priority, valid_value_list=['seg', 'bbox'])
        handler = LabelmeAnnotationHandler()
        for coco_image in self.images:
            labelme_ann = LabelmeAnnotation(
                img_path=coco_image.coco_url,
                img_h=coco_image.height, img_w=coco_image.width,
                shapes=LabelmeShapeHandler()
            )
            for coco_ann in self.annotations.get_annotations_from_imgIds([coco_image.id]):
                coco_cat = self.categories.get_obj_from_id(coco_ann.category_id)
                bbox_contains_seg = coco_ann.segmentation.within(coco_ann.bbox)
                if bbox_contains_seg and priority == 'seg':
                    for polygon in coco_ann.segmentation:
                        if len(polygon.to_list(demarcation=True)) < 3:
                            continue
                        labelme_ann.shapes.append(
                            LabelmeShape(
                                label=coco_cat.name,
                                points=Point2D_List.from_list(polygon.to_list(demarcation=True)),
                                shape_type='polygon'
                            )
                        )
                else:
                    labelme_ann.shapes.append(
                        LabelmeShape(
                            label=coco_cat.name,
                            points=coco_ann.bbox.to_point2d_list(),
                            shape_type='rectangle'
                        )
                    )
                if len(coco_ann.keypoints) > 0:
                    for i, kpt in enumerate(coco_ann.keypoints):
                        if kpt.visibility == 0:
                            continue
                        labelme_ann.shapes.append(
                            LabelmeShape(
                                label=coco_cat.keypoints[i],
                                points=Point2D_List.from_list([kpt.point.to_list()]),
                                shape_type='point'
                            )
                        )
            handler.append(labelme_ann)
        return handler

    @classmethod
    def from_labelme(
        cls, labelme_handler: LabelmeAnnotationHandler,
        categories: COCO_Category_Handler,
        img_dir: str=None, remove_redundant: bool=True,
        ensure_no_unbounded_kpts: bool=True,
        ensure_valid_shape_type: bool=True,
        ignore_unspecified_categories: bool=False,
        license_url: str='https://github.com/cm107/annotation_utils/blob/master/LICENSE',
        license_name: str='MIT License'
    ) -> COCO_Dataset:
        """
        Used to convert a LabelmeAnnotationHandler object to a COCO_Dataset object.
        This is meant to be used for converting a labelme dataset to a COCO dataset.

        labelme_handler: LabelmeAnnotationHandler object
        categories: COCO_Category_Handler object
        img_dir: Directory where all of the labelme dataset images are saved.
        remove_redundant: Remove bounding boxes that are contained withing segmentations and vice versa.
        ensure_no_unbounded_kpts: Ensures that all keypoints are bounded by either a bounding_box or segmentation.
        ensure_valid_shape_type: Ensures that all shape_type's in the LabelmeAnnotationHandler are valid.
        ignore_unspecified_categories: If true, all labels that are not specified in categories is ignored.
        license_url: The url of the license that you would like to associate with this converted dataset.
        license_name: The name of the license that is associated with this dataset.
        """
        dataset = COCO_Dataset.new(description='COCO Dataset converted from Labelme using annotation_utils')
        
        # Add a license to COCO Dataset
        dataset.licenses.append( # Assume that the dataset is free to use.
            COCO_License(
                url=license_url,
                name=license_name,
                id=0
            )
        )

        # Make sure at least one category is provided
        if type(categories) is list:
            categories = COCO_Category_Handler(category_list=categories)
        check_type(categories, valid_type_list=[COCO_Category_Handler])
        if len(categories) == 0:
            logger.error(f'Need to provide at least one COCO_Category for conversion to COCO format.')
            raise Exception
        category_names = [category.name for category in categories]

        # Add categories to COCO Dataset
        dataset.categories = categories

        for labelme_ann in labelme_handler:
            img_filename = get_filename(labelme_ann.img_path)
            if img_dir is not None:
                img_path = f'{img_dir}/{img_filename}'
            else:
                img_path = labelme_ann.img_path
            check_file_exists(img_path)
            
            kpt_label2points_list = {}
            bound_group_list = []
            poly_list = []
            poly_label_list = []
            bbox_list = []
            bbox_label_list = []

            if ensure_valid_shape_type:
                for shape in labelme_ann.shapes:
                    check_value(shape.shape_type, valid_value_list=['point', 'polygon', 'rectangle'])
            
            # Gather all segmentations
            for shape in labelme_ann.shapes:
                if shape.shape_type == 'polygon':
                    if shape.label not in category_names:
                        if ignore_unspecified_categories:
                            continue
                        else:
                            logger.error(f'shape.label={shape.label} does not exist in provided categories.')
                            logger.error(f'category_names: {category_names}')
                            raise Exception
                    poly_list.append(
                        Polygon.from_point2d_list(shape.points)
                    )
                    poly_label_list.append(shape.label)
            # Gather all bounding boxes
            for shape in labelme_ann.shapes:
                if shape.shape_type == 'rectangle':
                    if shape.label not in category_names:
                        if ignore_unspecified_categories:
                            continue
                        else:
                            logger.error(f'shape.label={shape.label} does not exist in provided categories.')
                            logger.error(f'category_names: {category_names}')
                            raise Exception
                    bbox_list.append(
                        BBox.from_point2d_list(shape.points)
                    )
                    bbox_label_list.append(shape.label)
            if remove_redundant:
                # Remove segmentation/bbox redundancies
                for poly in poly_list:
                    for i, [bbox, bbox_label] in enumerate(zip(bbox_list, bbox_label_list)):
                        if poly.contains(bbox):
                            del bbox_list[i]
                            del bbox_label_list[i]
                for bbox in bbox_list:
                    for i, [poly, poly_label] in enumerate(zip(poly_list, poly_label_list)):
                        if bbox.contains(poly):
                            del poly_list[i]
                            del poly_label_list[i]
            # Gather all keypoints
            for shape in labelme_ann.shapes:
                if shape.shape_type == 'point':
                    if shape.label not in kpt_label2points_list:
                        kpt_label2points_list[shape.label] = [shape.points[0]]
                    else:
                        kpt_label2points_list[shape.label].append(shape.points[0])

            # Group keypoints inside of polygon bounds
            for poly, poly_label in zip(poly_list, poly_label_list):
                coco_cat = dataset.categories.get_unique_category_from_name(poly_label)
                bound_group = KeypointGroup(bound_obj=poly, coco_cat=coco_cat)
                # Register the keypoints inside of each polygon
                for label, kpt_list in kpt_label2points_list.items():
                    for i, kpt in enumerate(kpt_list):
                        if kpt.within(poly):
                            bound_group.register(kpt=Keypoint2D(point=kpt, visibility=2), label=label)
                            del kpt_label2points_list[label][i]
                            if len(kpt_label2points_list[label]) == 0:
                                del kpt_label2points_list[label]
                            break
                bound_group_list.append(bound_group)
            # Group keypoints inside of bbox bounds
            for bbox, bbox_label in zip(bbox_list, bbox_label_list):
                coco_cat = dataset.categories.get_unique_category_from_name(bbox_label)
                bound_group = KeypointGroup(bound_obj=bbox, coco_cat=coco_cat)
                # Register the keypoints inside of each bounding box
                temp_dict = kpt_label2points_list.copy()
                for label, kpt_list in temp_dict.items():
                    for i, kpt in enumerate(kpt_list):
                        if kpt.within(bbox):
                            bound_group.register(kpt=Keypoint2D(point=kpt, visibility=2), label=label)
                            del kpt_label2points_list[label][i]
                            if len(kpt_label2points_list[label]) == 0:
                                del kpt_label2points_list[label]
                            break
                bound_group_list.append(bound_group)

            if ensure_no_unbounded_kpts:
                # Ensure that there are no leftover keypoints that are unbounded.
                # (This case often results from mistakes during annotation creation.)
                if len(kpt_label2points_list) > 0:
                    logger.error(f'The following keypoints were left unbounded:\n{kpt_label2points_list}')
                    logger.error(f'Image filename: {img_filename}')
                    raise Exception

            if len(bound_group_list) > 0:
                image_id = len(dataset.images)
                # Add image to COCO dataset images
                dataset.images.append(
                    COCO_Image(
                        license_id=0,
                        file_name=get_filename(img_path),
                        coco_url=img_path,
                        height=labelme_ann.img_h,
                        width=labelme_ann.img_w,
                        date_captured=get_ctime(img_path),
                        flickr_url=None,
                        id=image_id
                    )
                )

                # Add segmentation and/or bbox to COCO dataset annotations together with bounded keypoints
                for bound_group in bound_group_list:
                    keypoints = Keypoint2D_List()
                    for label in bound_group.coco_cat.keypoints:
                        label_found = False
                        for kpt, kpt_label in zip(bound_group.kpt_list, bound_group.kpt_label_list):
                            if kpt_label == label:
                                label_found = True
                                keypoints.append(kpt)
                                break
                        if not label_found:
                            keypoints.append(Keypoint2D.from_list([0, 0, 0]))
                    if type(bound_group.bound_obj) is Polygon:
                        bbox = bound_group.bound_obj.to_bbox()
                        dataset.annotations.append(
                            COCO_Annotation(
                                segmentation=Segmentation(polygon_list=[bound_group.bound_obj]),
                                num_keypoints=len(bound_group.coco_cat.keypoints),
                                area=bbox.area(),
                                iscrowd=0,
                                keypoints=keypoints,
                                image_id=image_id,
                                bbox=bbox,
                                category_id=bound_group.coco_cat.id,
                                id=len(dataset.annotations)
                            )
                        )
                    elif type(bound_group.bound_obj) is BBox:
                        dataset.annotations.append(
                            COCO_Annotation(
                                segmentation=Segmentation(polygon_list=[]),
                                num_keypoints=len(bound_group.coco_cat.keypoints),
                                area=bound_group.bound_obj.area(),
                                iscrowd=0,
                                keypoints=keypoints,
                                image_id=image_id,
                                bbox=bound_group.bound_obj,
                                category_id=bound_group.coco_cat.id,
                                id=len(dataset.annotations)
                            )
                        )
                    else:
                        raise Exception

        return dataset

    def update_img_dir(self, new_img_dir: str, check_paths: bool=True):
        """
        This method is used to update the image directory of all images in the dataset.
        Note that this assumes that all images are contained in the same directory.

        new_img_dir: The new image directory that all of your images are contained in.
        check_paths: When True, all image file paths are checked.
                     An error is thrown if an image cannot be found.
        """
        if check_paths:
            check_dir_exists(new_img_dir)

        for coco_image in self.images:
            coco_image.coco_url = f'{new_img_dir}/{coco_image.file_name}'
            if check_paths:
                check_file_exists(coco_image.coco_url)

    @classmethod
    def combine(cls, dataset_list: List[COCO_Dataset], img_dir_list: List[str]=None, show_pbar: bool=False) -> COCO_Dataset:
        """
        Combines a list of COCO_Dataset's into a single COCO_Dataset.

        dataset_list: A list of all of the COCO_Dataset objects that you would like to combine.
        img_dir_list: A list of all of the image directory paths that correspond to each COCO_Dataset in dataset_list.
        show_pbar: If True, a progress bar will be shown while the datasets are combined.
        """
        if img_dir_list is not None:
            if len(img_dir_list) != len(dataset_list):
                logger.error(f'len(img_dir_list) == {len(img_dir_list)} != {len(dataset_list)} == len(dataset_list)')
                raise Exception
            update_img_pbar = tqdm(total=len(img_dir_list), unit='dataset(s)') if show_pbar else None
            if update_img_pbar is not None:
                pbar.set_description(f'Updating Image Paths...')
            for img_dir, dataset in zip(img_dir_list, dataset_list):
                dataset = COCO_Dataset.buffer(dataset)
                dataset.update_img_dir(new_img_dir=img_dir, check_paths=True)
                if update_img_pbar is not None:
                    update_img_pbar.update(1)
            if update_img_pbar is not None:
                update_img_pbar.close()
        
        result_dataset = COCO_Dataset.new(
            description='A combination of many COCO datasets using annotation_utils'
        )
        map_handler = COCO_Mapper_Handler()
        merge_pbar = tqdm(total=len(dataset_list), unit='dataset(s)') if show_pbar else None
        if merge_pbar is not None:
            merge_pbar.set_description(f'Merging Datasets...')
        for i, dataset in enumerate(dataset_list):
            # Process Licenses
            for coco_license in dataset.licenses:
                already_exists = False
                for existing_license in result_dataset.licenses:
                    if coco_license.is_equal_to(existing_license, exclude_id=True):
                        already_exists = True
                        map_handler.license_mapper.add(
                            unique_key=i, old_id=coco_license.id, new_id=existing_license.id
                        )
                        break
                if not already_exists:
                    new_license = coco_license.copy()
                    new_license.id = len(result_dataset.licenses)
                    map_handler.license_mapper.add(
                        unique_key=i, old_id=coco_license.id, new_id=new_license.id
                    )
                    result_dataset.licenses.append(new_license)

            # Process Images
            for coco_image in dataset.images:
                check_file_exists(coco_image.coco_url)
                already_exists = False
                for existing_image in result_dataset.images:
                    if coco_image.is_equal_to(existing_image, exclude_id=True):
                        already_exists = True
                        map_handler.image_mapper.add(
                            unique_key=i, old_id=coco_image.id, new_id=existing_image.id
                        )
                        break
                if not already_exists:
                    new_image = coco_image.copy()
                    new_image.id = len(result_dataset.images)
                    map_handler.image_mapper.add(
                        unique_key=i, old_id=coco_image.id, new_id=new_image.id
                    )
                    found, new_image.license_id = map_handler.license_mapper.get_new_id(
                        unique_key=i, old_id=coco_image.license_id
                    )
                    if not found:
                        logger.error(f"Couldn't find license map using unique_key={i}, old_id={coco_image.license_id}")
                        raise Exception
                    result_dataset.images.append(new_image)

            # Process Categories
            for coco_category in dataset.categories:
                already_exists = False
                for existing_category in result_dataset.categories:
                    if coco_category.is_equal_to(existing_category, exclude_id=True):
                        already_exists = True
                        map_handler.category_mapper.add(
                            unique_key=i, old_id=coco_category.id, new_id=existing_category.id
                        )
                        break
                if not already_exists:
                    new_category = coco_category.copy()
                    new_category.id = len(result_dataset.categories)
                    map_handler.category_mapper.add(
                        unique_key=i, old_id=coco_category.id, new_id=new_category.id
                    )
                    result_dataset.categories.append(new_category)

            # Process Annotations
            for coco_ann in dataset.annotations:
                new_ann = coco_ann.copy()
                new_ann.id = len(result_dataset.annotations)
                found, new_ann.image_id = map_handler.image_mapper.get_new_id(
                    unique_key=i, old_id=coco_ann.image_id
                )
                if not found:
                    logger.error(f"Couldn't find image map using unique_key={i}, old_id={coco_ann.image_id}")
                    raise Exception
                found, new_ann.category_id = map_handler.category_mapper.get_new_id(
                    unique_key=i, old_id=coco_ann.category_id
                )
                if not found:
                    logger.error(f"Couldn't find category map using unique_key={i}, old_id={coco_ann.category_id}")
                    raise Exception
                result_dataset.annotations.append(new_ann)
            if merge_pbar is not None:
                merge_pbar.update(1)
        if merge_pbar is not None:
            merge_pbar.close()

        return result_dataset

    @classmethod
    def combine_from_config(cls, config_path: str, img_sort_attr_name: str=None, show_pbar: bool=False) -> COCO_Dataset:
        """
        This is the same as COCO_Dataset.combine, but with this method you don't have to construct each dataset manually.
        Instead, you can just provide a dataset configuration file that specifies the location of all of your coco json files
        ase well as their corresponding image directories.
        For more information about how to make this dataset configuration file, please refer to the DatasetConfigCollectionHandler class.

        config_path: The path to your dataset configuration file.
        img_sort_attr_name: The attribute name that you would like to sort the dataset images by before the datasets are combined.
                            (Example: img_sort_attr_name='file_name')
        show_pbar: If True, a progress bar will be shown while the images and annotations are loaded into the dataset.
        """

        dataset_path_config = DatasetConfigCollectionHandler.load_from_path(config_path)
        config_list = []
        for collection in dataset_path_config:
            for config in collection:
                check_value(config.ann_format, valid_value_list=['coco'])
                config_list.append(config)
        # dataset_dir_list, img_dir_list, ann_path_list, ann_format_list = dataset_path_config.get_paths()
        # check_value_from_list(item_list=ann_format_list, valid_value_list=['coco'])
        dataset_list = []
        # pbar = tqdm(total=len(img_dir_list), unit='dataset(s)') if show_pbar else None
        pbar = tqdm(total=len(config_list), unit='dataset(s)') if show_pbar else None
        if pbar is not None:
            pbar.set_description(f'Loading Dataset List...')
        # for img_dir, ann_path in zip(img_dir_list, ann_path_list):
        for config in config_list:
            # dataset = COCO_Dataset.load_from_path(json_path=ann_path, img_dir=img_dir, check_paths=True)
            dataset = COCO_Dataset.load_from_path(json_path=config.ann_path, img_dir=config.img_dir, check_paths=True)
            if img_sort_attr_name is not None:
                dataset.images.sort(attr_name=img_sort_attr_name)
            dataset_list.append(dataset)
            if pbar is not None:
                pbar.update(1)
        if pbar is not None:
            pbar.close()
        return COCO_Dataset.combine(dataset_list, show_pbar=show_pbar)

    def split(
        self, dest_dir: str,
        split_dirname_list: List[str]=['train', 'test', 'val'], ratio: list=[2, 1, 0], coco_filename_list: List[str]=None,
        shuffle: bool=True, preserve_filenames: bool=False, overwrite: bool=False
    ) -> List[COCO_Dataset]:
        """
        Use this method to split a single coco dataset into multiple datasets.
        This can be useful for when you need to split a single coco dataset into train, test, and validation datasets.

        dest_dir: Path to the folder where you would like to save the split datasets.
        split_dirname_list: Specify the folder names of each split dataset part that you would like to generate.
                            These folders are created in the folder specified by dest_dir.
                            For example, if you want to split your dataset into a train dataset and validation dataset,
                            use split_dirname_list=['train', 'val']
        ratio: Specify the ratio of images between each split dataset part.
               For example, if you would like to have twice as many train images as validation images, use ratio=[2, 1].
        coco_filename_list: Specify the filenames of the coco annotation file of each split dataset part.
                            Example: ['train.json', 'val.json']
        shuffle: If True, the images handler will be shuffled before splitting the dataset into parts.
        preserve_filenames: If True, the image filenames will be preserved during the split.
                            However, if a filename conflict is encountered as a result, an error will be thrown.
                            If False, the image filenames of each dataset part will be automatically generated
                            so as to avoid filename conflicts.
        overwrite: If True, the contents of dest_dir will be deleted before creating a new split dataset folder.
                   If False, an error will be thrown if dest_dir contains any files or directories.
        """

        # Checks
        check_type_from_list([split_dirname_list, ratio, coco_filename_list], valid_type_list=[list])
        if len(split_dirname_list) != len(ratio):
            logger.error(f'len(split_dirname_list) == {len(split_dirname_list)} != {len(ratio)} == len(ratio)')
            raise Exception
        check_type_from_list(split_dirname_list, valid_type_list=[str])
        check_type_from_list(ratio, valid_type_list=[int])
        if coco_filename_list is None:
            coco_filename_list = ['output.json'] * len(split_dirname_list)
        else:
            check_type_from_list(coco_filename_list, valid_type_list=[str])
            if len(coco_filename_list) != len(split_dirname_list):
                logger.error(f'len(coco_filename_list) == {len(coco_filename_list)} != {len(split_dirname_list)} == len(split_dirname_list)')
                raise Exception

        # Prepare Output Directory
        split_dirpath_list = [f'{dest_dir}/{split_dirname}' for split_dirname in split_dirname_list]
        split_imgdir_list = [f'{split_dirpath}/img' for split_dirpath in split_dirpath_list]
        split_cocodir_list = [f'{split_dirpath}/coco' for split_dirpath in split_dirpath_list]
        split_cocopath_list = [f'{split_cocodir}/{coco_filename}' for split_cocodir, coco_filename in zip(split_cocodir_list, coco_filename_list)]
        make_dir_if_not_exists(dest_dir)
        for split_dirpath, split_imgdir, split_cocodir in zip(split_dirpath_list, split_imgdir_list, split_cocodir_list):
            make_dir_if_not_exists(split_dirpath)
            if get_dir_contents_len(split_dirpath) > 0:
                if overwrite:
                    delete_all_files_in_dir(split_dirpath, ask_permission=False)
                else:
                    logger.error(f'Files/Directories were found in: {split_dirpath}')
                    logger.error('Use overwrite=True to overwrite all contents.')
                    raise Exception
            make_dir_if_not_exists(split_imgdir)
            make_dir_if_not_exists(split_cocodir)
            

        # Split COCO Images Into Samples
        locations = np.cumsum([val*int(len(self.images)/sum(ratio)) for val in ratio]) - 1
        start_location = None
        end_location = 0
        count = 0
        coco_image_samples = []
        if shuffle:
            self.images.shuffle()
        while count < len(locations):
            start_location = end_location
            end_location = locations[count]
            count += 1
            coco_image_samples.append(self.images[start_location:end_location].copy())

        # Construct New Datasets
        dataset_list = []
        for coco_image_list, split_dirname, split_imgdir, split_cocopath in \
            tqdm(zip(coco_image_samples, split_dirname_list, split_imgdir_list, split_cocopath_list), total=len(split_dirname_list), unit='part(s)', leave=True):
            dataset = COCO_Dataset.new(description=f'Split {split_dirname} Dataset')
            used_license_id_list = []
            used_category_id_list = []
            for coco_image0 in tqdm(coco_image_list, total=len(coco_image_list), unit='image(s)', leave=False):
                coco_image = coco_image0.copy()
                # Map Image Index
                coco_image = COCO_Image.buffer(coco_image)
                anns = self.annotations.get_annotations_from_imgIds([coco_image.id]).copy()
                new_image_id = len(dataset.images)
                
                # Copy Image
                old_img_path = coco_image.coco_url
                if not preserve_filenames:
                    img_extension = get_extension_from_filename(coco_image.file_name)
                    new_img_path = get_next_dump_path(dump_dir=split_imgdir, file_extension=img_extension)
                else:
                    new_img_path = f'{split_imgdir}/{coco_image.file_name}'
                if file_exists(new_img_path):
                    logger.error(f'Copy failed. Image already exists in destination directory: {new_img_path}')
                    logger.error(f'This is likely because the filenames in your dataset are not unique.')
                    logger.error(f'Use preserve_filenames=False to use automatically generated filenames.')
                    raise Exception
                copy_file(src_path=old_img_path, dest_path=new_img_path, silent=True)

                # Update COCO Image
                coco_image.id = new_image_id
                coco_image.coco_url = new_img_path
                coco_image.file_name = get_filename(new_img_path)
                dataset.images.append(coco_image)
                if coco_image.license_id not in used_license_id_list:
                    used_license_id_list.append(coco_image.license_id)

                for coco_ann0 in anns:
                    coco_ann = coco_ann0.copy()
                    # Map Annotation Index
                    new_ann_id = len(dataset.annotations)

                    # Update COCO Annotation
                    coco_ann.id = new_ann_id
                    coco_ann.image_id = new_image_id
                    dataset.annotations.append(coco_ann)
                    if coco_ann.category_id not in used_category_id_list:
                        used_category_id_list.append(coco_ann.category_id)

            # Add Used Licenses To Dataset and Update Ids
            for coco_license0 in self.licenses:
                coco_license = coco_license0.copy()
                if coco_license.id in used_license_id_list:
                    new_license_id = len(dataset.licenses)
                    for coco_image in dataset.images:
                        if coco_image.license_id == coco_license.id:
                            coco_image.license_id = new_license_id
                    coco_license.id = new_license_id
                    dataset.licenses.append(coco_license)
            
            # Add Used Categories To Dataset and Update Ids
            for coco_cat0 in self.categories:
                coco_cat = coco_cat0.copy()
                if coco_cat.id in used_category_id_list:
                    new_cat_id = len(dataset.categories)
                    for coco_ann in dataset.annotations:
                        if coco_ann.category_id == coco_cat.id:
                            coco_ann.category_id = new_cat_id
                    coco_cat.id = new_cat_id
                    dataset.categories.append(coco_cat)
            
            # Append Dataset To Split Dataset List
            dataset.save_to_path(save_path=split_cocopath, overwrite=False)
            dataset_list.append(dataset)
        return dataset_list

    def prune_keypoints(self, min_num_kpts: int, verbose: bool=False):
        """Used to prune out all of the annotations and images that contain below a certain level of keypoints, which is specified by min_num_kpts.
        
        Arguments:
            min_num_kpts {int} -- [Specify the mininum number of keypoints that can be allowed without the annotation being pruned.]
        
        Keyword Arguments:
            verbose {bool} -- [If True, prints some detailed information to the terminal.] (default: {False})

        Note:
            Using prune_keypoints by itself will not make any changes to image files or annotation files.
            It is merely pruning keypoints within the COCO_Dataset instance.
            In order to delete unnecessary image files and update the annotation file, using the following lines of code:
                dataset.prune_keypoints(min_num_kpts=[INT], verbose=True)
                dataset.move_images(dst_img_dir='/path/to/new/img_dir', preserve_filenames=True, update_img_paths=True, overwrite=True, show_pbar=True)
                dataset.save_to_path(save_path='/path/to/new/ann.json', overwrite=True)
            With this the new dataset is saved to a different location.
            In order to avoid accidently deleting files from the python script, please delete the old dataset files manually.
        """
        ann_idx_list = list(range(len(self.annotations)))
        ann_idx_list.reverse()
        for i in ann_idx_list:
            coco_ann = self.annotations[i]
            visibility_list = [kpt.visibility for kpt in coco_ann.keypoints]
            num_visible = 0
            for v in visibility_list:
                if v > 0.0:
                    num_visible += 1
            if num_visible < min_num_kpts:
                del self.annotations[i]
                if verbose:
                    logger.info(f'Deleted ann id: {coco_ann.id}')
        img_idx_list = list(range(len(self.images)))
        img_idx_list.reverse()
        for i in img_idx_list:
            coco_image = self.images[i]
            coco_anns = self.annotations.get_annotations_from_imgIds([coco_image.id])
            if len(coco_anns) == 0:
                coco_image = self.images[i]
                del self.images[i]
                if verbose:
                    logger.info(f'Deleted image id: {coco_image.id}')

    def remove_categories_by_name(self, category_names: List[str], verbose: bool=False):
        self.categories.remove_by_name(
            names=category_names,
            ann_handler=self.annotations,
            img_handler=self.images,
            license_handler=self.licenses,
            verbose=verbose
        )

    def print_handler_lengths(self):
        logger.info(f'len(licenses): {len(self.licenses)}')
        logger.info(f'len(images): {len(self.images)}')
        logger.info(f'len(annotations): {len(self.annotations)}')
        logger.info(f'len(categories): {len(self.categories)}')

    def draw_annotation(
        self, img: np.ndarray, ann_id: int,
        draw_order: list=['seg', 'bbox', 'skeleton', 'kpt'],
        bbox_color: list=[0, 255, 255], bbox_thickness: list=2, # BBox
        bbox_show_label: bool=True, bbox_label_thickness: int=None,
        bbox_label_only: bool=False,
        seg_color: list=[255, 255, 0], seg_transparent: bool=True, # Segmentation
        kpt_radius: int=4, kpt_color: list=[0, 0, 255], # Keypoints
        show_kpt_labels: bool=True, kpt_label_thickness: int=1,
        kpt_label_only: bool=False, ignore_kpt_idx: list=[],
        kpt_idx_offset: int=0,
        skeleton_thickness: int=5, skeleton_color: list=[255, 0, 0], # Skeleton
        details_corner_pos_ratio: float=0.02, details_height_ratio: float=0.10, # Details
        details_leeway: float=0.4, details_color: list=[255, 0, 255],
        details_thickness: int=2,
        show_bbox: bool=True, show_kpt: bool=True, # Show Flags
        show_skeleton: bool=True, show_seg: bool=True,
        show_details: bool=False
    ) -> np.ndarray:
        """
        Draws the annotation corresponding to ann_id on a given image.

        img: The image array that you would like to draw the annotation on.
        ann_id: The id that corresponds to the annotation that you would like to draw.
        draw_order: The order in which you would like to draw (render) the annotations.
                    Example: If you specify 'bbox' after 'seg', the bounding box will be
                    drawn after the segmentation is drawn.
        bbox_color: The color of the bbox that is to be drawn.
        bbox_thickness: The thickness of the bbox that is to be drawn.
        bbox_show_label: If True, the label of the bbox will be drawn directly above it.
        bbox_label_thickness: The thickness of the bbox label in the event that it is drawn.
        bbox_label_only: If you would rather not draw the bounding box and only show the label,
                         set this to True.
        seg_color: The color of the segmentation that is to be drawn.
        seg_transparent: If True, the segmentation that is drawn will be transparent.
                         If False, the segmentation will be a solid color.
        kpt_radius: The radius of the keypoints that are to be drawn.
        kpt_color: The color of the keypoints that are to be drawn.
        show_kpt_labels: If True, the labels of the keypoints will be drawn directly above each
                         keypoint.
        kpt_label_thickness: The thickness of the keypoint labels in the event that they are drawn.
        kpt_label_only: If True, the keypoints will not be drawn and only the keypoint labels will
                        be drawn.
        ignore_kpt_idx: A list of the keypoint indecies that you would like to skip when drawing
                        the keypoints. The skeleton segments connected to ignored keypoints will
                        also be excluded.
        kpt_idx_offset: If your keypoint skeleton indecies do not start at 0, you need to set an
                        offset so that the index will start at 0.
                        Example: If your keypoint index starts at 1, use kpt_idx_offset=-1.
        skeleton_thickness: The thickness of the skeleton segments that are to be drawn.
        skeleton_color: The color of the skeleton segments that are to be drawn.
        show_bbox: If False, the bbox will not be drawn at all.
        show_kpt: If False, the keypoints will not be drawn at all.
        show_skeleton: If False, the keypoint skeleton will not be drawn at all.
        show_seg: If False, the segmentation will not be drawn at all.
        """
        coco_ann = self.annotations.get_obj_from_id(ann_id)
        result = img.copy()

        if len(coco_ann.keypoints) > 0:
            vis_keypoints_arr = coco_ann.keypoints.to_numpy(demarcation=True)[:, :2]
            kpt_visibility = coco_ann.keypoints.to_numpy(demarcation=True)[:, 2:].reshape(-1)
            base_ignore_kpt_idx = np.argwhere(np.array(kpt_visibility) == 0.0).reshape(-1).tolist()
            ignore_kpt_idx_list = ignore_kpt_idx + list(set(base_ignore_kpt_idx) - set(ignore_kpt_idx))
        else:
            vis_keypoints_arr = np.array([])
            kpt_visibility = np.array([])
            ignore_kpt_idx_list = []
        coco_cat = self.categories.get_obj_from_id(coco_ann.category_id)
        for draw_target in draw_order:
            if draw_target.lower() == 'bbox':
                if show_bbox:
                    result = draw_bbox(
                        img=result, bbox=coco_ann.bbox, color=bbox_color, thickness=bbox_thickness, text=coco_cat.name,
                        label_thickness=bbox_label_thickness, label_only=bbox_label_only
                    )
            elif draw_target.lower() == 'seg':
                if show_seg:
                    result = draw_segmentation(
                        img=result, segmentation=coco_ann.segmentation, color=seg_color, transparent=seg_transparent
                    )
            elif draw_target.lower() == 'kpt':
                if show_kpt:
                    result = draw_keypoints(
                        img=result, keypoints=vis_keypoints_arr,
                        radius=kpt_radius, color=kpt_color, keypoint_labels=coco_cat.keypoints,
                        show_keypoints_labels=show_kpt_labels, label_thickness=kpt_label_thickness,
                        label_only=kpt_label_only, ignore_kpt_idx=ignore_kpt_idx_list
                    )
            elif draw_target.lower() == 'skeleton':
                if show_skeleton:
                    result = draw_skeleton(
                        img=result, keypoints=vis_keypoints_arr,
                        keypoint_skeleton=coco_cat.skeleton, index_offset=kpt_idx_offset,
                        thickness=skeleton_thickness, color=skeleton_color, ignore_kpt_idx=ignore_kpt_idx_list
                    )
            else:
                logger.error(f'Invalid target: {draw_target}')
                logger.error(f"Valid targets: {['bbox', 'seg', 'kpt', 'skeleton']}")
                raise Exception
        return result

    def get_preview(
        self, image_id: int,
        draw_order: list=['seg', 'bbox', 'skeleton', 'kpt'],
        bbox_color: list=[0, 255, 255], bbox_thickness: list=2, # BBox
        bbox_show_label: bool=True, bbox_label_thickness: int=None,
        bbox_label_only: bool=False,
        seg_color: list=[255, 255, 0], seg_transparent: bool=True, # Segmentation
        kpt_radius: int=4, kpt_color: list=[0, 0, 255], # Keypoints
        show_kpt_labels: bool=True, kpt_label_thickness: int=1,
        kpt_label_only: bool=False, ignore_kpt_idx: list=[],
        kpt_idx_offset: int=0,
        skeleton_thickness: int=5, skeleton_color: list=[255, 0, 0], # Skeleton
        details_corner_pos_ratio: float=0.02, details_height_ratio: float=0.10, # Details
        details_leeway: float=0.4, details_color: list=[255, 0, 255],
        details_thickness: int=2,
        show_bbox: bool=True, show_kpt: bool=True, # Show Flags
        show_skeleton: bool=True, show_seg: bool=True,
        show_details: bool=False
    ) -> np.ndarray:
        """
        Returns a preview of the image in the dataset that corresponds to image_id.
        Annotations are included in the preview.

        image_id: The id of the image that you would like to preview.
        draw_order: The order in which you would like to draw (render) the annotations.
                    Example: If you specify 'bbox' after 'seg', the bounding box will be
                    drawn after the segmentation is drawn.
        bbox_color: The color of the bbox that is to be drawn.
        bbox_thickness: The thickness of the bbox that is to be drawn.
        bbox_show_label: If True, the label of the bbox will be drawn directly above it.
        bbox_label_thickness: The thickness of the bbox label in the event that it is drawn.
        bbox_label_only: If you would rather not draw the bounding box and only show the label,
                         set this to True.
        seg_color: The color of the segmentation that is to be drawn.
        seg_transparent: If True, the segmentation that is drawn will be transparent.
                         If False, the segmentation will be a solid color.
        kpt_radius: The radius of the keypoints that are to be drawn.
        kpt_color: The color of the keypoints that are to be drawn.
        show_kpt_labels: If True, the labels of the keypoints will be drawn directly above each
                         keypoint.
        kpt_label_thickness: The thickness of the keypoint labels in the event that they are drawn.
        kpt_label_only: If True, the keypoints will not be drawn and only the keypoint labels will
                        be drawn.
        ignore_kpt_idx: A list of the keypoint indecies that you would like to skip when drawing
                        the keypoints. The skeleton segments connected to ignored keypoints will
                        also be excluded.
        kpt_idx_offset: If your keypoint skeleton indecies do not start at 0, you need to set an
                        offset so that the index will start at 0.
                        Example: If your keypoint index starts at 1, use kpt_idx_offset=-1.
        skeleton_thickness: The thickness of the skeleton segments that are to be drawn.
        skeleton_color: The color of the skeleton segments that are to be drawn.
        details_corner_pos_ratio: Ratio that determines where the details are positioned with respect
                                  to the upper lefthand cornder.
        details_height_ratio: The height ratio of the details box.
        details_leeway: The leeway between lines in the details box.
        details_color: The color of the text in the details box.
        details_thickness: The thickness of the text in the details box.
        show_bbox: If False, the bbox will not be drawn at all.
        show_kpt: If False, the keypoints will not be drawn at all.
        show_skeleton: If False, the keypoint skeleton will not be drawn at all.
        show_seg: If False, the segmentation will not be drawn at all.
        show_details: If True, the filename of the current frame and other information will be written to the screen.
        """
        coco_image = self.images.get_obj_from_id(image_id)
        img = cv2.imread(coco_image.coco_url)
        for coco_ann in self.annotations.get_annotations_from_imgIds([coco_image.id]):
            img = self.draw_annotation(
                img=img, ann_id=coco_ann.id,
                draw_order=draw_order,
                bbox_color=bbox_color, bbox_thickness=bbox_thickness, # BBox
                bbox_show_label=bbox_show_label, bbox_label_thickness=bbox_label_thickness,
                bbox_label_only=bbox_label_only,
                seg_color=seg_color, seg_transparent=seg_transparent, # Segmentation
                kpt_radius=kpt_radius, kpt_color=kpt_color, # Keypoints
                show_kpt_labels=show_kpt_labels, kpt_label_thickness=kpt_label_thickness,
                kpt_label_only=kpt_label_only, ignore_kpt_idx=ignore_kpt_idx,
                kpt_idx_offset=kpt_idx_offset,
                skeleton_thickness=skeleton_thickness, skeleton_color=skeleton_color, # Skeleton
                details_corner_pos_ratio=details_corner_pos_ratio,
                details_height_ratio=details_height_ratio,
                details_leeway=details_leeway, details_color=details_color,
                details_thickness=details_thickness,
                show_bbox=show_bbox, show_kpt=show_kpt,
                show_skeleton=show_skeleton, show_seg=show_seg,
                show_details=show_details
            )
        if show_details:
            img_h, img_w = img.shape[:2]
            coco_anns = self.annotations.get_annotations_from_imgIds([coco_image.id])
            coco_ann_id_list = [coco_ann.id for coco_ann in coco_anns]
            coco_ann_id_list.sort()
            img = draw_text_rows_at_point(
                img=img,
                row_text_list=[
                    f'{coco_image.file_name}',
                    f'image_id: {coco_image.id}',
                    f'ann_ids: {coco_ann_id_list}'
                ],
                x=int(details_corner_pos_ratio*img_w), y=int(details_corner_pos_ratio*img_h),
                combined_row_height=int(details_height_ratio*img_h),
                leeway=details_leeway,
                color=details_color,
                thickness=details_thickness
            )
        return img

    def display_preview(
        self,
        start_idx: int=0, end_idx: int=None, preview_width: int=1000,
        draw_order: list=['seg', 'bbox', 'skeleton', 'kpt'],
        bbox_color: list=[0, 255, 255], bbox_thickness: list=2, # BBox
        bbox_show_label: bool=True, bbox_label_thickness: int=None,
        bbox_label_only: bool=False,
        seg_color: list=[255, 255, 0], seg_transparent: bool=True, # Segmentation
        kpt_radius: int=4, kpt_color: list=[0, 0, 255], # Keypoints
        show_kpt_labels: bool=True, kpt_label_thickness: int=1,
        kpt_label_only: bool=False, ignore_kpt_idx: list=[],
        kpt_idx_offset: int=0,
        skeleton_thickness: int=5, skeleton_color: list=[255, 0, 0], # Skeleton
        details_corner_pos_ratio: float=0.02, details_height_ratio: float=0.10, # Details
        details_leeway: float=0.4, details_color: list=[255, 0, 255],
        details_thickness: int=2,
        show_bbox: bool=True, show_kpt: bool=True, # Show Flags
        show_skeleton: bool=True, show_seg: bool=True,
        show_details: bool=False
    ):
        """
        Displays a preview of the dataset in a popup window.
        Annotations are included in the preview.

        start_idx: The index that you would like to start previewing the dataset at.
                   Default: 0
        end_idx: The index that you would like to stop previewing the dataset at.
                 Defualt: None
        preview_width: The width of the window that you would like to display the
                       preview in. Default: 1000
        draw_order: The order in which you would like to draw (render) the annotations.
                    Example: If you specify 'bbox' after 'seg', the bounding box will be
                    drawn after the segmentation is drawn.
        bbox_color: The color of the bbox that is to be drawn.
        bbox_thickness: The thickness of the bbox that is to be drawn.
        bbox_show_label: If True, the label of the bbox will be drawn directly above it.
        bbox_label_thickness: The thickness of the bbox label in the event that it is drawn.
        bbox_label_only: If you would rather not draw the bounding box and only show the label,
                         set this to True.
        seg_color: The color of the segmentation that is to be drawn.
        seg_transparent: If True, the segmentation that is drawn will be transparent.
                         If False, the segmentation will be a solid color.
        kpt_radius: The radius of the keypoints that are to be drawn.
        kpt_color: The color of the keypoints that are to be drawn.
        show_kpt_labels: If True, the labels of the keypoints will be drawn directly above each
                         keypoint.
        kpt_label_thickness: The thickness of the keypoint labels in the event that they are drawn.
        kpt_label_only: If True, the keypoints will not be drawn and only the keypoint labels will
                        be drawn.
        ignore_kpt_idx: A list of the keypoint indecies that you would like to skip when drawing
                        the keypoints. The skeleton segments connected to ignored keypoints will
                        also be excluded.
        kpt_idx_offset: If your keypoint skeleton indecies do not start at 0, you need to set an
                        offset so that the index will start at 0.
                        Example: If your keypoint index starts at 1, use kpt_idx_offset=-1.
        skeleton_thickness: The thickness of the skeleton segments that are to be drawn.
        skeleton_color: The color of the skeleton segments that are to be drawn.
        details_corner_pos_ratio: Ratio that determines where the details are positioned with respect
                                  to the upper lefthand cornder.
        details_height_ratio: The height ratio of the details box.
        details_leeway: The leeway between lines in the details box.
        details_color: The color of the text in the details box.
        details_thickness: The thickness of the text in the details box.
        show_bbox: If False, the bbox will not be drawn at all.
        show_kpt: If False, the keypoints will not be drawn at all.
        show_skeleton: If False, the keypoint skeleton will not be drawn at all.
        show_seg: If False, the segmentation will not be drawn at all.
        show_details: If True, the filename of the current frame and other information will be written to the screen.
        """
        last_idx = len(self.images) if end_idx is None else end_idx
        for coco_image in self.images[start_idx:last_idx]:
            img = self.get_preview(
                image_id=coco_image.id,
                draw_order=draw_order,
                bbox_color=bbox_color, bbox_thickness=bbox_thickness, # BBox
                bbox_show_label=bbox_show_label, bbox_label_thickness=bbox_label_thickness,
                bbox_label_only=bbox_label_only,
                seg_color=seg_color, seg_transparent=seg_transparent, # Segmentation
                kpt_radius=kpt_radius, kpt_color=kpt_color, # Keypoints
                show_kpt_labels=show_kpt_labels, kpt_label_thickness=kpt_label_thickness,
                kpt_label_only=kpt_label_only, ignore_kpt_idx=ignore_kpt_idx,
                kpt_idx_offset=kpt_idx_offset,
                details_corner_pos_ratio=details_corner_pos_ratio,
                details_height_ratio=details_height_ratio,
                details_leeway=details_leeway, details_color=details_color,
                details_thickness=details_thickness,
                skeleton_thickness=skeleton_thickness, skeleton_color=skeleton_color, # Skeleton
                show_bbox=show_bbox, show_kpt=show_kpt,
                show_skeleton=show_skeleton, show_seg=show_seg,
                show_details=show_details
            )
            quit_flag = cv_simple_image_viewer(img=img, preview_width=preview_width)
            if quit_flag:
                break

    def save_visualization(
        self, save_dir: str='vis_preview', show_preview: bool=False, preserve_filenames: bool=True,
        show_annotations: bool=True, overwrite: bool=False,
        start_idx: int=0, end_idx: int=None, preview_width: int=1000,
        draw_order: list=['seg', 'bbox', 'skeleton', 'kpt'],
        bbox_color: list=[0, 255, 255], bbox_thickness: list=2, # BBox
        bbox_show_label: bool=True, bbox_label_thickness: int=None,
        bbox_label_only: bool=False,
        seg_color: list=[255, 255, 0], seg_transparent: bool=True, # Segmentation
        kpt_radius: int=4, kpt_color: list=[0, 0, 255], # Keypoints
        show_kpt_labels: bool=True, kpt_label_thickness: int=1,
        kpt_label_only: bool=False, ignore_kpt_idx: list=[],
        kpt_idx_offset: int=0,
        skeleton_thickness: int=5, skeleton_color: list=[255, 0, 0], # Skeleton
        details_corner_pos_ratio: float=0.02, details_height_ratio: float=0.10, # Details
        details_leeway: float=0.4, details_color: list=[255, 0, 255],
        details_thickness: int=2,
        show_bbox: bool=True, show_kpt: bool=True, # Show Flags
        show_skeleton: bool=True, show_seg: bool=True,
        show_details: bool=False
    ):
        """
        Generates and saves visualizations of the annotations of this dataset to a dump folder.

        save_dir: The directory where you would like to save all of your dataset visualizations.
        show_preview: If True, the visualizations will be displayed in a popup window as they
                      are being generated.
        preserve_filenames: If true, the visualization images generated will use the same filename
                            as the file_name field in each COCO_Image.
                            Otherwise a unique filename will be automatically generated.
        show_annotations: If True, the annotations will be drawn on the visualizations.
                          Otherwise, no annotations will be drawn.
        overwrite: If True, the files contained in the save_dir will be deleted being generating
                   the visualizations.
        start_idx: The index that you would like to start previewing the dataset at.
                   Default: 0
        end_idx: The index that you would like to stop previewing the dataset at.
                 Defualt: None
        preview_width: The width of the window that you would like to display the
                       preview in. Default: 1000
        draw_order: The order in which you would like to draw (render) the annotations.
                    Example: If you specify 'bbox' after 'seg', the bounding box will be
                    drawn after the segmentation is drawn.
        bbox_color: The color of the bbox that is to be drawn.
        bbox_thickness: The thickness of the bbox that is to be drawn.
        bbox_show_label: If True, the label of the bbox will be drawn directly above it.
        bbox_label_thickness: The thickness of the bbox label in the event that it is drawn.
        bbox_label_only: If you would rather not draw the bounding box and only show the label,
                         set this to True.
        seg_color: The color of the segmentation that is to be drawn.
        seg_transparent: If True, the segmentation that is drawn will be transparent.
                         If False, the segmentation will be a solid color.
        kpt_radius: The radius of the keypoints that are to be drawn.
        kpt_color: The color of the keypoints that are to be drawn.
        show_kpt_labels: If True, the labels of the keypoints will be drawn directly above each
                         keypoint.
        kpt_label_thickness: The thickness of the keypoint labels in the event that they are drawn.
        kpt_label_only: If True, the keypoints will not be drawn and only the keypoint labels will
                        be drawn.
        ignore_kpt_idx: A list of the keypoint indecies that you would like to skip when drawing
                        the keypoints. The skeleton segments connected to ignored keypoints will
                        also be excluded.
        kpt_idx_offset: If your keypoint skeleton indecies do not start at 0, you need to set an
                        offset so that the index will start at 0.
                        Example: If your keypoint index starts at 1, use kpt_idx_offset=-1.
        skeleton_thickness: The thickness of the skeleton segments that are to be drawn.
        skeleton_color: The color of the skeleton segments that are to be drawn.
        details_corner_pos_ratio: Ratio that determines where the details are positioned with respect
                                  to the upper lefthand cornder.
        details_height_ratio: The height ratio of the details box.
        details_leeway: The leeway between lines in the details box.
        details_color: The color of the text in the details box.
        details_thickness: The thickness of the text in the details box.
        show_bbox: If False, the bbox will not be drawn at all.
        show_kpt: If False, the keypoints will not be drawn at all.
        show_skeleton: If False, the keypoint skeleton will not be drawn at all.
        show_seg: If False, the segmentation will not be drawn at all.
        show_details: If True, the filename of the current frame and other information will be written to the screen.
        """

        # Prepare save directory
        make_dir_if_not_exists(save_dir)
        if get_dir_contents_len(save_dir) > 0:
            if not overwrite:
                logger.error(f'save_dir={save_dir} is not empty.')
                logger.error(f"Hint: If you want to erase the directory's contents, use overwrite=True")
                raise Exception
            delete_all_files_in_dir(save_dir, ask_permission=False)

        if show_preview:
            # Prepare Viewer
            viewer = SimpleVideoViewer(preview_width=1000, window_name='Annotation Visualization')

        last_idx = len(self.images) if end_idx is None else end_idx
        total_iter = len(self.images[start_idx:last_idx])
        for coco_image in tqdm(self.images[start_idx:last_idx], total=total_iter, leave=False):
            if show_annotations:
                img = self.get_preview(
                    image_id=coco_image.id,
                    draw_order=draw_order,
                    bbox_color=bbox_color, bbox_thickness=bbox_thickness, # BBox
                    bbox_show_label=bbox_show_label, bbox_label_thickness=bbox_label_thickness,
                    bbox_label_only=bbox_label_only,
                    seg_color=seg_color, seg_transparent=seg_transparent, # Segmentation
                    kpt_radius=kpt_radius, kpt_color=kpt_color, # Keypoints
                    show_kpt_labels=show_kpt_labels, kpt_label_thickness=kpt_label_thickness,
                    kpt_label_only=kpt_label_only, ignore_kpt_idx=ignore_kpt_idx,
                    kpt_idx_offset=kpt_idx_offset,
                    skeleton_thickness=skeleton_thickness, skeleton_color=skeleton_color, # Skeleton
                    details_corner_pos_ratio=details_corner_pos_ratio,
                    details_height_ratio=details_height_ratio,
                    details_leeway=details_leeway, details_color=details_color,
                    details_thickness=details_thickness,
                    show_bbox=show_bbox, show_kpt=show_kpt,
                    show_skeleton=show_skeleton, show_seg=show_seg,
                    show_details=show_details
                )
            else:
                img = cv2.imread(coco_image.coco_url)

            if preserve_filenames:
                save_path = f'{save_dir}/{coco_image.file_name}'
                if file_exists(save_path):
                    logger.error(f"Your dataset contains multiple instances of the same filename.")
                    logger.error(f"Either make all filenames unique or use preserve_filenames=False")
                    raise Exception
                cv2.imwrite(save_path, img)
            else:
                file_extension = get_extension_from_filename(coco_image.file_name)
                save_path = get_next_dump_path(dump_dir=save_dir, file_extension=file_extension)
                cv2.imwrite(save_path, img)

            if show_preview:
                quit_flag = viewer.show(img)
                if quit_flag:
                    break

    def save_video(
        self, save_path: str='viz.mp4', show_preview: bool=False,
        fps: int=20, rescale_before_pad: bool=True,
        show_annotations: bool=True, overwrite: bool=False,
        start_idx: int=0, end_idx: int=None, preview_width: int=1000,
        draw_order: list=['seg', 'bbox', 'skeleton', 'kpt'],
        bbox_color: list=[0, 255, 255], bbox_thickness: list=2, # BBox
        bbox_show_label: bool=True, bbox_label_thickness: int=None,
        bbox_label_only: bool=False,
        seg_color: list=[255, 255, 0], seg_transparent: bool=True, # Segmentation
        kpt_radius: int=4, kpt_color: list=[0, 0, 255], # Keypoints
        show_kpt_labels: bool=True, kpt_label_thickness: int=1,
        kpt_label_only: bool=False, ignore_kpt_idx: list=[],
        kpt_idx_offset: int=0,
        skeleton_thickness: int=5, skeleton_color: list=[255, 0, 0], # Skeleton
        details_corner_pos_ratio: float=0.02, details_height_ratio: float=0.10, # Details
        details_leeway: float=0.4, details_color: list=[255, 0, 255],
        details_thickness: int=2,
        show_bbox: bool=True, show_kpt: bool=True, # Show Flags
        show_skeleton: bool=True, show_seg: bool=True,
        show_details: bool=False
    ):
        """
        save_path: Path to where you would like to save the visualization video of this dataset.
        show_preview: If True, the visualizations will be displayed in a popup window as they
                      are being generated.
        fps: The frames per second of the output video saved to save_path.
        rescale_before_pad: Since all output frames need to be of the same dimensions, output images
                            need to be rescaled and padded in order to fit the frame.
                            If rescale_before_pad=True, output images will be rescaled to fit with
                            either the vertical or horizontal borders of the frame before applying
                            padding.
                            If rescale_before_pad=False, the output images will only be padded.
        show_annotations: If True, the annotations will be drawn on the visualizations.
                          Otherwise, no annotations will be drawn.
        overwrite: If True, the files contained in the save_dir will be deleted being generating
                   the visualizations.
        start_idx: The index that you would like to start previewing the dataset at.
                   Default: 0
        end_idx: The index that you would like to stop previewing the dataset at.
                 Defualt: None
        preview_width: The width of the window that you would like to display the
                       preview in. Default: 1000
        draw_order: The order in which you would like to draw (render) the annotations.
                    Example: If you specify 'bbox' after 'seg', the bounding box will be
                    drawn after the segmentation is drawn.
        bbox_color: The color of the bbox that is to be drawn.
        bbox_thickness: The thickness of the bbox that is to be drawn.
        bbox_show_label: If True, the label of the bbox will be drawn directly above it.
        bbox_label_thickness: The thickness of the bbox label in the event that it is drawn.
        bbox_label_only: If you would rather not draw the bounding box and only show the label,
                         set this to True.
        seg_color: The color of the segmentation that is to be drawn.
        seg_transparent: If True, the segmentation that is drawn will be transparent.
                         If False, the segmentation will be a solid color.
        kpt_radius: The radius of the keypoints that are to be drawn.
        kpt_color: The color of the keypoints that are to be drawn.
        show_kpt_labels: If True, the labels of the keypoints will be drawn directly above each
                         keypoint.
        kpt_label_thickness: The thickness of the keypoint labels in the event that they are drawn.
        kpt_label_only: If True, the keypoints will not be drawn and only the keypoint labels will
                        be drawn.
        ignore_kpt_idx: A list of the keypoint indecies that you would like to skip when drawing
                        the keypoints. The skeleton segments connected to ignored keypoints will
                        also be excluded.
        kpt_idx_offset: If your keypoint skeleton indecies do not start at 0, you need to set an
                        offset so that the index will start at 0.
                        Example: If your keypoint index starts at 1, use kpt_idx_offset=-1.
        skeleton_thickness: The thickness of the skeleton segments that are to be drawn.
        skeleton_color: The color of the skeleton segments that are to be drawn.
        details_corner_pos_ratio: Ratio that determines where the details are positioned with respect
                                  to the upper lefthand cornder.
        details_height_ratio: The height ratio of the details box.
        details_leeway: The leeway between lines in the details box.
        details_color: The color of the text in the details box.
        details_thickness: The thickness of the text in the details box.
        show_bbox: If False, the bbox will not be drawn at all.
        show_kpt: If False, the keypoints will not be drawn at all.
        show_skeleton: If False, the keypoint skeleton will not be drawn at all.
        show_seg: If False, the segmentation will not be drawn at all.
        show_details: If True, the filename of the current frame and other information will be written to the screen.
        """
        # Check Output Path
        if file_exists(save_path) and not overwrite:
            logger.error(f'File already exists at {save_path}')
            raise Exception

        # Prepare Video Writer
        dim_list = np.array([[coco_image.height, coco_image.width] for coco_image in self.images])
        max_h, max_w = dim_list.max(axis=0).tolist()
        recorder = Recorder(output_path=save_path, output_dims=(max_w, max_h), fps=fps)

        if show_preview:
            # Prepare Viewer
            viewer = SimpleVideoViewer(preview_width=1000, window_name='Annotation Visualization')

        last_idx = len(self.images) if end_idx is None else end_idx
        total_iter = len(self.images[start_idx:last_idx])
        for coco_image in tqdm(self.images[start_idx:last_idx], total=total_iter, leave=False):
            if show_annotations:
                img = self.get_preview(
                    image_id=coco_image.id,
                    draw_order=draw_order,
                    bbox_color=bbox_color, bbox_thickness=bbox_thickness, # BBox
                    bbox_show_label=bbox_show_label, bbox_label_thickness=bbox_label_thickness,
                    bbox_label_only=bbox_label_only,
                    seg_color=seg_color, seg_transparent=seg_transparent, # Segmentation
                    kpt_radius=kpt_radius, kpt_color=kpt_color, # Keypoints
                    show_kpt_labels=show_kpt_labels, kpt_label_thickness=kpt_label_thickness,
                    kpt_label_only=kpt_label_only, ignore_kpt_idx=ignore_kpt_idx,
                    kpt_idx_offset=kpt_idx_offset,
                    skeleton_thickness=skeleton_thickness, skeleton_color=skeleton_color, # Skeleton
                    details_corner_pos_ratio=details_corner_pos_ratio,
                    details_height_ratio=details_height_ratio,
                    details_leeway=details_leeway, details_color=details_color,
                    details_thickness=details_thickness,
                    show_bbox=show_bbox, show_kpt=show_kpt,
                    show_skeleton=show_skeleton, show_seg=show_seg,
                    show_details=show_details
                )
            else:
                img = cv2.imread(coco_image.coco_url)

            if rescale_before_pad:
                img = scale_to_max(img=img, target_shape=[max_h, max_w])
            img = pad_to_max(img=img, target_shape=[max_h, max_w])
            recorder.write(img)

            if show_preview:
                quit_flag = viewer.show(img)
                if quit_flag:
                    break
        recorder.close()