from __future__ import annotations
from typing import List
import json
import cv2
import numpy as np

from logger import logger
from common_utils.check_utils import check_required_keys, check_file_exists, \
    check_dir_exists, check_value, check_type_from_list, check_type
from common_utils.file_utils import file_exists
from common_utils.cv_drawing_utils import cv_simple_image_viewer, \
    draw_bbox, draw_keypoints, draw_segmentation, draw_skeleton, \
    draw_text_rows_at_point
from common_utils.common_types.point import Point2D_List
from common_utils.common_types.segmentation import Polygon, Segmentation
from common_utils.common_types.bbox import BBox
from common_utils.common_types.keypoint import Keypoint2D, Keypoint2D_List
from common_utils.path_utils import get_filename
from common_utils.time_utils import get_ctime

from .objects import COCO_Info
from .handlers import COCO_License_Handler, COCO_Image_Handler, \
    COCO_Annotation_Handler, COCO_Category_Handler, \
    COCO_License, COCO_Image, COCO_Annotation, COCO_Category
from .misc import KeypointGroup
from ....labelme.refactored import LabelmeAnnotationHandler, LabelmeAnnotation, LabelmeShapeHandler, LabelmeShape

class COCO_Dataset:
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
        return coco_dataset

    def copy(self) -> COCO_Dataset:
        return COCO_Dataset(
            info=self.info.copy(),
            licenses=self.licenses.copy(),
            images=self.images.copy(),
            annotations=self.annotations.copy(),
            categories=self.categories.copy()
        )

    @classmethod
    def new(cls, description: str=None) -> COCO_Dataset:
        coco_info = COCO_Info(description=description) if description is not None else COCO_Info()
        return COCO_Dataset(
            info=coco_info,
            licenses=COCO_License_Handler(),
            images=COCO_Image_Handler(),
            annotations=COCO_Annotation_Handler(),
            categories=COCO_Category_Handler()
        )

    def to_dict(self) -> dict:
        return {
            'info': self.info.to_dict(),
            'licenses': self.licenses.to_dict_list(),
            'images': self.images.to_dict_list(),
            'annotations': self.annotations.to_dict_list(),
            'categories': self.categories.to_dict_list()
        }

    @classmethod
    def from_dict(cls, dataset_dict: dict) -> COCO_Dataset:
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
            annotations=COCO_Annotation_Handler.from_dict_list(dataset_dict['annotations']),
            categories=COCO_Category_Handler.from_dict_list(dataset_dict['categories']),
        )

    def save_to_path(self, save_path: str, overwrite: bool=False):
        if file_exists(save_path) and not overwrite:
            logger.error(f'File already exists at save_path: {save_path}')
            raise Exception
        json_dict = self.to_dict()
        json.dump(json_dict, open(save_path, 'w'), indent=2, ensure_ascii=False)

    @classmethod
    def load_from_path(cls, json_path: str, img_dir: str=None, check_paths: bool=True) -> COCO_Dataset:
        check_file_exists(json_path)
        json_dict = json.load(open(json_path, 'r'))
        dataset = COCO_Dataset.from_dict(json_dict)
        if img_dir is not None:
            check_dir_exists(img_dir)
            for coco_image in dataset.images:
                coco_image.coco_url = f'{img_dir}/{coco_image.file_name}'
        if check_paths:
            for coco_image in dataset.images:
                check_file_exists(coco_image.coco_url)
        return dataset

    def to_labelme(self, priority: str='seg') -> LabelmeAnnotationHandler:
        check_value(priority, valid_value_list=['seg', 'bbox'])
        handler = LabelmeAnnotationHandler()
        for coco_image in self.images:
            labelme_ann = LabelmeAnnotation(
                img_path=coco_image.coco_url,
                img_h=coco_image.height, img_w=coco_image.width,
                shapes=LabelmeShapeHandler()
            )
            for coco_ann in self.annotations.get_annotations_from_imgIds([coco_image.id]):
                coco_cat = self.categories.get_category_from_id(coco_ann.category_id)
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
        self, labelme_handler: LabelmeAnnotationHandler,
        categories: COCO_Category_Handler,
        img_dir: str=None, remove_redundant: bool=True,
        ensure_no_unbounded_kpts: bool=True,
        ensure_valid_shape_type: bool=True,
        ignore_unspecified_categories: bool=False,
        license_url: str='https://github.com/cm107/annotation_utils/blob/master/LICENSE',
        license_name: str='MIT License'
    ) -> COCO_Dataset:
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

    def display_preview(
        self, draw_order: list=['seg', 'bbox', 'skeleton', 'kpt'], preview_start_idx: int=0,
        bbox_color: list=[0, 255, 255], bbox_thickness: list=2, # BBox
        bbox_show_label: bool=True, bbox_label_thickness: int=None,
        bbox_label_only: bool=False,
        seg_color: list=[255, 255, 0], seg_transparent: bool=True, # Segmentation
        kpt_radius: int=4, kpt_color: list=[0, 0, 255], # Keypoints
        show_kpt_labels: bool=True, kpt_label_thickness: int=1,
        kpt_label_only: bool=False, ignore_kpt_idx: list=[],
        kpt_idx_offset: int=0,
        skeleton_thickness: int=5, skeleton_color: list=[255, 0, 0] # Skeleton
    ):
        for i, coco_image in enumerate(self.images):
            if i < preview_start_idx:
                continue
            img = cv2.imread(coco_image.coco_url)
            for coco_ann in self.annotations.get_annotations_from_imgIds([coco_image.id]):
                vis_keypoints_arr = coco_ann.keypoints.to_numpy(demarcation=True)[:, :2]
                kpt_visibility = coco_ann.keypoints.to_numpy(demarcation=True)[:, 2:].reshape(-1)
                base_ignore_kpt_idx = np.argwhere(np.array(kpt_visibility) == 0.0).reshape(-1).tolist()
                ignore_kpt_idx_list = ignore_kpt_idx + list(set(base_ignore_kpt_idx) - set(ignore_kpt_idx))
                coco_cat = self.categories.get_category_from_id(coco_ann.category_id)
                for draw_target in draw_order:
                    if draw_target.lower() == 'bbox':
                        img = draw_bbox(
                            img=img, bbox=coco_ann.bbox, color=bbox_color, thickness=bbox_thickness, text=coco_cat.name,
                            label_thickness=bbox_label_thickness, label_only=bbox_label_only
                        )
                    elif draw_target.lower() == 'seg':
                        img = draw_segmentation(
                            img=img, segmentation=coco_ann.segmentation, color=seg_color, transparent=seg_transparent
                        )
                    elif draw_target.lower() == 'kpt':
                        img = draw_keypoints(
                            img=img, keypoints=vis_keypoints_arr,
                            radius=kpt_radius, color=kpt_color, keypoint_labels=coco_cat.keypoints,
                            show_keypoints_labels=show_kpt_labels, label_thickness=kpt_label_thickness,
                            label_only=kpt_label_only, ignore_kpt_idx=ignore_kpt_idx_list
                        )
                    elif draw_target.lower() == 'skeleton':
                        img = draw_skeleton(
                            img=img, keypoints=vis_keypoints_arr,
                            keypoint_skeleton=coco_cat.skeleton, index_offset=kpt_idx_offset,
                            thickness=skeleton_thickness, color=skeleton_color, ignore_kpt_idx=ignore_kpt_idx_list
                        )
                    else:
                        logger.error(f'Invalid target: {draw_target}')
                        logger.error(f"Valid targets: {['bbox', 'seg', 'kpt', 'skeleton']}")
                        raise Exception
            quit_flag = cv_simple_image_viewer(img=img, preview_width=1000)
            if quit_flag:
                break
