from __future__ import annotations
import json
from logger import logger
from ..structs import COCO_Info, COCO_License_Handler, \
    COCO_Image_Handler, COCO_Annotation_Handler, COCO_Category_Handler
from ...util.utils.coco import COCO_Field_Buffer

class COCO_Writer:
    def __init__(
        self, info: COCO_Info, licenses: COCO_License_Handler,
        images: COCO_Image_Handler, annotations: COCO_Annotation_Handler,
        categories: COCO_Category_Handler, output_path: str
    ):
        self.info = info
        self.licenses = licenses
        self.images = images
        self.annotations = annotations
        self.categories = categories

        self.output_path = output_path

    @classmethod
    def from_buffer(self, buffer: COCO_Field_Buffer, output_path: str) -> COCO_Writer:
        return COCO_Writer(
            info=buffer.info,
            licenses=buffer.licenses,
            images=buffer.images,
            annotations=buffer.annotations,
            categories=buffer.categories,
            output_path=output_path
        )

    def build_json_dict(self, verbose: bool=False) -> dict:
        info_dict = self.get_info_dict()
        licenses_list = self.get_licenses_list()
        images_list = self.get_images_list()
        annotations_list = self.get_annotations_list()
        categories_list = self.get_categories_list()

        json_dict = {}
        json_dict['info'] = info_dict
        json_dict['licenses'] = licenses_list
        json_dict['images'] = images_list
        json_dict['annotations'] = annotations_list
        json_dict['categories'] = categories_list
        if verbose: logger.info("COCO json has been built successfully.")
        return json_dict
    
    def get_info_dict(self) -> dict:
        info_dict = {}
        info_dict['description'] = self.info.description
        info_dict['url'] = self.info.url
        info_dict['version'] = self.info.version
        info_dict['year'] = self.info.year
        info_dict['contributor'] = self.info.contributor
        info_dict['date_created'] = self.info.date_created
        return info_dict

    def get_licenses_list(self) -> list:
        licenses_list = []
        for coco_license in self.licenses.license_list:
            license_dict = {}
            license_dict['url'] = coco_license.url
            license_dict['id'] = coco_license.id
            license_dict['name'] = coco_license.name
            licenses_list.append(license_dict)
        return licenses_list

    def get_images_list(self) -> list:
        images_list = []
        for coco_image in self.images.image_list:
            image_dict = {}
            image_dict['license'] = coco_image.license_id
            image_dict['file_name'] = coco_image.file_name
            image_dict['coco_url'] = coco_image.coco_url
            image_dict['height'] = coco_image.height
            image_dict['width'] = coco_image.width
            image_dict['date_captured'] = coco_image.date_captured
            image_dict['flickr_url'] = coco_image.flickr_url
            image_dict['id'] = coco_image.id
            images_list.append(image_dict)
        return images_list

    def get_annotations_list(self) -> list:
        annotations_list = []
        for coco_annotation in self.annotations.annotation_list:
            annotation_dict = {}
            annotation_dict['segmentation'] = coco_annotation.segmentation
            annotation_dict['num_keypoints'] = coco_annotation.num_keypoints
            annotation_dict['area'] = coco_annotation.area
            annotation_dict['iscrowd'] = coco_annotation.iscrowd
            annotation_dict['keypoints'] = coco_annotation.keypoints
            annotation_dict['image_id'] = coco_annotation.image_id
            annotation_dict['bbox'] = coco_annotation.bbox
            annotation_dict['category_id'] = coco_annotation.category_id
            annotation_dict['id'] = coco_annotation.id
            annotations_list.append(annotation_dict)
        return annotations_list

    def get_categories_list(self) -> list:
        categories_list = []
        for coco_category in self.categories.category_list:
            category_dict = {}
            category_dict['supercategory'] = coco_category.supercategory
            category_dict['id'] = coco_category.id
            category_dict['name'] = coco_category.name
            category_dict['keypoints'] = coco_category.keypoints
            category_dict['skeleton'] = coco_category.skeleton
            categories_list.append(category_dict)
        return categories_list

    def write_json_dict(self, json_dict: dict, verbose: bool=False):
        json.dump(json_dict, open(self.output_path, 'w'), indent=2, ensure_ascii=False)
        if verbose: logger.info(f"JSON dict has been written to:\n{self.output_path}")

    def test(self):
        json_dict = self.build_json_dict()
        self.write_json_dict(json_dict)