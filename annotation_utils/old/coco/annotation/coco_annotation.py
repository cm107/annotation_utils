from logger import logger
import json
from ..structs import COCO_Info, COCO_License_Handler, COCO_License, \
    COCO_Image_Handler, COCO_Image, COCO_Annotation_Handler, COCO_Annotation, \
    COCO_Category_Handler, COCO_Category
from ....coco.camera import Camera

class COCO_AnnotationFileParser:
    def __init__(self, annotation_path: str):
        self.annotation_path = annotation_path

        self.info = None
        self.licenses = None
        self.images = None
        self.annotations = None
        self.categories = None

    def load(self, verbose: bool=False):
        self.load_data()
        if verbose: logger.info("COCO Annotation Loaded")

    def get_info(self, info: dict) -> COCO_Info:
        description = info['description']
        url = info['url']
        version = info['version']
        year = info['year']
        contributor = info['contributor']
        date_created = info['date_created']
        return COCO_Info(
            description=description,
            url=url,
            version=version,
            year=year,
            contributor=contributor,
            date_created=date_created
        )

    def get_licenses(self, licenses: list) -> COCO_License_Handler:
        coco_license_handler = COCO_License_Handler()
        for license_dict in licenses:
            url = license_dict['url']
            id = license_dict['id']
            name = license_dict['name']
            coco_license = COCO_License(
                url=url,
                id=id,
                name=name
            )
            coco_license_handler.add(coco_license)
        return coco_license_handler

    def get_images(self, images: list) -> COCO_Image_Handler:
        coco_image_handler = COCO_Image_Handler()
        for image_dict in images:
            license_id = image_dict['license']
            file_name = image_dict['file_name']
            coco_url = image_dict['coco_url'] if 'coco_url' in image_dict else None
            height = image_dict['height']
            width = image_dict['width']
            date_captured = image_dict['date_captured']
            flickr_url = image_dict['flickr_url'] if 'flickr_url' in image_dict else None
            id = image_dict['id']
            coco_image = COCO_Image(
                license_id=license_id,
                file_name=file_name,
                coco_url=coco_url,
                height=height,
                width=width,
                date_captured=date_captured,
                flickr_url=flickr_url,
                id=id
            )
            coco_image_handler.add(coco_image)
        return coco_image_handler

    def get_annotations(self, annotations: list):
        coco_annotation_handler = COCO_Annotation_Handler()
        for annotation_dict in annotations:
            segmentation = annotation_dict['segmentation']
            num_keypoints = annotation_dict['num_keypoints']
            area = annotation_dict['area']
            iscrowd = annotation_dict['iscrowd']
            keypoints = annotation_dict['keypoints']
            keypoints_3d = annotation_dict['keypoints_3d'] if 'keypoints_3d' in annotation_dict else None # Custom Field
            camera_params = annotation_dict['camera_params'] if 'camera_params' in annotation_dict else None # Custom Field
            image_id = annotation_dict['image_id']
            bbox = annotation_dict['bbox']
            category_id = annotation_dict['category_id']
            id = annotation_dict['id']
            coco_annotation = COCO_Annotation(
                segmentation=segmentation,
                num_keypoints=num_keypoints,
                area=area,
                iscrowd=iscrowd,
                keypoints=keypoints,
                image_id=image_id,
                bbox=bbox,
                category_id=category_id,
                id=id,
                keypoints_3d=keypoints_3d,
                camera=Camera.from_dict(intrinsic_param_dict=camera_params) if camera_params is not None else None
            )
            coco_annotation_handler.add(coco_annotation)
        return coco_annotation_handler

    def get_categories(self, categories: list) -> COCO_Category_Handler:
        coco_category_handler = COCO_Category_Handler()
        for category in categories:
            supercategory = category['supercategory']
            id = category['id']
            name = category['name']
            keypoints = category['keypoints']
            skeleton = category['skeleton']
            coco_category = COCO_Category(
                supercategory=supercategory,
                id=id,
                name=name,
                keypoints=keypoints,
                skeleton=skeleton
            )
            coco_category_handler.add(coco_category)
        return coco_category_handler

    def load_data(self):
        data = json.load(open(self.annotation_path, 'r'))
        info = data['info']
        self.info = self.get_info(info)
        licenses = data['licenses']
        self.licenses = self.get_licenses(licenses)
        images = data['images']
        self.images = self.get_images(images)
        annotations = data['annotations']
        self.annotations = self.get_annotations(annotations)
        categories = data['categories'] if 'categories' in data else None
        self.categories = self.get_categories(categories)