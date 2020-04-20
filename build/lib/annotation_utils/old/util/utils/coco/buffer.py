from __future__ import annotations
from logger import logger
from common_utils.time_utils import get_present_year, get_present_time_Ymd
from common_utils.user_utils import get_username
from common_utils.path_utils import get_filename

from ....coco.structs import COCO_Info, \
    COCO_License, COCO_Image, COCO_Annotation, COCO_Category, \
    COCO_License_Handler, COCO_Image_Handler, \
    COCO_Annotation_Handler, COCO_Category_Handler
from ....coco.annotation.coco_annotation import COCO_AnnotationFileParser
from .id_map import COCO_Mapper_Handler

class COCO_Field_Buffer:
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
    def from_scratch(self, description: str, url: str, version: str='1.0') -> COCO_Field_Buffer:
        info = COCO_Info(
            description=description,
            url=url,
            version=version,
            year=get_present_year(),
            contributor=get_username(),
            date_created=get_present_time_Ymd()
        )
        licenses = COCO_License_Handler()
        images = COCO_Image_Handler()
        annotations = COCO_Annotation_Handler()
        categories = COCO_Category_Handler()
        return COCO_Field_Buffer(
            info=info, licenses=licenses,
            images=images, annotations=annotations,
            categories=categories
        )

    @classmethod
    def from_parser(
        self, parser: COCO_AnnotationFileParser,
        new_description: str=None, new_url: str=None, new_version: str='1.0'
    ) -> COCO_Field_Buffer:
        info = parser.info
        info.description = new_description if new_description is not None else info.description
        info.url = new_url if new_url is not None else info.url
        info.version = new_version if new_version is not None else info.version

        return COCO_Field_Buffer(
            info=info,
            licenses=parser.licenses,
            images=parser.images,
            annotations=parser.annotations,
            categories=parser.categories
        )

    def update_info(self, description: str, url: str, version: str='1.0'):
        self.info = COCO_Info(
            description=description,
            url=url,
            version=version,
            year=self.info.year,
            contributor=self.info.contributor,
            date_created=self.info.date_created
        )

    def update_fields(
        self, licenses: COCO_License_Handler, images: COCO_Image_Handler,
        annotations: COCO_Annotation_Handler, categories: COCO_Category_Handler,
        info: COCO_Info=None
    ):
        self.licenses = licenses
        self.images = images
        self.annotations = annotations
        self.categories = categories
        if info is not None:
            self.info = info

    def process_license(self, coco_license: COCO_License, id_mapper: COCO_Mapper_Handler, unique_key: str) -> (bool, int, int):
        pending_coco_license = coco_license.copy()

        found = False
        added_new = False
        found_coco_license = None
        for loaded_coco_license in self.licenses.license_list:
            if pending_coco_license.url == loaded_coco_license.url and pending_coco_license.name == loaded_coco_license.name:
                found = True
                found_coco_license = loaded_coco_license
                break
        old_id = pending_coco_license.id
        if not found:
            new_id = len(self.licenses.license_list)
            pending_coco_license.id = new_id
            self.licenses.add(pending_coco_license)
            added_new = True
        else: # Add map only
            new_id = found_coco_license.id

        id_mapper.license_mapper.add(
            unique_key=unique_key,
            old_id=old_id,
            new_id=new_id
        )
        return added_new, old_id, new_id

    def process_image(self, coco_image: COCO_Image, id_mapper: COCO_Mapper_Handler, unique_key: str, img_dir: str=None, update_img_path: bool=False) -> (bool, int, int):
        pending_coco_image = coco_image.copy()

        if update_img_path:
            if img_dir is None:
                logger.error(f"img_dir is required in order to update the img_path")
                raise Exception
            coco_url_filename = get_filename(pending_coco_image.coco_url)
            if coco_url_filename != pending_coco_image.file_name:
                logger.error(f"coco_url_filename == {coco_url_filename} != pending_coco_image.file_name == {pending_coco_image.file_name}")
                raise Exception
            pending_coco_image.coco_url = f"{img_dir}/{coco_url_filename}"

        # No checks should be necessary
        old_id = pending_coco_image.id
        new_id = len(self.images.image_list)

        # Get License Id
        found, new_license_id = id_mapper.license_mapper.get_new_id(
            unique_key=unique_key, old_id=pending_coco_image.license_id
        )
        if not found:
            logger.error(f"Couldn't find unique_key: {unique_key}, old_id: {pending_coco_image.license_id} pair in license_mapper.")
            raise Exception

        # Update Id References
        pending_coco_image.license_id = new_license_id
        pending_coco_image.id = new_id
        
        self.images.add(pending_coco_image)
        added_new = True
        id_mapper.image_mapper.add(
            unique_key=unique_key,
            old_id=old_id,
            new_id=new_id
        )
        return added_new, old_id, new_id

    def process_annotation(self, coco_annotation: COCO_Annotation, id_mapper: COCO_Mapper_Handler, unique_key: str) -> (bool, int, int):
        pending_coco_annotation = coco_annotation.copy()
        
        # No checks should be necessary
        old_id = pending_coco_annotation.id
        new_id = len(self.annotations.annotation_list)
        
        # Get Image Id
        found, new_image_id = id_mapper.image_mapper.get_new_id(
            unique_key=unique_key, old_id=pending_coco_annotation.image_id
        )
        if not found:
            logger.error(f"Couldn't find unique_key: {unique_key}, old_id: {pending_coco_annotation.image_id} pair in image_mapper.")
            raise Exception

        # Get Category Id
        found, new_category_id = id_mapper.category_mapper.get_new_id(
            unique_key=unique_key, old_id=pending_coco_annotation.category_id
        )
        if not found:
            logger.error(f"Couldn't find unique_key: {unique_key}, old_id: {pending_coco_annotation.category_id} pair in category_mapper.")
            raise Exception

        # Update Id References
        pending_coco_annotation.image_id = new_image_id
        pending_coco_annotation.category_id = new_category_id
        pending_coco_annotation.id = new_id
        
        self.annotations.add(pending_coco_annotation)
        added_new = True
        id_mapper.annotation_mapper.add(
            unique_key=unique_key,
            old_id=old_id,
            new_id=new_id
        )
        return added_new, old_id, new_id

    def process_category(self, coco_category: COCO_Category, id_mapper: COCO_Mapper_Handler, unique_key: str) -> (bool, int, int):
        pending_coco_category = coco_category.copy()
        
        found = False
        added_new = False
        found_coco_category = None
        for loaded_coco_category in self.categories.category_list:
            if pending_coco_category.supercategory == loaded_coco_category.supercategory and \
                pending_coco_category.name == loaded_coco_category.name:
                found = True
                found_coco_category = loaded_coco_category
                break
        old_id = pending_coco_category.id
        if not found:
            new_id = len(self.categories.category_list)
            pending_coco_category.id = new_id
            self.categories.add(pending_coco_category)
            added_new = True
        else: # Add map only
            new_id = found_coco_category.id

        id_mapper.category_mapper.add(
            unique_key=unique_key,
            old_id=old_id,
            new_id=new_id
        )
        return added_new, old_id, new_id