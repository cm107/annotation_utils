from __future__ import annotations
from typing import List
from logger import logger
from common_utils.common_types import Point, Keypoint, Rectangle
from common_utils.common_types.bbox import BBox
from common_utils.path_utils import get_extension_from_filename
from common_utils.check_utils import check_type

from ....coco.camera import Camera

class COCO_Info:
    def __init__(
        self, description: str, url: str, version: str,
        year: str, contributor: str, date_created: str
    ):
        self.description = description
        self.url = url
        self.version = version
        self.year = year
        self.contributor = contributor
        self.date_created = date_created

    def __str__(self):
        print_str = ""
        print_str += f"description:\n\t{self.description}\n"
        print_str += f"url:\n\t{self.url}\n"
        print_str += f"version:\n\t{self.version}\n"
        print_str += f"year:\n\t{self.year}\n"
        print_str += f"contributor:\n\t{self.contributor}\n"
        print_str += f"date_created:\n\t{self.date_created}\n"
        return print_str

    def __repr__(self):
        return self.__str__()

    @classmethod
    def buffer(self, coco_info: COCO_Info) -> COCO_Info:
        return coco_info

    def copy(self) -> COCO_Info:
        return COCO_Info(
            description=self.description,
            url=self.url,
            version=self.version,
            year=self.year,
            contributor=self.contributor,
            date_created=self.date_created
        )

class COCO_License:
    def __init__(self, url: str, id: int, name: str):
        self.url = url
        self.id = id
        self.name = name

    def __str__(self):
        return f"url: {self.url}, id: {self.id}, name: {self.name}"

    def __repr__(self):
        return self.__str__()

    @classmethod
    def buffer(self, coco_license: COCO_License) -> COCO_License:
        return coco_license

    def copy(self) -> COCO_License:
        return COCO_License(
            url=self.url,
            id=self.id,
            name=self.name
        )

class COCO_License_Handler:
    def __init__(self):
        self.license_list = []

    def __str__(self):
        print_str = ""
        for coco_license in self.license_list:
            print_str += f"{coco_license}\n"

        return print_str

    def __repr__(self):
        return self.__str__()

    def __len__(self) -> int:
        return len(self.license_list)

    def __getitem__(self, idx: int) -> COCO_License:
        if len(self.license_list) == 0:
            logger.error(f"COCO_License_Handler is empty.")
            raise IndexError
        elif idx < 0 or idx >= len(self.license_list):
            logger.error(f"Index out of range: {idx}")
            raise IndexError
        else:
            return self.license_list[idx]

    def __setitem__(self, idx: int, value: COCO_License):
        check_type(value, valid_type_list=[COCO_License])
        self.license_list[idx] = value

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self) -> COCO_License:
        if self.n < len(self.license_list):
            result = self.license_list[self.n]
            self.n += 1
            return result
        else:
            raise StopIteration

    def copy(self) -> COCO_License_Handler:
        result = COCO_License_Handler()
        result.license_list = self.license_list
        return result

    def add(self, coco_license: COCO_License):
        self.license_list.append(coco_license)

    def get_license_from_id(self, id: int) -> COCO_License:
        license_id_list = []
        for coco_license in self.license_list:
            if id == coco_license.id:
                return coco_license
            else:
                license_id_list.append(coco_license.id)
        license_id_list.sort()
        logger.error(f"Couldn't find coco_license with id={id}")
        logger.error(f"Possible ids: {license_id_list}")
        raise Exception

    def is_in_handler(self, coco_license: COCO_License, ignore_ids: bool=True, check_name_only: bool=False) -> bool:
        same_url = False
        same_id = False
        same_name = False

        found = False

        for existing_coco_license in self.license_list:
            same_url = True if existing_coco_license.url == coco_license.url else False
            same_id = True if existing_coco_license.id == coco_license.id else False
            same_name = True if existing_coco_license.name == coco_license.name else False
            if check_name_only:
                found = same_name
            else:
                if ignore_ids:
                    found = same_url and same_name
                else:
                    found = same_url and same_name and same_id
            if found:
                break
        return found

class COCO_Image:
    def __init__(
        self, license_id: int, file_name: str, coco_url: str,
        height: int, width: int, date_captured: str, flickr_url: str, id: int
    ):
        self.license_id = license_id
        self.file_name = file_name
        self.coco_url = coco_url
        self.height = height
        self.width = width
        self.date_captured = date_captured
        self.flickr_url = flickr_url
        self.id = id

    def __str__(self):
        print_str = "========================\n"
        print_str += f"license_id:\n\t{self.license_id}\n"
        print_str += f"file_name:\n\t{self.file_name}\n"
        print_str += f"coco_url:\n\t{self.coco_url}\n"
        print_str += f"height:\n\t{self.height}\n"
        print_str += f"width:\n\t{self.width}\n"
        print_str += f"date_captured:\n\t{self.date_captured}\n"
        print_str += f"flickr_url:\n\t{self.flickr_url}\n"
        print_str += f"id:\n\t{self.id}\n"
        return print_str

    def __repr__(self):
        return self.__str__()

    @classmethod
    def buffer(self, coco_image: COCO_Image) -> COCO_Image:
        return coco_image

    def copy(self) -> COCO_Image:
        return COCO_Image(
            license_id=self.license_id,
            file_name=self.file_name,
            coco_url=self.coco_url,
            height=self.height,
            width=self.width,
            date_captured=self.date_captured,
            flickr_url=self.flickr_url,
            id=self.id
        )

class COCO_Image_Handler:
    def __init__(self):
        self.image_list = []

    def __str__(self):
        print_str = ""
        for image in self.image_list:
            print_str += f"{image}\n"

        return print_str

    def __repr__(self):
        return self.__str__()

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, idx: int) -> COCO_Image:
        if len(self.image_list) == 0:
            logger.error(f"COCO_Image_Handler is empty.")
            raise IndexError
        elif idx < 0 or idx >= len(self.image_list):
            logger.error(f"Index out of range: {idx}")
            raise IndexError
        else:
            return self.image_list[idx]

    def __setitem__(self, idx: int, value: COCO_Image):
        check_type(value, valid_type_list=[COCO_Image])
        self.image_list[idx] = value

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self) -> COCO_Image:
        if self.n < len(self.image_list):
            result = self.image_list[self.n]
            self.n += 1
            return result
        else:
            raise StopIteration

    def copy(self) -> COCO_Image_Handler:
        result = COCO_Image_Handler()
        result.image_list = self.image_list
        return result

    def add(self, coco_image: COCO_Image):
        self.image_list.append(coco_image)

    def get_image_from_id(self, id: int) -> COCO_Image:
        image_id_list = []
        for coco_image in self.image_list:
            if id == coco_image.id:
                return coco_image
            else:
                image_id_list.append(coco_image.id)
        image_id_list.sort()
        logger.error(f"Couldn't find coco_image with id={id}")
        logger.error(f"Possible ids: {image_id_list}")
        raise Exception

    def get_images_from_file_name(self, file_name: str) -> List[COCO_Image]:
        image_list = []
        for coco_image in self.image_list:
            if file_name == coco_image.file_name:
                image_list.append(coco_image)
        return image_list

    def get_images_from_coco_url(self, coco_url: str) -> List[COCO_Image]:
        image_list = []
        for coco_image in self.image_list:
            if coco_url == coco_image.coco_url:
                image_list.append(coco_image)
        return image_list

    def get_images_from_flickr_url(self, flickr_url: str) -> List[COCO_Image]:
        image_list = []
        for coco_image in self.image_list:
            if flickr_url == coco_image.flickr_url:
                image_list.append(coco_image)
        return image_list

    def get_extensions(self) -> List[str]:
        extension_list = []
        for coco_image in self.image_list:
            extension = get_extension_from_filename(coco_image.file_name)
            if extension not in extension_list:
                extension_list.append(extension)
        return extension_list

    def get_images_from_imgIds(self, imgIds: list) -> List[COCO_Image]:		
	        return [x for x in self.image_list if x.id in imgIds]

    def is_in_handler(self, coco_image: COCO_Image, ignore_ids: bool=True, check_file_name_only: bool=False) -> bool:
        same_license_id = False
        same_file_name = False
        same_coco_url = False
        same_height = False
        same_width = False
        same_date_captured = False
        same_flickr_url = False
        same_id = False

        found = False

        for existing_coco_image in self.image_list:
            same_file_name = True if existing_coco_image.file_name == coco_image.file_name else False
            same_coco_url = True if existing_coco_image.coco_url == coco_image.coco_url else False
            same_height = True if existing_coco_image.height == coco_image.height else False
            same_width = True if existing_coco_image.width == coco_image.width else False
            same_date_captured = True if existing_coco_image.date_captured == coco_image.date_captured else False
            same_flickr_url = True if existing_coco_image.flickr_url == coco_image.flickr_url else False
            same_license_id = True if existing_coco_image.license_id == coco_image.license_id else False
            same_id = True if existing_coco_image.id == coco_image.id else False
            if check_file_name_only:
                found = same_file_name
            else:
                if ignore_ids:
                    found = same_file_name and same_coco_url and same_height and same_width and \
                        same_date_captured and same_flickr_url
                else:
                    found = same_file_name and same_coco_url and same_height and same_width and \
                        same_date_captured and same_flickr_url and same_license_id and same_id
            if found:
                break
        return found


class COCO_Annotation:
    def __init__(
        self, segmentation: dict, num_keypoints: int, area: int, iscrowd: int,
        keypoints: list, image_id: int, bbox: list, category_id: int, id: int,
        keypoints_3d: list=None, camera: Camera=None
    ):
        self.segmentation = segmentation
        self.encoded_format = True if type(segmentation) is dict and 'size' in segmentation and 'counts' in segmentation else False
        self.num_keypoints = num_keypoints
        self.area = area
        self.iscrowd = iscrowd
        self.keypoints = keypoints
        self.image_id = image_id
        self.bbox = bbox
        self.category_id = category_id
        self.id = id

        self.keypoints_3d = keypoints_3d
        self.camera = camera

        self.seg_point_lists = self.get_seg_point_lists()
        self.keypoint_list = self.get_keypoint_list()
        self.bounding_box = self.get_bounding_box()

    def __str__(self):
        print_str = "========================\n"
        print_str += f"encoded_format:\n\t{self.encoded_format}\n"
        print_str += f"segmentation:\n\t{self.segmentation}\n"
        print_str += f"num_keypoints:\n\t{self.num_keypoints}\n"
        print_str += f"area:\n\t{self.area}\n"
        print_str += f"iscrowd:\n\t{self.iscrowd}\n"
        print_str += f"keypoints:\n\t{self.keypoints}\n"
        print_str += f"image_id:\n\t{self.image_id}\n"
        print_str += f"bbox:\n\t{self.bbox}\n"
        print_str += f"category_id:\n\t{self.category_id}\n"
        print_str += f"id:\n\t{self.id}\n"
        return print_str

    def __repr__(self):
        return self.__str__()

    @classmethod
    def buffer(self, coco_annotation: COCO_Annotation) -> COCO_Annotation:
        return coco_annotation

    def copy(self) -> COCO_Annotation:
        return COCO_Annotation(
            segmentation=self.segmentation,
            num_keypoints=self.num_keypoints,
            area=self.area,
            iscrowd=self.iscrowd,
            keypoints=self.keypoints,
            image_id=self.image_id,
            bbox=self.bbox,
            category_id=self.category_id,
            id=self.id,
            keypoints_3d=self.keypoints_3d,
            camera=self.camera
        )

    def _chunk2pointlist(self, chunk: list) -> list:
        if len(chunk) % 2 != 0:
            logger.warning(f"Chunk is not evenly divisible by 2. len(chunk)={len(chunk)}")
            return []
        point_list = []
        for i in range(int(len(chunk)/2)):
            coord = chunk[2*i:2*(i+1)]
            point = Point(coord[0], coord[1])
            point_list.append(point)
        return point_list

    def get_seg_point_lists(self) -> list:
        if self.encoded_format:
            # TODO: Figure out how to convert the encoded format to the standard format.
            #logger.warning(f"Not implemented yet for encoded segmentation format.")
            return []
        else:
            point_lists = []
            for value_list in self.segmentation:
                pointlist = self._chunk2pointlist(value_list)
                point_lists.append(pointlist)
            return point_lists

    def get_keypoint_list(self) -> list:
        if self.keypoints is None:
            if self.num_keypoints > 0:
                logger.warning(f"No keypoints provided to COCO_Annotation constructor. Expecting {self.num_keypoints}.")
            return []
        if len(self.keypoints) % 3 != 0:
            logger.warning(f"self.keypoints is not evenly divisible by 3. len(self.keypoints)={len(self.keypoints)}")
            return []
        keypoint_list = []
        for i in range(int(len(self.keypoints)/3)):
            coord = self.keypoints[3*i:3*(i+1)]
            keypoint = Keypoint(coord[0], coord[1], coord[2])
            keypoint_list.append(keypoint)
        return keypoint_list

    def get_keypoint_3d_list(self) -> list:
        if self.keypoints_3d is None:
            return []
        if len(self.keypoints_3d) % 4 != 0:
            logger.warning(f"self.keypoints_3d is not evenly divisible by 4. len(self.keypoints_3d)={len(self.keypoints_3d)}")
            return []
        keypoint_3d_list = []
        for i in range(int(len(self.keypoints_3d)/4)):
            coord = self.keypoints_3d[4*i:4*(i+1)]
            keypoint_3d_list.append(coord)
        return keypoint_3d_list

    def get_bounding_box(self) -> BBox:
        return BBox.from_list([self.bbox[0], self.bbox[1], self.bbox[0]+self.bbox[2], self.bbox[1]+self.bbox[3]])

    def get_rect(self) -> Rectangle:
        return Rectangle(self.bbox[0], self.bbox[1], self.bbox[0]+self.bbox[2], self.bbox[1]+self.bbox[3])

class COCO_Annotation_Handler:
    def __init__(self):
        self.annotation_list = []

    def __str__(self):
        print_str = ""
        for annotation in self.annotation_list:
            print_str += f"{annotation}\n"

        return print_str

    def __repr__(self):
        return self.__str__()

    def __len__(self) -> int:
        return len(self.annotation_list)

    def __getitem__(self, idx: int) -> COCO_Annotation:
        if len(self.annotation_list) == 0:
            logger.error(f"COCO_Annotation_Handler is empty.")
            raise IndexError
        elif idx < 0 or idx >= len(self.annotation_list):
            logger.error(f"Index out of range: {idx}")
            raise IndexError
        else:
            return self.annotation_list[idx]

    def __setitem__(self, idx: int, value: COCO_Annotation):
        check_type(value, valid_type_list=[COCO_Annotation])
        self.annotation_list[idx] = value

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self) -> COCO_Annotation:
        if self.n < len(self.annotation_list):
            result = self.annotation_list[self.n]
            self.n += 1
            return result
        else:
            raise StopIteration

    def copy(self) -> COCO_Annotation_Handler:
        result = COCO_Annotation_Handler()
        result.annotation_list = self.annotation_list
        return result

    def add(self, coco_annotation: COCO_Annotation):
        self.annotation_list.append(coco_annotation)

    def get_annotation_from_id(self, id: int) -> COCO_Annotation:
        annotation_id_list = []
        for coco_annotation in self.annotation_list:
            if id == coco_annotation.id:
                return coco_annotation
            else:
                annotation_id_list.append(coco_annotation.id)
        annotation_id_list.sort()
        logger.error(f"Couldn't find coco_annotation with id={id}")
        logger.error(f"Possible ids: {annotation_id_list}")
        raise Exception

    def get_annotations_from_annIds(self, annIds: list) -> List[COCO_Annotation]:		
        return [x for x in self.annotation_list if x.id in annIds]		
        
    def get_annotations_from_imgIds(self, imgIds: list) -> List[COCO_Annotation]:		
        return [x for x in self.annotation_list if x.image_id in imgIds]

class COCO_Category:
    def __init__(
        self, supercategory: str, id: int, name: str, keypoints: list, skeleton: list
    ):
        self.supercategory = supercategory
        self.id = id
        self.name = name
        self.keypoints = keypoints
        self.skeleton = skeleton

        self.label_skeleton = self.get_label_skeleton()

    def __str__(self):
        print_str = "========================\n"
        print_str += f"supercategory:\n\t{self.supercategory}\n"
        print_str += f"id:\n\t{self.id}\n"
        print_str += f"name:\n\t{self.name}\n"
        print_str += f"keypoints:\n\t{self.keypoints}\n"
        print_str += f"skeleton:\n\t{self.skeleton}\n"
        return print_str

    def __repr__(self):
        return self.__str__()

    @classmethod
    def buffer(self, coco_category: COCO_Category) -> COCO_Category:
        return coco_category

    def copy(self) -> COCO_Category:
        return COCO_Category(
            supercategory=self.supercategory,
            id=self.id,
            name=self.name,
            keypoints=self.keypoints,
            skeleton=self.skeleton
        )

    def get_label_skeleton(self) -> list:
        str_skeleton = []
        for int_bone in self.skeleton:
            bone_start = self.keypoints[int_bone[0]-1] 
            bone_end = self.keypoints[int_bone[1]-1]
            str_bone = [bone_start, bone_end]
            str_skeleton.append(str_bone)
        return str_skeleton

class COCO_Category_Handler:
    def __init__(self):
        self.category_list = []

    def __str__(self):
        print_str = ""
        for category in self.category_list:
            print_str += f"{category}\n"

        return print_str

    def __repr__(self):
        return self.__str__()

    def __len__(self) -> int:
        return len(self.category_list)

    def __getitem__(self, idx: int) -> COCO_Category:
        if len(self.category_list) == 0:
            logger.error(f"COCO_Category_Handler is empty.")
            raise IndexError
        elif idx < 0 or idx >= len(self.category_list):
            logger.error(f"Index out of range: {idx}")
            raise IndexError
        else:
            return self.category_list[idx]

    def __setitem__(self, idx: int, value: COCO_Category):
        check_type(value, valid_type_list=[COCO_Category])
        self.category_list[idx] = value

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self) -> COCO_Category:
        if self.n < len(self.category_list):
            result = self.category_list[self.n]
            self.n += 1
            return result
        else:
            raise StopIteration

    def copy(self) -> COCO_Category_Handler:
        result = COCO_Category_Handler()
        result.category_list = self.category_list
        return result

    def add(self, coco_category: COCO_Category):
        self.category_list.append(coco_category)

    def get_categories_from_name(self, name: str) -> List[COCO_Category]:		
        return [x for x in self.category_list if x.name == name]

    def get_unique_category_from_name(self, name: str) -> COCO_Category:
        found_categories = self.get_categories_from_name(name)
        if len(found_categories) == 0:
            logger.error(f"Couldn't find any categories by the name: {name}")
            raise Exception
        elif len(found_categories) > 1:
            logger.error(f"Found {len(found_categories)} categories with the name {name}")
            logger.error(f"Found Categories:")
            for category in found_categories:
                logger.error(category)
            raise Exception
        return found_categories[0]

    def get_category_from_id(self, id: int) -> COCO_Category:
        category_id_list = []
        for coco_category in self.category_list:
            if id == coco_category.id:
                return coco_category
            else:
                category_id_list.append(coco_category.id)
        category_id_list.sort()
        logger.error(f"Couldn't find coco_category with id={id}")
        logger.error(f"Possible ids: {category_id_list}")
        raise Exception

    def get_skeleton_from_name(self, name: str) -> (list, list):
        unique_category = self.get_unique_category_from_name(name)
        skeleton = unique_category.skeleton
        label_skeleton = unique_category.get_label_skeleton()
        return skeleton, label_skeleton

    def is_in_handler(self, coco_category: COCO_Category, ignore_ids: bool=True, check_name_only: bool=True) -> bool:
        same_supercategory = False
        same_id = False
        same_name = False
        same_keypoints = False
        same_skeleton = False

        found = False

        for existing_coco_category in self.category_list:
            same_supercategory = True if existing_coco_category.supercategory == coco_category.supercategory else False
            same_id = True if existing_coco_category.id == coco_category.id else False
            same_name = True if existing_coco_category.name == coco_category.name else False
            same_keypoints = True if existing_coco_category.keypoints == coco_category.keypoints else False
            same_skeleton = True if existing_coco_category.skeleton == coco_category.skeleton else False
            if check_name_only:
                found = same_name
            else:
                if ignore_ids:
                    found = same_supercategory and same_name and same_keypoints and same_skeleton
                else:
                    found = same_supercategory and same_name and same_keypoints and same_skeleton and same_id
            if found:
                break
        return found