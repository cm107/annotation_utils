from ..logger.logger_handler import logger
from ..common_utils.common_types import Point, Keypoint, BoundingBox

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

class COCO_License:
    def __init__(self, url: str, id: int, name: str):
        self.url = url
        self.id = id
        self.name = name

    def __str__(self):
        return f"url: {self.url}, id: {self.id}, name: {self.name}"

    def __repr__(self):
        return self.__str__()

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

    def add(self, coco_license: COCO_License):
        self.license_list.append(coco_license)

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

    def add(self, coco_image: COCO_Image):
        self.image_list.append(coco_image)

    def get_images_from_imgIds(self, imgIds: list):		
	        return [x for x in self.image_list if x.id in imgIds]

class COCO_Annotation:
    def __init__(
        self, segmentation: dict, num_keypoints: int, area: int, iscrowd: int,
        keypoints: list, image_id: int, bbox: list, category_id: int, id: int
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

    def get_bounding_box(self) -> BoundingBox:
        return BoundingBox(self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3])

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

    def add(self, coco_annotation: COCO_Annotation):
        self.annotation_list.append(coco_annotation)

    def get_annotations_from_annIds(self, annIds: list):		
        return [x for x in self.annotation_list if x.id in annIds]		
        
    def get_annotations_from_imgIds(self, imgIds: list):		
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

    def get_label_skeleton(self):
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

    def add(self, coco_category: COCO_Category):
        self.category_list.append(coco_category)

    def get_categories_from_name(self, name: str):		
        return [x for x in self.category_list if x.name == name]

    def get_unique_category_from_name(self, name: str):
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

    def get_skeleton_from_name(self, name: str) -> (list, list):
        unique_category = self.get_unique_category_from_name(name)
        skeleton = unique_category.skeleton
        label_skeleton = unique_category.get_label_skeleton()
        return skeleton, label_skeleton