from __future__ import annotations
import json
import numpy as np
import os
from pathlib import Path
from typing import List
from logger import logger
from common_utils.path_utils import rel_to_abs_path, get_script_dir, get_all_files_of_extension
from common_utils.common_types.bbox import BBox
from common_utils.check_utils import check_list_length, check_type_from_list
from common_utils.common_types.segmentation import Segmentation
import shutil
from coco_base import CocoInfo, CocoLicense, CocoImage, CocoAnnotation, CocoCategory, CocoDataset
import datetime
import cv2

class CameraParam:
    def __init__(self,f: [float], c: List[float], T: List[float]):
        self.f = f
        self.c = c
        self.T = T
    
    def __str__(self):
        return f"camera intrinsics ({self.f},{self.c}, {self.T})"
    
    def __repr__(self):
        return self.__str__
    
    def to_dict(self):
        return self.__dict__



class Point2D:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __str__(self):
        return f"Point2D({self.x},{self.y})"

    def __repr__(self):
        return self.__str__()

    @classmethod
    def from_list(self, coords: list) -> Point2D:
        check_list_length(coords, correct_length=2)
        return Point2D(x=coords[0], y=coords[1])

    def to_array(self):
        return [self.x, self.y]

class Cuboid2D:
    def __init__(self, point_list: list):
        check_list_length(point_list, correct_length=8)
        check_type_from_list(point_list, valid_type_list=[Point2D])
        self.point_list = point_list

    def __str__(self):
        return f"Cuboid2D({self.point_list})"

    def __repr__(self):
        return self.__str__()

    @classmethod
    def from_list(self, coords_list: list) -> Cuboid2D:
        return Cuboid2D(point_list=[Point2D.from_list(coords=coords) for coords in coords_list])

class Point3D:
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return f"Point3D({self.x},{self.y},{self.z})"

    def __repr__(self):
        return self.__str__()

    @classmethod
    def from_list(self, coords: list) -> Point3D:
        check_list_length(coords, correct_length=3)
        return Point3D(x=coords[0], y=coords[1], z=coords[2])
    

    def to_array(self):
        return [self.x, self.y, self.z]

class Cuboid3D:
    def __init__(self, point_list: list):
        check_list_length(point_list, correct_length=8)
        check_type_from_list(point_list, valid_type_list=[Point3D])
        self.point_list = point_list

    def __str__(self):
        return f"Cuboid3D({self.point_list})"

    def __repr__(self):
        return self.__str__()

    @classmethod
    def from_list(self, coords_list: list) -> Cuboid3D:
        return Cuboid3D(point_list=[Point3D.from_list(coords=coords) for coords in coords_list])

class Quaternion:
    def __init__(self, x: float, y: float, z: float, w: float):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    @classmethod
    def from_list(self, coords: list) -> Quaternion:
        check_list_length(coords, correct_length=4)
        return Quaternion(x=coords[0], y=coords[1], z=coords[2], w=coords[3])

    def __str__(self):
        return f"Quaternion({self.x},{self.y},{self.z},{self.w})"

    def __repr__(self):
        return self.__str__()
    
    def to_dict(self):
        return self.__dict__

class NDDS_Annotation_Object:
    def __init__(
        self,
        class_name: str, instance_id: int, visibility: int, location: Point3D, quaternion_xyzw: Quaternion,
        pose_transform: np.ndarray, cuboid_centroid: Point3D, projected_cuboid_centroid: Point2D,
        bounding_box: BBox, cuboid: Cuboid3D, projected_cuboid: Cuboid2D
    ):
        self.class_name = class_name
        self.instance_id = instance_id
        self.visibility = visibility
        self.location = location
        self.quaternion_xyzw = quaternion_xyzw
        self.pose_transform = pose_transform
        self.cuboid_centroid = cuboid_centroid
        self.projected_cuboid_centroid = projected_cuboid_centroid
        self.bounding_box = bounding_box
        self.cuboid = cuboid
        self.projected_cuboid = projected_cuboid

    def __str__(self):
        return f"NDDS_Annotation_Object({self.__dict__})"

    def __repr__(self):
        return self.__str__()
    
    @classmethod
    def from_object_dict(self, object_dict: dict) -> NDDS_Annotation_Object:
        return NDDS_Annotation_Object(
            class_name=object_dict['class'],
            instance_id=object_dict['instance_id'],
            visibility=object_dict['visibility'],
            location=Point3D.from_list(object_dict['location']),
            quaternion_xyzw=Quaternion.from_list(object_dict['quaternion_xyzw']),
            pose_transform=np.array(object_dict['pose_transform']),
            cuboid_centroid=Point3D.from_list(object_dict['cuboid_centroid']),
            projected_cuboid_centroid=Point2D.from_list(object_dict['projected_cuboid_centroid']),
            bounding_box=BBox.from_list(object_dict['bounding_box']['top_left']+object_dict['bounding_box']['bottom_right']),
            cuboid=Cuboid3D.from_list(object_dict['cuboid']),
            projected_cuboid=Cuboid2D.from_list(object_dict['projected_cuboid'])
        )



class Custom_Object_From_NDDS():
    def __init__(
        self,
        class_name: str, instance_id: int, visibility: int, location: Point3D, quaternion_xyzw: Quaternion,
        pose_transform: np.ndarray, cuboid_centroid: Point3D, projected_cuboid_centroid: Point2D,
        bounding_box: BBox, cuboid: Cuboid3D, projected_cuboid: Cuboid2D
        ):
        self.a = 1


class CocoNDDSConverter:
    def __init__(self, data_dir: str, info_dict: dict, license_dict_list: List[dict], category_dict_list:List[dict], save_path: str):
        self.data_dir = data_dir
        self.save_path = os.path.abspath(save_path)
        
        self.coco_info = CocoInfo().from_dict(coco_dict=info_dict)
        self.coco_license_list = [CocoLicense().from_dict(coco_dict=license_dict) for license_dict in license_dict_list]
        self.coco_image_list = []
        self.coco_annotation_list= []
        self.coco_category_list = [CocoCategory().from_dict(coco_dict=category_dict) for category_dict in category_dict_list]
    

    def get_all_object_json_files(self):

        json_path_list = get_all_files_of_extension(dir_path=self.data_dir, extension='json')
        json_path_list = [json_path for json_path in json_path_list if '_camera' not in json_path]
        json_path_list = [json_path for json_path in json_path_list if '_object' not in json_path]
        json_path_list.sort()
        return json_path_list

    def get_camera_settings(self):

        json_path_list = get_all_files_of_extension(dir_path=self.data_dir, extension='json')
        camera_settings_path = [json_path for json_path in json_path_list if '_camera' in json_path][0]
        camera_settings_json = json.load(open(camera_settings_path, 'r'))["camera_settings"]
        return camera_settings_json
    

    def process(self):

        object_name = np.array([item.name for item in self.coco_category_list])
        json_path_list = self.get_all_object_json_files()
        camera_settings_json = self.get_camera_settings()

        for i, json_path in enumerate(json_path_list):
            logger.blue(f"{i}: {json_path}")
            ann_dict = json.load(open(json_path, 'r'))
            camera_dict = ann_dict['camera_data']
            camera_instrinsic_settings = camera_settings_json[0]["intrinsic_settings"]
            camera_params = CameraParam(f=[camera_instrinsic_settings["fx"], camera_instrinsic_settings["fy"]], c=[camera_instrinsic_settings["cx"], camera_instrinsic_settings["cy"]], T=camera_dict['location_worldframe'])
            location_worldframe = Point3D.from_list(coords=camera_dict['location_worldframe'])
            quaternion_worldframe = Quaternion.from_list(coords=camera_dict['quaternion_xyzw_worldframe'])

            logger.cyan(f"location_worldframe: {location_worldframe}")
            logger.cyan(f"quaternion_worldframe: {quaternion_worldframe}")

            ann_object_list = ann_dict['objects']
            image_location = os.path.abspath(json_path[:-5]+'.png')

            coco_image = CocoImage(license=1, file_name=os.path.basename(image_location), coco_url=image_location, height= camera_settings_json[0]["captured_image_size"]["height"], width= camera_settings_json[0]["captured_image_size"]["width"], date_captured=today, flickr_url=None, id=i)
            self.coco_image_list.append(coco_image)
            # coco_annotation(bbox=[1,1,1,1], image_id= i, category_id= 0, is_crowd=0, id=i)
            ndds_ann_object_list = []
            # print(coco_image)
            for ann_object in ann_object_list:
                ndds_ann_object = NDDS_Annotation_Object.from_object_dict(ann_object)
                if any(ele in ndds_ann_object.class_name for ele in object_name):
                    mask = np.array([ele in ndds_ann_object.class_name for ele in object_name])
                    object_name_single = object_name[mask][0]

                    category_id = [category.id for category in self.coco_category_list if category.name == object_name_single][0]
                    object_index = ndds_ann_object.class_name.replace(object_name_single, '')

                    # get segmentation color
                    RGBint = ndds_ann_object.instance_id
                    pixel_b =  RGBint & 255
                    pixel_g = (RGBint >> 8) & 255
                    pixel_r =   (RGBint >> 16) & 255
                    color_instance_rgb = [pixel_b,pixel_g,pixel_r]

                    # get segmentation and bbox
                    instance_img = cv2.imread(coco_image.coco_url.replace('.png', '.is.png'))
                        
                    bgr_lower, bgr_upper = (int(color_instance_rgb[0]-1), int(color_instance_rgb[1]-1), int(color_instance_rgb[2]-1)), (int(color_instance_rgb[0]+1), int(color_instance_rgb[1]+1), int(color_instance_rgb[2]+1))
                    color_mask = cv2.inRange(src=instance_img, lowerb=bgr_lower, upperb=bgr_upper)
                    color_contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    seg = Segmentation.from_contour(contour_list=color_contours)
                        
                    if len(seg) == 0:
                        print("image segmentation not found, please check color code")
                        continue
                            
                    seg_bbox = seg.to_bbox()
                        
                    outer_xmin = float(seg_bbox.xmin)
                    outer_ymin = float(seg_bbox.ymin)
                    outer_xmax = float(seg_bbox.xmax)
                    outer_ymax = float(seg_bbox.ymax)
                    outer_bbox = BBox(xmin=outer_xmin, ymin=outer_ymin, xmax=outer_xmax, ymax=outer_ymax)
                    outer_bbox_h, outer_bbox_w = outer_bbox.shape()
                    outer_bbox_area = outer_bbox.area()

                    bbox_final =[outer_bbox.xmin, outer_bbox.ymin, outer_bbox_w, outer_bbox_h]

                    if outer_bbox_area < 10:
                        print("object too small to be considered")
                        continue
                    
                    # try to get keypoints
                    keypoint_dict = {}
                    keypoint_3d_dict = {}
                    for ann_object in ann_object_list:
                        ndds_keypoint_object = NDDS_Annotation_Object.from_object_dict(ann_object)
                        if "point" in ndds_keypoint_object.class_name:
                            point_name = ndds_keypoint_object.class_name.replace("point", "")
                            keypoint_list = [category.keypoints for category in self.coco_category_list if category.name == object_name_single][0]
                            if object_index in point_name:
                                point_name = point_name.replace(object_index,"")
                                keypoint_dict.update({
                                    point_name: ndds_keypoint_object.projected_cuboid_centroid
                                })
                                keypoint_3d_dict.update({
                                    point_name: ndds_keypoint_object.cuboid_centroid
                                })

                    keypoints = [item for sublist in [[keypoint_dict[item].x, keypoint_dict[item].y, 2] for item in keypoint_list] for item in sublist]
                    keypoints_3d = [item for sublist in [[keypoint_3d_dict[item].x, keypoint_3d_dict[item].y, keypoint_3d_dict[item].z, 2] for item in keypoint_list] for item in sublist]
                    coco_annotation = CocoAnnotation(bbox=bbox_final, image_id= i, category_id= category_id, is_crowd=0, id=len(self.coco_annotation_list)+1, 
                                                    keypoints=keypoints, keypoints_3d = keypoints_3d, segmentation=seg.to_list(), area=outer_bbox_area,
                                                    orientation_xyzw=ndds_ann_object.quaternion_xyzw.to_dict(), camera_params=camera_params.to_dict(), num_keypoints=len(keypoints))
                    self.coco_annotation_list.append(coco_annotation)
        
        self.save_images()
       
        return CocoDataset(info=self.coco_info, licenses=self.coco_license_list, images=self.coco_image_list, annotations=self.coco_annotation_list, categories=self.coco_category_list), self.save_path

    def save_images(self):

        save_path_dir = os.path.dirname(os.path.abspath(self.save_path))
        Path(save_path_dir).mkdir(parents=True, exist_ok=True)
        # move images
        for images in self.coco_image_list:
            shutil.copy(images.coco_url, f'{save_path_dir}/'+images.file_name)
            images.coco_url = save_path_dir+images.file_name

        CocoDataset(info=self.coco_info, licenses=self.coco_license_list, images=self.coco_image_list, annotations=self.coco_annotation_list, categories=self.coco_category_list).save_to_path(save_path=save_location)

if __name__ == '__main__': 
    # data location
    test_dir = rel_to_abs_path(get_script_dir())
    data_dir = f"{test_dir}/../../../HSR"

    # save location
    save_location = "../../../hsr_coco/annot-coco.json"

    # initiate coco info, license, and category
    today = datetime.datetime.now().strftime("%Y/%m/%d")
    year = datetime.datetime.now().strftime("%Y")

    info_dict = {
        "description": "HSR 2020 Dataset",
        "url": "",
        "version": "1.0",
        "year": int(year),
        "contributor": "Pasonatech",
        "date_created": today
    }
    license_dict_list = [{
      "url": "",
      "id": 1,
      "name": "Private License"
    }]
    # change this to desired object
    category_dict_list = [{
      "supercategory": "hsr",
      "id": 1,
      "name": "hsr",
      "keypoints": ["A","B","C","D","E","F","G","H","I","J","K","L"],
      "skeleton": [[1,2],[2,3],[3,4],[4,1],[1,5],[2,6],[3,7],[4,8],[5,6],[6,7],[7,8],[8,5],[5,9],[6,10],[7,11],[8,12],[9,10],[10,11],[11,12],[12,9]]
      }
      ]
    
    coco_convert = CocoNDDSConverter(data_dir=data_dir, info_dict=info_dict, license_dict_list=license_dict_list, category_dict_list=category_dict_list, save_path=save_location)

    # process data, return json and saved directory
    coco_dataset, img_dir = coco_convert.process()
    
    #  check saved annotation
    json_dict_list = json.load(open(save_location, 'r'))

    logger.yellow(json_dict_list)





