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
from common_utils.time_utils import get_present_time_Ymd
import shutil
from .coco_base import CocoInfo, CocoLicense, CocoImage, CocoAnnotation, CocoCategory, CocoDataset
import datetime
import cv2
from .common.point import Point2D, Point3D
from .common.cuboid0 import Cuboid2D, Cuboid3D
from .common.angle import Quaternion
from .common.camera import CameraParam

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
    def from_dict(self, object_dict: dict) -> NDDS_Annotation_Object:
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


class CocoNDDSConverter:
    def __init__(self, data_dir: str, info_dict: dict, license_dict_list: List[dict], category_dict_list:List[dict], save_path: str):
        self.data_dir = data_dir
        self.save_path = os.path.abspath(save_path)
        
        self.coco_info = CocoInfo().from_dict(coco_dict=info_dict)
        self.coco_license_list = [CocoLicense().from_dict(coco_dict=license_dict) for license_dict in license_dict_list]
        self.coco_image_list = []
        self.coco_annotation_list= []
        self.coco_category_list = [CocoCategory().from_dict(coco_dict=category_dict) for category_dict in category_dict_list]
        self.camera_settings_json = self.get_camera_settings()

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
    

    def get_instance_segmentation(self, img, lower_thres: list = (0,0,0), upper_thresh: list = (255,255,255)):

        color_mask = cv2.inRange(src=img, lowerb=lower_thres, upperb=upper_thresh)
        color_contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        seg = Segmentation.from_contour(contour_list=color_contours)

        return seg

    def get_keypoints(self, ann_object_list: list, object_name_single, object_index):
    
        keypoint_dict = {}
        keypoint_3d_dict = {}
        for ann_object in ann_object_list:
            ndds_keypoint_object = NDDS_Annotation_Object.from_dict(ann_object)
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

        if len(keypoints) == 0:
            print("keypoints is nil")
        
        return keypoints, keypoints_3d
    
    def segmentation_id_to_color(self, ndds_ann_object: NDDS_Annotation_Object ):

        RGBint = ndds_ann_object.instance_id
        pixel_b =  RGBint & 255
        pixel_g = (RGBint >> 8) & 255
        pixel_r =   (RGBint >> 16) & 255
        color_instance_rgb = [pixel_b,pixel_g,pixel_r]

        return color_instance_rgb
    
    def camera_params_from_dict(self, ann_dict: dict):
        camera_dict = ann_dict['camera_data']
        camera_instrinsic_settings = self.camera_settings_json[0]["intrinsic_settings"]
        camera_params = CameraParam(f=[camera_instrinsic_settings["fx"], camera_instrinsic_settings["fy"]], c=[camera_instrinsic_settings["cx"], camera_instrinsic_settings["cy"]], T=camera_dict['location_worldframe'], resx= camera_instrinsic_settings["resX"], resy= camera_instrinsic_settings["resY"])

        return camera_params

    def save_images(self):

        save_path_dir = os.path.dirname(os.path.abspath(self.save_path))
        Path(save_path_dir).mkdir(parents=True, exist_ok=True)
        # move images
        for images in self.coco_image_list:
            shutil.copy(images.coco_url, f'{save_path_dir}/'+images.file_name)
            images.coco_url = save_path_dir+images.file_name


    def process(self, save_image_to_dir:bool= True):

        object_name = np.array([item.name for item in self.coco_category_list])
        json_path_list = self.get_all_object_json_files()

        for i, json_path in enumerate(json_path_list):
            ann_dict = json.load(open(json_path, 'r'))
            camera_params = self.camera_params_from_dict(ann_dict=ann_dict)
            ann_object_list = ann_dict['objects']

            image_location = os.path.abspath(json_path[:-5]+'.png')
            coco_image = CocoImage(
                license=1, file_name=os.path.basename(image_location), coco_url=image_location,
                height= camera_params.resx, width= camera_params.resy, date_captured=get_present_time_Ymd(),
                flickr_url=None, id=i
            )
            self.coco_image_list.append(coco_image)

            ndds_ann_object_list = []

            for ann_object in ann_object_list:
                ndds_ann_object = NDDS_Annotation_Object.from_dict(ann_object)
                if any(ele in ndds_ann_object.class_name for ele in object_name):
                    mask = np.array([ele in ndds_ann_object.class_name for ele in object_name])
                    object_name_single = object_name[mask][0]

                    category_id = [category.id for category in self.coco_category_list if category.name == object_name_single][0]
                    object_index = ndds_ann_object.class_name.replace(object_name_single, '')

                    color_instance_rgb = self.segmentation_id_to_color(ndds_ann_object)

                    # get segmentation and bbox
                    instance_img = cv2.imread(coco_image.coco_url.replace('.png', '.is.png'))
                        
                    seg = self.get_instance_segmentation(img= instance_img, lower_thres=(int(color_instance_rgb[0]-1), int(color_instance_rgb[1]-1), int(color_instance_rgb[2]-1)), upper_thresh=(int(color_instance_rgb[0]+1), int(color_instance_rgb[1]+1), int(color_instance_rgb[2]+1)))
                        
                    if len(seg) == 0:
                        print("image segmentation not found, please check color code")
                        continue
                            
                    outer_bbox = seg.to_bbox()

                    if outer_bbox.area() < 10:
                        print("object too small to be considered")
                        continue
                    
                    keypoints, keypoints_3d = self.get_keypoints(ann_object_list=ann_object_list, object_name_single=object_name_single, object_index=object_index)

                    coco_annotation = CocoAnnotation(bbox=outer_bbox.to_list(output_format='pminsize'), 
                                                    image_id= i, 
                                                    category_id= category_id, 
                                                    is_crowd=0, 
                                                    id=len(self.coco_annotation_list)+1, 
                                                    keypoints=keypoints, 
                                                    keypoints_3d = keypoints_3d, 
                                                    segmentation=seg.to_list(),
                                                    area=outer_bbox.area(),
                                                    orientation_xyzw=ndds_ann_object.quaternion_xyzw.to_dict(), 
                                                    camera_params=camera_params.to_dict_fct(), 
                                                    num_keypoints=int(len(keypoints)/3))
                    self.coco_annotation_list.append(coco_annotation)
        if save_image_to_dir:
            self.save_images()
        
        return CocoDataset(info=self.coco_info, licenses=self.coco_license_list, images=self.coco_image_list, annotations=self.coco_annotation_list, categories=self.coco_category_list), self.save_path
