from __future__ import annotations
from typing import List
import numpy as np
import cv2
from logger import logger
from common_utils.common_types.bbox import BBox
from common_utils.common_types.segmentation import Segmentation
from common_utils.check_utils import check_required_keys, check_list_length

from common_utils.common_types.point import Point2D, Point3D
from ..common.cuboid import Cuboid2D, Cuboid3D
from ..common.angle import Quaternion
from common_utils.base.basic import BasicLoadableObject

class NDDS_Annotation_Object(BasicLoadableObject['NDDS_Annotation_Object']):
    def __init__(
        self,
        class_name: str, instance_id: int, visibility: float, location: Point3D, quaternion_xyzw: Quaternion,
        pose_transform: np.ndarray, cuboid_centroid: Point3D, projected_cuboid_centroid: Point2D,
        bounding_box: BBox, cuboid: Cuboid3D, projected_cuboid: Cuboid2D
    ):
        super().__init__()
        self.class_name = class_name
        self.instance_id = instance_id
        if visibility < 0 or visibility > 1:
            logger.error(f'visibility must be between 0 and 1')
            logger.error(f'visibility: {visibility}')
            raise Exception
        self.visibility = visibility
        self.location = location
        self.quaternion_xyzw = quaternion_xyzw
        self.pose_transform = pose_transform
        self.cuboid_centroid = cuboid_centroid
        self.projected_cuboid_centroid = projected_cuboid_centroid
        self.bounding_box = bounding_box
        self.cuboid = cuboid
        self.projected_cuboid = projected_cuboid

    def to_dict(self) -> dict:
        return {
            'class': self.class_name,
            'instance_id': self.instance_id,
            'visibility': self.visibility,
            'location': self.location.to_list(),
            'quaternion_xyzw': self.quaternion_xyzw.to_list(),
            'pose_transform': self.pose_transform.tolist(),
            'cuboid_centroid': self.cuboid_centroid.to_list(),
            'projected_cuboid_centroid': self.projected_cuboid_centroid.to_list(),
            'bounding_box': {
                'top_left': [self.bounding_box.xmin, self.bounding_box.ymin][::-1],
                'bottom_right': [self.bounding_box.xmax, self.bounding_box.ymax][::-1]
            },
            'cuboid': self.cuboid.to_list(demarcation=True),
            'projected_cuboid': self.projected_cuboid.to_list(demarcation=True)
        }

    @classmethod
    def from_dict(self, object_dict: dict) -> NDDS_Annotation_Object:
        check_required_keys(
            object_dict,
            required_keys=[
                'class', 'instance_id', 'visibility',
                'location', 'quaternion_xyzw', 'pose_transform',
                'cuboid_centroid', 'projected_cuboid_centroid', 'bounding_box',
                'cuboid', 'projected_cuboid'
            ]
        )
        check_required_keys(
            object_dict['bounding_box'],
            required_keys=['top_left', 'bottom_right']
        )
        return NDDS_Annotation_Object(
            class_name=object_dict['class'],
            instance_id=object_dict['instance_id'],
            visibility=object_dict['visibility'],
            location=Point3D.from_list(object_dict['location']),
            quaternion_xyzw=Quaternion.from_list(object_dict['quaternion_xyzw']),
            pose_transform=np.array(object_dict['pose_transform']),
            cuboid_centroid=Point3D.from_list(object_dict['cuboid_centroid']),
            projected_cuboid_centroid=Point2D.from_list(object_dict['projected_cuboid_centroid']),
            bounding_box=BBox.from_list(object_dict['bounding_box']['top_left'][::-1]+object_dict['bounding_box']['bottom_right'][::-1], input_format='pminpmax'),
            cuboid=Cuboid3D.from_list(object_dict['cuboid'], demarcation=True),
            projected_cuboid=Cuboid2D.from_list(object_dict['projected_cuboid'], demarcation=True)
        )

    def parse_obj_info(self, naming_rule: str='type_object_instance_contained', delimiter: str='_') -> (str, str, str):
        if naming_rule == 'type_object_instance_contained':
            class_name_parts = self.class_name.split(delimiter)
            if len(class_name_parts) == 4:
                obj_type, obj_name, instance_name, contained_name = class_name_parts
            elif len(class_name_parts) == 3:
                obj_type, obj_name, instance_name = class_name_parts
                contained_name = None
            elif len(class_name_parts) == 2:
                obj_type, obj_name = class_name_parts
                instance_name, contained_name = None, None
            elif len(class_name_parts) == 1:
                obj_name = class_name_parts
                obj_type = 'seg'
                instance_name, contained_name = None, None
            else:
                logger.error(f"Too many delimiters ('{delimiter}') found in class_name: {self.class_name}")
                logger.error(f'Parsed {len(class_name_parts)} parts. Expected <= 4.')
                logger.error(f'self.instance_id: {self.instance_id}')
                raise Exception
            return obj_type, obj_name, instance_name, contained_name
        else:
            logger.error(f'Invalid naming rule: {naming_rule}')
            raise NotImplementedError
    
    def get_color_from_id(self) -> List[int]:
        RGBint = self.instance_id
        pixel_b =  RGBint & 255
        pixel_g = (RGBint >> 8) & 255
        pixel_r =   (RGBint >> 16) & 255
        color_instance_bgr = [pixel_b,pixel_g,pixel_r]
        return color_instance_bgr

    def get_instance_segmentation(self, img: np.ndarray, target_bgr: List[int]=None, interval: int=1, exclude_invalid_polygons: bool=True):
        target_bgr = target_bgr if target_bgr is not None else self.get_color_from_id()
        check_list_length(target_bgr, correct_length=3)
        lower_bgr = [val - interval if val - interval >= 0 else 0 for val in target_bgr]
        upper_bgr = [val + interval if val + interval <= 255 else 255 for val in target_bgr]
        color_mask = cv2.inRange(src=img, lowerb=tuple(lower_bgr), upperb=tuple(upper_bgr))
        color_contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        seg = Segmentation.from_contour(contour_list=color_contours, exclude_invalid_polygons=exclude_invalid_polygons)
        return seg

    def is_in_frame(self, frame_shape: List[int]) -> bool:
        frame_h, frame_w = frame_shape[:2]
        frame_bbox = BBox(xmin=0, ymin=0, xmax=frame_w, ymax=frame_h)
        return self.bounding_box.within(frame_bbox)
    
class CameraData(BasicLoadableObject['CameraData']):
    def __init__(self, location_worldframe: Point3D, quaternion_xyzw_worldframe: Quaternion):
        super().__init__()
        self.location_worldframe = location_worldframe
        self.quaternion_xyzw_worldframe = quaternion_xyzw_worldframe
    
    def to_dict(self) -> dict:
        return {
            'location_worldframe': self.location_worldframe.to_list(),
            'quaternion_xyzw_worldframe': self.quaternion_xyzw_worldframe.to_list()
        }
    
    @classmethod
    def from_dict(cls, item_dict: dict) -> CameraData:
        check_required_keys(
            item_dict,
            required_keys=['location_worldframe', 'quaternion_xyzw_worldframe']
        )
        return CameraData(
            location_worldframe=Point3D.from_list(coords=item_dict['location_worldframe']),
            quaternion_xyzw_worldframe=Quaternion.from_list(coords=item_dict['quaternion_xyzw_worldframe'])
        )
    
