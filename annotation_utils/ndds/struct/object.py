from __future__ import annotations
import numpy as np
from common_utils.common_types.bbox import BBox

# from ..common.point import Point2D, Point3D
from common_utils.common_types.point import Point2D, Point3D
from ..common.cuboid import Cuboid2D, Cuboid3D
from ..common.angle import Quaternion
from ...base.structs import BaseStructObject

class NDDS_Annotation_Object(BaseStructObject['NDDS_Annotation_Object']):
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

    def to_dict(self) -> dict:
        # TODO: Finish implementing
        raise NotImplementedError
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
                'top_left': [self.bounding_box.xmin, self.bounding_box.ymin],
                'bottom_right': [self.bounding_box.xmax, self.bounding_box.ymax]
            },
            'cuboid': self.cuboid.to_list()
        }

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