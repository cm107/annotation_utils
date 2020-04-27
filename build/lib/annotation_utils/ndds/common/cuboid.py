from __future__ import annotations
from typing import List
import numpy as np
from shapely.geometry import Point as ShapelyPoint

from common_utils.check_utils import check_list_length, check_type_from_list
# from .point import Point2D, Point3D
from common_utils.common_types.point import Point2D, Point3D, Point2D_List, Point3D_List
# TODO: Make Point3D_List in common_utils

class Cuboid2D(Point2D_List): # Move to common_utils
    def __init__(self, point_list: List[Point2D]):
        check_list_length(point_list, correct_length=8)
        check_type_from_list(point_list, valid_type_list=[Point2D])
        super().__init__(point_list=point_list)

    def __str__(self):
        return f"Cuboid2D({self.point_list})"

    @classmethod
    def from_numpy(cls, arr: np.ndarray, demarcation: bool=True) -> Cuboid2D:
        return Cuboid2D(point_list=super().from_numpy(arr=arr, demarcation=demarcation).point_list)

    @classmethod
    def from_list(cls, value_list: list, demarcation: bool=True) -> Cuboid2D:
        return Cuboid2D(point_list=super().from_list(value_list=value_list, demarcation=demarcation).point_list)

    @classmethod
    def from_shapely(cls, shapely_point_list: List[ShapelyPoint]) -> Cuboid2D:
        return Cuboid2D(point_list=super().from_shapely(shapely_point_list=shapely_point_list).point_list)

class Cuboid3D(Point3D_List): # Move to common_utils
    def __init__(self, point_list: List[Point3D]):
        check_list_length(point_list, correct_length=8)
        check_type_from_list(point_list, valid_type_list=[Point3D])
        super().__init__(point_list=point_list)

    def __str__(self):
        return f"Cuboid3D({self.point_list})"

    @classmethod
    def from_numpy(cls, arr: np.ndarray, demarcation: bool=True) -> Cuboid3D:
        return Cuboid3D(point_list=super().from_numpy(arr=arr, demarcation=demarcation).point_list)

    @classmethod
    def from_list(cls, value_list: list, demarcation: bool=True) -> Cuboid3D:
        return Cuboid3D(point_list=super().from_list(value_list=value_list, demarcation=demarcation).point_list)