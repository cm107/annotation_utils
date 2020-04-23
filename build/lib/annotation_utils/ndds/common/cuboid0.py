from __future__ import annotations
from typing import List
from common_utils.check_utils import check_list_length, check_type_from_list
from .point import Point2D, Point3D

class Cuboid2D: # Move to common_utils
    def __init__(self, point_list: List[Point2D]):
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

class Cuboid3D: # Move to common_utils
    def __init__(self, point_list: List[Point3D]):
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