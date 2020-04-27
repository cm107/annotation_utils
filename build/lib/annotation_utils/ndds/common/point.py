from __future__ import annotations
from common_utils.check_utils import check_list_length

class Point2D: # TODO: Replace with Point2D from common_utils
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

class Point3D: # TODO: Replace with Point3D from common_utils
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