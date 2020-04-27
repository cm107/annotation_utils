from __future__ import annotations
from typing import List
from common_utils.check_utils import check_list_length

class Quaternion: # TODO: Replace with Quaternion from common_utils
    def __init__(self, x: float, y: float, z: float, w: float):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    def to_list(self) -> List[float]:
        return [self.x, self.y, self.z, self.w]

    @classmethod
    def from_list(self, coords: List[float]) -> Quaternion:
        check_list_length(coords, correct_length=4)
        return Quaternion(x=coords[0], y=coords[1], z=coords[2], w=coords[3])

    def __str__(self):
        return f"Quaternion({self.x},{self.y},{self.z},{self.w})"

    def __repr__(self):
        return self.__str__()
    
    def to_dict(self):
        return self.__dict__