from __future__ import annotations
import numpy as np
from logger import logger
from common_utils.check_utils import check_required_keys, check_type_from_list, check_type
from common_utils.base.basic import BasicObject

class Transforms:
    @classmethod
    def pad_to_4d(self, points: np.ndarray) -> np.ndarray:
        return np.c_[points, np.ones(points.shape[0])]

class Camera(BasicObject['Camera']):
    def __init__(self, f: list, c: list, T: list):
        super().__init__()
        self.f = f
        self.c = c
        self.T = T

    def __str__(self) -> str:
        return f"Camera(f={self.f}, c={self.c}, T={self.T})"

    def __repr__(self) -> str:
        return self.__str__()

    def to_dict(self) -> dict:
        return {
            'f': self.f,
            'c': self.c,
            'T': self.T
        }

    @classmethod
    def from_dict(cls, intrinsic_param_dict: dict) -> Camera:
        check_type(intrinsic_param_dict, valid_type_list=[dict])
        if len(intrinsic_param_dict) > 0:
            check_required_keys(intrinsic_param_dict, required_keys=['f', 'c', 'T'])
            check_type_from_list(list(intrinsic_param_dict.values()), valid_type_list=[list])
            return Camera(
                f=intrinsic_param_dict['f'],
                c=intrinsic_param_dict['c'],
                T=intrinsic_param_dict['T']
            )
        else:
            return None

    def project_3d_to_2d(self, kpts_3d: np.ndarray) -> np.ndarray:
        """
        Note:
        kpts_3d.shape = (N, 3)
        extended_kpts_3d.shape = (N, 4)
        extended_kpts_3d.T.shape = (4, N)
        projection_mat.shape = (3, 4)
        result = projection_mat.dot(extended_kpts_3d.T)
        result.shape = (3, 4) x (4, N) -> (3, N)
        result.T.shape = (N, 3)
        """

        # camera intrinsic
        fx, fy = self.f
        cx, cy = self.c
        if len(self.T) == 2:
            Tx, Ty = self.T
            Tz = 0
        elif len(self.T) == 3:
            Tx, Ty, Tz = self.T
        else:
            logger.error(f"Invalid dimensions: len(self.T) == {len(self.T)} != 2 or 3")
            raise Exception

        # extended_kpts_3d = np.c_[kpts_3d, np.ones(kpts_3d.shape[0])]
        extended_kpts_3d = Transforms.pad_to_4d(points=kpts_3d)
        projection_mat = np.array(
            [
                [fx, 0, cx, Tx],
                [0, fy, cy, Ty],
                [0, 0, 1, Tz]
            ], dtype=np.float32
        )
        result = projection_mat.dot(extended_kpts_3d.T)
        result = np.where(result[2] != 0, result[:2] / result[2], 0)
        return result.T