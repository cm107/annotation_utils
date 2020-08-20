from __future__ import annotations
from typing import List

from logger import logger
from common_utils.base.basic import BasicLoadableHandler, BasicHandler
from .objects import NDDS_Annotation_Object

class NDDS_Annotation_Object_Handler(
    BasicLoadableHandler['NDDS_Annotation_Object_Handler', 'NDDS_Annotation_Object'],
    BasicHandler['NDDS_Annotation_Object_Handler', 'NDDS_Annotation_Object']
):
    def __init__(self, ndds_obj_list: List[NDDS_Annotation_Object]=None):
        super().__init__(obj_type=NDDS_Annotation_Object, obj_list=ndds_obj_list)
        self.objects = self.obj_list

    @classmethod
    def from_dict_list(cls, dict_list: List[dict]) -> NDDS_Annotation_Object_Handler:
        return NDDS_Annotation_Object_Handler(
            ndds_obj_list=[NDDS_Annotation_Object.from_dict(obj_dict) for obj_dict in dict_list]
        )

    def delete(self, idx_list: List[int], verbose: bool=False, verbose_ref=None):
        if len(idx_list) > 0 and verbose and verbose_ref:
            logger.info(f'verbose_ref: {verbose_ref}')
        for i in idx_list:
            if verbose:
                if verbose_ref:
                    logger.info(f'\tDeleted duplicate of {self.objects[i].class_name}')
                else:
                    logger.info(f'Deleted duplicate of {self.objects[i].class_name}')
            del self[i]

    def find_duplicates(self) -> List[int]:
        duplicate_idx_list = []
        for i in range(len(self)):
            for j in range(i+1, len(self)):
                if self[i].class_name == self[j].class_name:
                    if j not in duplicate_idx_list:
                        duplicate_idx_list.append(j)
        return duplicate_idx_list

    def delete_duplicates(self, verbose: bool=False, verbose_ref=None):
        self.delete(
            idx_list=self.find_duplicates(),
            verbose=verbose, verbose_ref=verbose_ref
        )