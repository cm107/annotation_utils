from __future__ import annotations
import json
from typing import List
# from tqdm import tqdm

from logger import logger
from common_utils.check_utils import check_file_exists
from ...base.basic import BasicLoadableHandler, BasicHandler
from .objects import NDDS_Annotation_Object
# from .instance import LabeledObjectHandler, LabeledObject, ObjectInstance

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

    # def to_labeled_obj_handler(self, naming_rule: str='type_object_instance_contained', delimiter: str='_', show_pbar: bool=True) -> LabeledObjectHandler:
    #     # TODO: Moved to NDDS_Frame_Handler
    #     def process_non_contained(handler: LabeledObjectHandler, ann_obj: NDDS_Annotation_Object):
    #         if obj_name not in handler.get_obj_names(): # New Object
    #             labeled_obj = LabeledObject(obj_name=obj_name)
    #             labeled_obj.instances.append(
    #                 ObjectInstance(
    #                     instance_type=obj_type,
    #                     ndds_ann_obj=ann_obj,
    #                     instance_name=instance_name
    #                 )
    #             )
    #         else: # Object Name already in handler
    #             handler[handler.index(obj_name=obj_name)].instances.append(
    #                 ObjectInstance(
    #                     instance_type=obj_type,
    #                     ndds_ann_obj=ann_obj,
    #                     instance_name=instance_name
    #                 )
    #             )

    #     def process_contained(handler: LabeledObjectHandler, ann_obj: NDDS_Annotation_Object):
    #         if obj_name not in handler.get_obj_names():
    #             logger.error(
    #                 f"Contained object (contained_name={contained_name}) " + \
    #                 f"cannot be defined before container object (obj_name={obj_name}) is defined."
    #             )
    #             raise Exception
    #         obj_idx = handler.index(obj_name=obj_name)
    #         if instance_name not in handler[obj_idx].instances.get_instance_names():
    #             logger.error(
    #                 f"Contained object (contained_name={contained_name}) " + \
    #                 f"cannot be defined before container object (obj_name={obj_name}) " + \
    #                 f" and container instance (instance_name={instance_name}) are defined."
    #             )
    #         instance_idx = handler[obj_idx].instances.index(instance_name=instance_name)
    #         handler[obj_idx].instances[instance_idx].append_contained(
    #             ObjectInstance(
    #                 instance_type=obj_type,
    #                 ndds_ann_obj=ann_obj,
    #                 instance_name=instance_name
    #             )
    #         )

    #     handler = LabeledObjectHandler()

    #     if naming_rule == 'type_object_instance_contained':
    #         # Add Non-contained Objects First
    #         if show_pbar:
    #             non_contained_pbar = tqdm(total=len(self), unit='ann_obj', leave=False)
    #             non_contained_pbar.set_description('Loading Containers')
    #         for ann_obj in self:
    #             obj_type, obj_name, instance_name, contained_name = ann_obj.parse_obj_info(naming_rule=naming_rule, delimiter=delimiter)
    #             if contained_name is None: # Non-contained Object
    #                 process_non_contained(handler=handler, ann_obj=ann_obj)
    #             if show_pbar:
    #                 non_contained_pbar.update()
            
    #         # Add Contained Objects Second
    #         if show_pbar:
    #             contained_pbar = tqdm(total=len(self), unit='ann_obj', leave=False)
    #             contained_pbar.set_description('Loading Containables')
    #         for ann_obj in self:
    #             obj_type, obj_name, instance_name, contained_name = ann_obj.parse_obj_info(naming_rule=naming_rule, delimiter=delimiter)
    #             if contained_name is not None: # Contained Object
    #                 process_non_contained(handler=handler, ann_obj=ann_obj)
    #             if show_pbar:
    #                 contained_pbar.update()
    #     return handler