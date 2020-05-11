from __future__ import annotations
from typing import List

from logger import logger
from common_utils.check_utils import check_value, check_type

from ...base.basic import BasicObject, BasicHandler
from .objects import NDDS_Annotation_Object

class ObjectInstance(BasicObject['ObjectInstance']):
    def __init__(
        self, instance_type: str, ndds_ann_obj: NDDS_Annotation_Object, instance_name: str=None, contained_instance_list: List[ObjectInstance]=None
    ):
        super().__init__()
        
        # Required
        if instance_type.startswith('bbox') and len(instance_type.replace('bbox', '')) > 0:
            if instance_type.replace('bbox', '').isdigit():
                self.part_num = int(instance_type.replace('bbox', ''))
                self.instance_type = 'bbox'
            else:
                logger.error(f'Part number must be a string that can be converted to an integer.')
                logger.error(f'Valid example: bbox0')
                logger.error(f'Invalid example: bboxzero')
                raise Exception
        elif instance_type.startswith('seg') and len(instance_type.replace('seg', '')) > 0:
            if instance_type.replace('seg', '').isdigit():
                self.part_num = int(instance_type.replace('seg', ''))
                self.instance_type = 'seg'
            else:
                logger.error(f'Part number must be a string that can be converted to an integer.')
                logger.error(f'Valid example: seg0')
                logger.error(f'Invalid example: segzero')
                raise Exception
        else:
            check_value(instance_type, valid_value_list=['bbox', 'seg', 'kpt'])
            self.instance_type = instance_type
            self.part_num = None
        self.ndds_ann_obj = ndds_ann_obj

        # Optional
        self.instance_name = instance_name
        self.contained_instance_list = contained_instance_list if contained_instance_list is not None else []
    
    def __str__(self) -> str:
        constructor_dict = self.to_constructor_dict()
        param_str = ''
        for key, val in constructor_dict.items():
            if param_str == '':
                param_str += f'{key}={val}'
            else:
                param_str += f', {key}={val}'
        return f'ObjectInstance({param_str})'

    def append_contained(self, new_contained_instance: ObjectInstance):
        check_value(self.instance_type, valid_value_list=['bbox', 'seg'])
        check_type(new_contained_instance, valid_type_list=[ObjectInstance])
        check_value(new_contained_instance.instance_type, valid_value_list=['bbox', 'seg', 'kpt'])

        # Check Instance Id
        if new_contained_instance.ndds_ann_obj.instance_id == self.ndds_ann_obj.instance_id:
            logger.error(f'new_contained_instance.ndds_ann_obj.instance_id == self.ndds_ann_obj.instance_id')
            logger.error(f'new_contained_instance: {new_contained_instance}')
            logger.error(f'self: {self}')
            raise Exception
        if new_contained_instance.ndds_ann_obj.instance_id in [
            contained_instance.ndds_ann_obj.instance_id for contained_instance in self.contained_instance_list
        ]:
            logger.error(
                f'new_contained_instance.ndds_ann_obj.instance_id in ' + \
                f'[contained_instance.ndds_ann_obj.instance_id for contained_instance in self.contained_instance_list] == True'
            )
            logger.error(f'new_contained_instance: {new_contained_instance}')
            logger.error(f'self: {self}')
            raise Exception
        # Check (instance_type, instance_name) pair
        if (new_contained_instance.instance_type, new_contained_instance.instance_name) in [
            (contained_instance.instance_type, contained_instance.instance_name) for contained_instance in self.contained_instance_list
        ]:
            logger.error(
                f'(new_contained_instance.instance_type, new_contained_instance.instance_name)=' + \
                f'{(new_contained_instance.instance_type, new_contained_instance.instance_name)} ' + \
                f'pair already exists in self.contained_instance_list'
            )
            logger.error(f'Existing pairs:')
            found_inst = None
            for inst in self.contained_instance_list:
                logger.error(f'\t(inst.instance_type, inst.instance_name)={(inst.instance_type, inst.instance_name)}')
                if (inst.instance_type, inst.instance_name) == (new_contained_instance.instance_type, new_contained_instance.instance_name):
                    found_inst = inst.copy()
            found_inst = ObjectInstance.buffer(found_inst)
            logger.error(f'\n')
            if new_contained_instance.ndds_ann_obj != found_inst.ndds_ann_obj:
                # logger.error(f'new_contained_instance:\n{new_contained_instance}')
                # logger.error(f'found_inst:\n{found_inst}')
                for key in NDDS_Annotation_Object.get_constructor_params():
                    if new_contained_instance.ndds_ann_obj.__dict__[key] != found_inst.ndds_ann_obj.__dict__[key]:
                        logger.error(f'Found difference in key={key}')
                        logger.error(f'\tnew_contained_instance.ndds_ann_obj.__dict__[{key}]:\n\t{new_contained_instance.ndds_ann_obj.__dict__[key]}')
                        logger.error(f'\tfound_inst.ndds_ann_obj.__dict__[{key}]:\n\t{found_inst.ndds_ann_obj.__dict__[key]}')
            else:
                logger.error(f"The two instance's ndds_ann_obj are identical.")
            raise Exception
        
        self.contained_instance_list.append(new_contained_instance)

class ObjectInstanceHandler(BasicHandler['ObjectInstanceHandler', 'ObjectInstance']):
    def __init__(self, instance_list: List[ObjectInstance]=None):
        super().__init__(obj_type=ObjectInstance, obj_list=instance_list)
        self.instance_list = self.obj_list

    def append(self, obj_instance: ObjectInstance):
        for instance in self:
            if obj_instance.ndds_ann_obj.instance_id == instance.ndds_ann_obj.instance_id:
                logger.error(f'obj_instance.ndds_ann_obj.instance_id={obj_instance.ndds_ann_obj.instance_id} already exists in {self.__class__.__name__}')
                raise Exception
            if obj_instance.instance_type == instance.instance_type and obj_instance.instance_name == instance.instance_name and obj_instance.part_num == instance.part_num:
                logger.error(
                    f'(obj_instance.instance_type, obj_instance.instance_name, obj_instance.part_num)=({obj_instance.instance_type}, {obj_instance.instance_name}, {obj_instance.part_num}) ' + \
                    f'pair already exists in {self.__class__.__name__}'
                )
                if obj_instance.instance_name is None:
                    logger.error(f"Note: You haven't specified any instance_name.")
                    logger.error(f'Hint: Unless you only plan on making one instance of an object, you need to specify an instance_name.')
                raise Exception
        super().append(item=obj_instance)
    
    def get_instance_from_id(self, instance_id: int) -> ObjectInstance:
        id_list = []
        for obj in self:
            if instance_id == obj.ndds_ann_obj.instance_id:
                return obj
            else:
                id_list.append(obj.ndds_ann_obj.instance_id)
        id_list.sort()
        logger.error(f"Couldn't find {self.obj_type.__name__} with self.ndds_ann_obj.instance_id={instance_id}")
        logger.error(f"Possible ids: {id_list}")
        raise Exception

    def get_instance_names(self) -> List[str]:
        return [instance.instance_name for instance in self]

    def index(self, **kwargs) -> int:
        if len(kwargs) != 1:
            logger.error(f'Expected exactly one input parameter.')
            logger.error(f'kwargs: {kwargs}')
            raise Exception
        if 'instance_id' in kwargs:
            instance_id = kwargs['instance_id']
            for i in range(len(self)):
                if self[i].ndds_ann_obj.instance_id == instance_id:
                    return i
            return None
        elif 'instance_name' in kwargs:
            instance_name = kwargs['instance_name']
            for i in range(len(self)):
                if self[i].instance_name == instance_name:
                    return i
            return None
        else:
            raise Exception

class LabeledObject(BasicObject['LabeledObject']):
    def __init__(self, obj_name: str, instances: ObjectInstanceHandler=None):
        self.obj_name = obj_name
        self.instances = instances if instances is not None else ObjectInstanceHandler()

    def __str__(self) -> str:
        constructor_dict = self.to_constructor_dict()
        param_str = ''
        for key, val in constructor_dict.items():
            if param_str == '':
                param_str += f'{key}={val}'
            else:
                param_str += f', {key}={val}'
        return f'LabeledObject({param_str})'

class LabeledObjectHandler(BasicHandler['LabeledObjectHandler', 'LabeledObject']):
    def __init__(self, labeled_obj_list: List[LabeledObject]=None):
        super().__init__(obj_type=LabeledObject, obj_list=labeled_obj_list)
        self.labeled_obj_list = self.obj_list

    def append(self, labeled_obj: LabeledObject):
        for obj in self:
            if obj.obj_name == labeled_obj.obj_name:
                logger.error(f'labeled_obj.obj_name={labeled_obj.obj_name} already exists in {self.__class__.__name__}')
                raise Exception
        super().append(item=labeled_obj)
    
    def get_obj_from_name(self, name: str) -> LabeledObject:
        name_list = []
        for obj in self:
            if name == obj.obj_name:
                return obj
            else:
                name_list.append(obj.obj_name)
        name_list.sort()
        logger.error(f"Couldn't find {self.obj_type.__name__} with obj_name={name}")
        logger.error(f"Possible labels: {name_list}")
        raise Exception

    def get_obj_names(self) -> List[str]:
        return [labeled_obj.obj_name for labeled_obj in self]

    def index(self, **kwargs) -> int:
        if len(kwargs) != 1:
            logger.error(f'Expected exactly one input parameter.')
            logger.error(f'kwargs: {kwargs}')
            raise Exception
        if 'obj_name' in kwargs:
            obj_name = kwargs['obj_name']
            for i in range(len(self)):
                if self[i].obj_name == obj_name:
                    return i
            return None
        else:
            raise Exception