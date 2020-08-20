from __future__ import annotations
from abc import ABCMeta, abstractmethod
from typing import TypeVar, Type, Generic, List
import json
import operator
import random

from logger import logger
from common_utils.check_utils import check_required_keys, check_type_from_list, \
    check_type
from common_utils.file_utils import file_exists

T = TypeVar('T')
H = TypeVar('H')

class BaseStructObject(Generic[T]):
    def __init__(self):
        pass

    @abstractmethod
    def __str__(self) -> str:
        ''' To override '''
        raise NotImplementedError

    def __repr__(self):
        return self.__str__()

    def __key(self) -> tuple:
        return tuple([self.__class__] + list(self.__dict__.values()))

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other) -> bool:
        if isinstance(other, self.__class__):
            return self.__key() == other.__key()
        return NotImplemented

    @classmethod
    def buffer(cls: T, obj) -> T:
        return obj

    def copy(self: T) -> T:
        """Note: All class variables must be part of the constructor."""
        return type(self)(*self.__dict__.values())

    def to_dict(self: T) -> dict:
        """Note: All class variables will be put into the dict."""
        return self.__dict__

    def save_to_path(self: T, save_path: str, overwrite: bool=False):
        if file_exists(save_path) and not overwrite:
            logger.error(f'File already exists at save_path: {save_path}')
            raise Exception
        json_dict = self.to_dict()
        json.dump(json_dict, open(save_path, 'w'), indent=2, ensure_ascii=False)

class BaseStructHandler(Generic[H, T]):
    def __init__(self: H, obj_type: type, obj_list: List[T]=None):
        check_type(obj_type, valid_type_list=[type])
        self.obj_type = obj_type
        if obj_list is not None:
            check_type_from_list(obj_list, valid_type_list=[obj_type])
        self.obj_list = obj_list if obj_list is not None else []

    def __str__(self):
        print_str = ""
        for obj in self.obj_list:
            print_str += f"{obj}\n"

        return print_str

    def __repr__(self):
        return self.__str__()

    def __len__(self) -> int:
        return len(self.obj_list)
    
    def __eq__(self, other) -> bool:
        if isinstance(other, type(self)):
            if len(self) != len(other):
                return False
            else:
                for obj0, obj1 in zip(self, other):
                    if obj0 != obj1:
                        return False
                return True
        else:
            return NotImplemented

    def __getitem__(self, idx: int) -> T:
        if type(idx) is int:
            if len(self.obj_list) == 0:
                logger.error(f"{type(self).__name__} is empty.")
                raise IndexError
            elif idx < 0 or idx >= len(self.obj_list):
                logger.error(f"Index out of range: {idx}")
                raise IndexError
            else:
                return self.obj_list[idx]
        elif type(idx) is slice:
            return self.obj_list[idx.start:idx.stop:idx.step]
        else:
            logger.error(f'Expected int or slice. Got type(idx)={type(idx)}')
            raise TypeError

    def __setitem__(self, idx: int, value: T):
        check_type(value, valid_type_list=[self.obj_type])
        if type(idx) is int:
            self.obj_list[idx] = value
        elif type(idx) is slice:
            self.obj_list[idx.start:idx.stop:idx.step] = value
        else:
            logger.error(f'Expected int or slice. Got type(idx)={type(idx)}')
            raise TypeError

    def __delitem__(self, idx):
        if type(idx) is int:
            if len(self.obj_list) == 0:
                logger.error(f"{type(self).__name__} is empty.")
                raise IndexError
            elif idx < 0 or idx >= len(self.obj_list):
                logger.error(f"Index out of range: {idx}")
                raise IndexError
            else:
                del self.obj_list[idx]
        elif type(idx) is slice:
            del self.obj_list[idx.start:idx.stop:idx.step]
        else:
            logger.error(f'Expected int or slice. Got type(idx)={type(idx)}')
            raise TypeError

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self) -> T:
        if self.n < len(self.obj_list):
            result = self.obj_list[self.n]
            self.n += 1
            return result
        else:
            raise StopIteration

    def copy(self) -> H:
        return type(self)(self.obj_list.copy())

    def append(self, item: T):
        check_type(item, valid_type_list=[self.obj_type])
        self.obj_list.append(item)

    def sort(self, attr_name: str, reverse: bool=False):
        if len(self) > 0:
            attr_list = list(self.obj_list[0].__dict__.keys())    
            if attr_name not in attr_list:
                logger.error(f"{self.obj_type.__name__} class has not attribute: '{attr_name}'")
                logger.error(f'Possible attribute names:')
                for name in attr_list:
                    logger.error(f'\t{name}')
                raise Exception

            self.obj_list.sort(key=operator.attrgetter(attr_name), reverse=reverse)
        else:
            logger.error(f"Cannot sort. {type(self).__name__} is empty.")
            raise Exception

    def shuffle(self):
        random.shuffle(self.obj_list)

    def get_obj_from_id(self, id: int) -> T: # Need to move this to a different base class
        id_list = []
        for obj in self:
            if id == obj.id:
                return obj
            else:
                id_list.append(obj.id)
        id_list.sort()
        logger.error(f"Couldn't find {self.obj_type.__name__} with id={id}")
        logger.error(f"Possible ids: {id_list}")
        raise Exception

    def to_dict_list(self) -> List[dict]:
        return [item.to_dict() for item in self]

    def save_to_path(self, save_path: str, overwrite: bool=False):
        if file_exists(save_path) and not overwrite:
            logger.error(f'File already exists at save_path: {save_path}')
            raise Exception
        json_data = self.to_dict_list()
        json.dump(json_data, open(save_path, 'w'), indent=2, ensure_ascii=False)