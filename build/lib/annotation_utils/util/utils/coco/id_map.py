from __future__ import annotations
import operator
from logger import logger
from common_utils.check_utils import check_type

class ID_Map:
    def __init__(self, unique_key: str, old_id: int, new_id: int):
        self.unique_key = unique_key
        self.old_id = old_id
        self.new_id = new_id

class ID_Mapper:
    def __init__(self):
        self.id_maps = []

    def __len__(self) -> int:
        return len(self.id_maps)

    def __getitem__(self, idx: int) -> ID_Mapper:
        if len(self.id_maps) == 0:
            logger.error(f"ID_Mapper is empty.")
            raise IndexError
        elif idx < 0 or idx >= len(self.id_maps):
            logger.error(f"Index out of range: {idx}")
            raise IndexError
        else:
            return self.id_maps[idx]

    def __setitem__(self, idx: int, value: ID_Mapper):
        check_type(value, valid_type_list=[ID_Mapper])
        self.id_maps[idx] = value

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self) -> ID_Mapper:
        if self.n < len(self.id_maps):
            result = self.id_maps[self.n]
            self.n += 1
            return result
        else:
            raise StopIteration

    def sort(self):
        self.id_maps.sort(key=operator.attrgetter('old_id'), reverse=False)

    def add_id_map(self, id_map: ID_Map):
        self.id_maps.append(id_map)

    def add(self, unique_key: str, old_id: int, new_id: int):
        new_id_map = ID_Map(
            unique_key=unique_key,
            old_id=old_id,
            new_id=new_id
        )
        self.add_id_map(new_id_map)
    
    def get_new_id(self, unique_key: str, old_id: int) -> (bool, int):
        found = False
        new_id = None
        for id_map in self.id_maps:
            if id_map.unique_key == unique_key and id_map.old_id == old_id:
                found = True
                new_id = id_map.new_id
        return found, new_id

class COCO_Mapper_Handler:
    def __init__(self):
        self.license_mapper = ID_Mapper()
        self.image_mapper = ID_Mapper()
        self.annotation_mapper = ID_Mapper()
        self.category_mapper = ID_Mapper()