import operator

class ID_Map:
    def __init__(self, unique_key: str, old_id: int, new_id: int):
        self.unique_key = unique_key
        self.old_id = old_id
        self.new_id = new_id

class ID_Mapper:
    def __init__(self):
        self.id_maps = []

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