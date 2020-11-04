from __future__ import annotations
from typing import List
from common_utils.base.basic import BasicLoadableIdObject, BasicLoadableObject, BasicLoadableIdHandler, BasicHandler
from common_utils.common_types.point import Point2D, Point3D, Point2D_List, Point3D_List
from common_utils.common_types.angle import QuaternionList

class Linemod_Image(
    BasicLoadableIdObject['Linemod_Image'],
    BasicLoadableObject['Linemod_Image']
):
    def __init__(self, file_name: str, width: int, height: int, id: int):
        super().__init__(id=id)
        self.file_name = file_name # Note: This can also be a path.
        self.width = width
        self.height = height

class Linemod_Image_Handler(
    BasicLoadableIdHandler['Linemod_Image_Handler', 'Linemod_Image'],
    BasicHandler['Linemod_Image_Handler', 'Linemod_Image']
):
    def __init__(self, images: List[Linemod_Image]=None):
        super().__init__(obj_type=Linemod_Image, obj_list=images)
        self.images = self.obj_list
    
    @classmethod
    def from_dict_list(cls, dict_list: List[dict]) -> Linemod_Image_Handler:
        return Linemod_Image_Handler([Linemod_Image.from_dict(item_dict) for item_dict in dict_list])

class Linemod_Annotation(
    BasicLoadableIdObject['Linemod_Annotation'],
    BasicLoadableObject['Linemod_Annotation'],
):
    def __init__(
        self,
        data_root: str,
        mask_path: str,
        type: str, class_name: str,
        corner_2d: Point2D_List, corner_3d: Point3D_List,
        center_2d: Point2D, center_3d: Point3D,
        fps_2d: Point2D_List, fps_3d: Point3D_List,
        K: Point3D_List,
        pose: QuaternionList,
        image_id: int, category_id: int, id: int,
        depth_path: str=None
    ):
        super().__init__(id=id)
        self.data_root = data_root
        self.mask_path = mask_path
        self.depth_path = depth_path
        self.type = type
        self.class_name = class_name
        self.corner_2d = corner_2d
        self.corner_3d = corner_3d
        self.center_2d = center_2d
        self.center_3d = center_3d
        self.fps_2d = fps_2d
        self.fps_3d = fps_3d
        self.K = K # Is this a camera matrix? Figure this out later.
        self.pose = pose
        self.image_id = image_id
        self.category_id = category_id

    def to_dict(self) -> dict:
        result = {
            'data_root': self.data_root,
            'mask_path': self.mask_path,
            'type': self.type,
            'cls': self.class_name,
            'corner_2d': self.corner_2d.to_list(demarcation=True),
            'corner_3d': self.corner_3d.to_list(demarcation=True),
            'center_2d': self.center_2d.to_list(),
            'center_3d': self.center_3d.to_list(),
            'fps_2d': self.fps_2d.to_list(demarcation=True),
            'fps_3d': self.fps_3d.to_list(demarcation=True),
            'K': self.K.to_list(demarcation=True),
            'pose': self.pose.to_list(),
            'image_id': self.image_id,
            'category_id': self.category_id,
            'id': self.id
        }
        if self.depth_path is not None:
            result['depth_path'] = self.depth_path
        return result
    
    @classmethod
    def from_dict(cls, item_dict: dict) -> Linemod_Annotation:
        return Linemod_Annotation(
            data_root=item_dict['data_root'],
            mask_path=item_dict['mask_path'],
            depth_path=item_dict['depth_path'] if 'depth_path' in item_dict else None,
            type=item_dict['type'],
            class_name=item_dict['cls'],
            corner_2d=Point2D_List.from_list(item_dict['corner_2d'], demarcation=True),
            corner_3d=Point3D_List.from_list(item_dict['corner_3d'], demarcation=True),
            center_2d=Point2D.from_list(item_dict['center_2d']),
            center_3d=Point3D.from_list(item_dict['center_3d']),
            fps_2d=Point2D_List.from_list(item_dict['fps_2d'], demarcation=True),
            fps_3d=Point3D_List.from_list(item_dict['fps_3d'], demarcation=True),
            K=Point3D_List.from_list(item_dict['K'], demarcation=True),
            pose=QuaternionList.from_list(item_dict['pose']),
            image_id=item_dict['image_id'],
            category_id=item_dict['category_id'],
            id=item_dict['id']
        )

class Linemod_Annotation_Handler(
    BasicLoadableIdHandler['Linemod_Annotation_Handler', 'Linemod_Annotation'],
    BasicHandler['Linemod_Annotation_Handler', 'Linemod_Annotation']
):
    def __init__(self, images: List[Linemod_Annotation]=None):
        super().__init__(obj_type=Linemod_Annotation, obj_list=images)
        self.images = self.obj_list
    
    @classmethod
    def from_dict_list(cls, dict_list: List[dict]) -> Linemod_Annotation_Handler:
        return Linemod_Annotation_Handler([Linemod_Annotation.from_dict(item_dict) for item_dict in dict_list])

class Linemod_Category(
    BasicLoadableIdObject['Linemod_Category'],
    BasicLoadableObject['Linemod_Category']
):
    def __init__(self, supercategory: str, name: str, id: int):
        super().__init__(id=id)
        self.supercategory = supercategory
        self.name = name

class Linemod_Category_Handler(
    BasicLoadableIdHandler['Linemod_Category_Handler', 'Linemod_Category'],
    BasicHandler['Linemod_Category_Handler', 'Linemod_Category']
):
    def __init__(self, images: List[Linemod_Category]=None):
        super().__init__(obj_type=Linemod_Category, obj_list=images)
        self.images = self.obj_list
    
    @classmethod
    def from_dict_list(cls, dict_list: List[dict]) -> Linemod_Category_Handler:
        return Linemod_Category_Handler([Linemod_Category.from_dict(item_dict) for item_dict in dict_list])

class Linemod_Dataset(BasicLoadableObject['Linemod_Dataset']):
    def __init__(
        self,
        images: Linemod_Image_Handler,
        annotations: Linemod_Annotation_Handler,
        categories: Linemod_Category_Handler
    ):
        super().__init__()
        self.images = images
        self.annotations = annotations
        self.categories = categories
    
    def to_dict(self) -> dict:
        return {
            'images': self.images.to_dict_list(),
            'annotations': self.annotations.to_dict_list(),
            'categories': self.categories.to_dict_list()
        }
    
    @classmethod
    def from_dict(cls, item_dict: dict) -> Linemod_Dataset:
        return Linemod_Dataset(
            images=Linemod_Image_Handler.from_dict_list(item_dict['images']),
            annotations=Linemod_Annotation_Handler.from_dict_list(item_dict['annotations']),
            categories=Linemod_Category_Handler.from_dict_list(item_dict['categories'])
        )