from __future__ import annotations
from typing import List, Tuple
import numpy as np
from logger import logger
from common_utils.check_utils import check_required_keys, check_type, check_type_from_list
from common_utils.base.basic import BasicLoadableObject, BasicLoadableHandler, BasicHandler

class IntrinsicSettings(BasicLoadableObject['IntrinsicSettings']):
    def __init__(self, resX: int, resY: int, fx: float, fy: float, cx: float, cy: float, s: int):
        super().__init__()
        check_type(resX, valid_type_list=[int])
        self.resX = resX
        check_type(resY, valid_type_list=[int])
        self.resY = resY
        check_type(fx, valid_type_list=[float, int])
        self.fx = fx
        check_type(fy, valid_type_list=[float, int])
        self.fy = fy
        check_type(cx, valid_type_list=[float, int])
        self.cx = cx
        check_type(cy, valid_type_list=[float, int])
        self.cy = cy
        check_type(s, valid_type_list=[int])
        self.s = s

class CapturedImageSize(BasicLoadableObject['CapturedImageSize']):
    def __init__(self, width: int, height: int):
        super().__init__()
        check_type(width, valid_type_list=[int])
        self.width = width
        check_type(height, valid_type_list=[int])
        self.height = height

    def shape(self) -> Tuple[int]:
        """Returns (self.height, self.width)

        Returns:
            Tuple[int] -- [Image shape]
        """
        return (self.height, self.width, 3)

class CameraSettings(BasicLoadableObject['CameraSettings']):
    def __init__(self, name: str, horizontal_fov: int, intrinsic_settings: IntrinsicSettings, captured_image_size: CapturedImageSize):
        super().__init__()
        check_type(name, valid_type_list=[str])
        self.name = name
        check_type(horizontal_fov, valid_type_list=[int])
        self.horizontal_fov = horizontal_fov
        check_type(intrinsic_settings, valid_type_list=[IntrinsicSettings])
        self.intrinsic_settings = intrinsic_settings
        check_type(captured_image_size, valid_type_list=[CapturedImageSize])
        self.captured_image_size = captured_image_size
    
    @classmethod
    def from_dict(cls, item_dict: dict) -> CameraSettings:
        check_required_keys(
            item_dict,
            required_keys=['name', 'horizontal_fov', 'intrinsic_settings', 'captured_image_size']
        )
        return CameraSettings(
            name=item_dict['name'],
            horizontal_fov=item_dict['horizontal_fov'],
            intrinsic_settings=IntrinsicSettings.from_dict(item_dict['intrinsic_settings']),
            captured_image_size=CapturedImageSize.from_dict(item_dict['captured_image_size'])
        )

class CameraSettingsHandler(
    BasicLoadableHandler['CameraSettingsHandler', 'CameraSettings'],
    BasicHandler['CameraSettingsHandler', 'CameraSettings']
):
    def __init__(self, settings_list: List[CameraSettings]=None):
        super().__init__(obj_type=CameraSettings, obj_list=settings_list)
        self.settings_list = self.obj_list
        check_type_from_list(self.settings_list, valid_type_list=[CameraSettings])
    
    @classmethod
    def from_dict_list(cls, dict_list: List[dict]) -> CameraSettingsHandler:
        return CameraSettingsHandler(settings_list=[CameraSettings.from_dict(item_dict) for item_dict in dict_list])

class CameraConfig(BasicLoadableObject['CameraConfig']):
    def __init__(self, camera_settings: CameraSettingsHandler=None):
        super().__init__()
        self.camera_settings = camera_settings if camera_settings is not None else CameraSettingsHandler()
        check_type(self.camera_settings, valid_type_list=[CameraSettingsHandler])
    
    @classmethod
    def from_dict(cls, item_dict: dict) -> CameraConfig:
        check_required_keys(item_dict, required_keys=['camera_settings'])
        return CameraConfig(
            camera_settings=CameraSettingsHandler.from_dict_list(item_dict['camera_settings'])
        )

class ExportedObject(BasicLoadableObject['ExportedObject']):
    def __init__(
        self, class_name: str, segmentation_class_id: int, segmentation_instance_id: int,
        fixed_model_transform: np.ndarray, cuboid_dimensions: list
    ):
        super().__init__()
        check_type(class_name, valid_type_list=[str])
        self.class_name = class_name
        check_type(segmentation_class_id, valid_type_list=[int])
        self.segmentation_class_id = segmentation_class_id
        check_type(segmentation_instance_id, valid_type_list=[int])
        self.segmentation_instance_id = segmentation_instance_id
        check_type(fixed_model_transform, valid_type_list=[np.ndarray])
        if fixed_model_transform.shape != (4, 4):
            logger.error(f'fixed_model_transform.shape == {fixed_model_transform.shape} != (4, 4)')
            raise Exception
        self.fixed_model_transform = fixed_model_transform
        check_type(cuboid_dimensions, valid_type_list=[list])
        self.cuboid_dimensions = cuboid_dimensions
    
    def to_dict(self) -> dict:
        return {
            'class': self.class_name,
            'segmentation_class_id': self.segmentation_class_id,
            'segmentation_instance_id': self.segmentation_instance_id,
            'fixed_model_transform': self.fixed_model_transform.tolist(),
            'cuboid_dimensions': self.cuboid_dimensions
        }
    
    @classmethod
    def from_dict(self, item_dict: dict) -> ExportedObject:
        check_required_keys(
            item_dict,
            required_keys=[
                'class', 'segmentation_class_id',
                'segmentation_instance_id', 'fixed_model_transform',
                'cuboid_dimensions'
            ]
        )
        return ExportedObject(
            class_name=item_dict['class'],
            segmentation_class_id=item_dict['segmentation_class_id'],
            segmentation_instance_id=item_dict['segmentation_instance_id'],
            fixed_model_transform=np.array(item_dict['fixed_model_transform']),
            cuboid_dimensions=item_dict['cuboid_dimensions']
        )

class ExportedObjectHandler(
    BasicLoadableHandler['ExportedObjectHandler', 'ExportedObject'],
    BasicHandler['ExportedObjectHandler', 'ExportedObject']
):
    def __init__(self, exported_obj_list: List[ExportedObject]=None):
        super().__init__(obj_type=ExportedObject, obj_list=exported_obj_list)
        self.exported_obj_list = self.obj_list
        check_type_from_list(self.exported_obj_list, valid_type_list=[ExportedObject])
    
    @classmethod
    def from_dict_list(cls, dict_list: List[dict]) -> ExportedObjectHandler:
        return ExportedObjectHandler(exported_obj_list=[ExportedObject.from_dict(item_dict) for item_dict in dict_list])

class ObjectSettings(BasicLoadableObject['ObjectSettings']):
    def __init__(self, exported_object_classes: List[str]=None, exported_objects: ExportedObjectHandler=None):
        super().__init__()
        self.exported_object_classes = exported_object_classes if exported_object_classes is not None else []
        check_type_from_list(self.exported_object_classes, valid_type_list=[str])
        self.exported_objects = exported_objects if exported_objects is not None else ExportedObjectHandler()
        check_type(self.exported_objects, valid_type_list=[ExportedObjectHandler])
    
    @classmethod
    def from_dict(cls, item_dict: dict) -> ObjectSettings:
        check_required_keys(
            item_dict,
            required_keys=['exported_object_classes', 'exported_objects']
        )
        return ObjectSettings(
            exported_object_classes=item_dict['exported_object_classes'],
            exported_objects=ExportedObjectHandler.from_dict_list(item_dict['exported_objects'])
        )