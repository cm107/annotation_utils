from __future__ import annotations
import sys
import traceback
from typing import List
from tqdm import tqdm
from logger import logger
from common_utils.check_utils import check_file_exists, check_required_keys, \
    check_dir_exists
from common_utils.path_utils import get_filename, get_rootname_from_path, \
    get_all_files_of_extension, get_valid_image_paths
from common_utils.file_utils import make_dir_if_not_exists, delete_all_files_in_dir, \
    copy_file
from ...base.basic import BasicLoadableObject, BasicLoadableHandler, BasicHandler
from .annotation import NDDS_Annotation
from .objects import NDDS_Annotation_Object
from .instance import LabeledObjectHandler, LabeledObject, ObjectInstance

class NDDS_Frame(BasicLoadableObject['NDDS_Frame']):
    def __init__(
        self, img_path: str, ndds_ann: NDDS_Annotation,
        cs_img_path: str=None, depth_img_path: str=None, is_img_path: str=None
    ):
        super().__init__()
        self.img_path = img_path
        self.ndds_ann = ndds_ann
        self.cs_img_path = cs_img_path
        self.depth_img_path = depth_img_path
        self.is_img_path = is_img_path

    def to_dict(self) -> dict:
        result = super().to_dict()
        none_keys = []
        for key, value in result.items():
            if value is None:
                none_keys.append(key)
        for key in none_keys:
            del result[key]
        return result

    @classmethod
    def from_dict(cls, item_dict: dict) -> NDDS_Frame:
        check_required_keys(
            item_dict,
            required_keys=[
                'img_path', 'ndds_ann'
            ]
        )
        return NDDS_Frame(
            img_path=item_dict['img_path'],
            ndds_ann=NDDS_Annotation.from_dict(item_dict['ndds_ann']),
            cs_img_path=item_dict['cs_img_path'] if 'cs_img_path' in item_dict else None,
            depth_img_path=item_dict['depth_img_path'] if 'depth_img_path' in item_dict else None,
            is_img_path=item_dict['is_img_path'] if 'is_img_path' in item_dict else None
        )

    def to_labeled_obj_handler(self, naming_rule: str='type_object_instance_contained', delimiter: str='_', show_pbar: bool=False) -> LabeledObjectHandler:
        # TODO: Debug this method
        def process_non_contained(handler: LabeledObjectHandler, ann_obj: NDDS_Annotation_Object):
            obj_type, obj_name, instance_name, contained_name = ann_obj.parse_obj_info(naming_rule=naming_rule, delimiter=delimiter)
            if obj_name not in handler.get_obj_names(): # New Object
                labeled_obj = LabeledObject(obj_name=obj_name)
                labeled_obj.instances.append(
                    ObjectInstance(
                        instance_type=obj_type,
                        ndds_ann_obj=ann_obj,
                        instance_name=instance_name
                    )
                )
                handler.append(labeled_obj)
            else: # Object Name already in handler
                try:
                    handler[handler.index(obj_name=obj_name)].instances.append(
                        ObjectInstance(
                            instance_type=obj_type,
                            ndds_ann_obj=ann_obj,
                            instance_name=instance_name
                        )
                    )
                except:
                    etype, evalue, tb = sys.exc_info()
                    e = traceback.format_tb(tb=tb)
                    print(''.join(e))
                    logger.error(f"Failed to add instance to {obj_type}_{obj_name}")
                    raise Exception

        def process_contained(handler: LabeledObjectHandler, ann_obj: NDDS_Annotation_Object):
            obj_type, obj_name, instance_name, contained_name = ann_obj.parse_obj_info(naming_rule=naming_rule, delimiter=delimiter)
            if obj_name not in handler.get_obj_names():
                logger.error(
                    f"Contained object (contained_name={contained_name}) " + \
                    f"cannot be defined before container object (obj_name={obj_name}) is defined."
                )
                raise Exception
            obj_idx = handler.index(obj_name=obj_name)
            if instance_name not in handler[obj_idx].instances.get_instance_names():
                logger.error(
                    f"Contained object (contained_name={contained_name}) " + \
                    f"cannot be defined before container object (obj_name={obj_name}) " + \
                    f" and container instance (instance_name={instance_name}) are defined."
                )
            instance_idx = handler[obj_idx].instances.index(instance_name=instance_name)
            try:
                handler[obj_idx].instances[instance_idx].append_contained(
                    ObjectInstance(
                        instance_type=obj_type,
                        ndds_ann_obj=ann_obj,
                        instance_name=contained_name
                    )
                )
            except:
                etype, evalue, tb = sys.exc_info()
                e = traceback.format_tb(tb=tb)
                print(''.join(e))
                logger.error(f"Failed to add contained instance to {obj_type}_{obj_name}_{instance_name}")
                raise Exception

        handler = LabeledObjectHandler()
        if naming_rule == 'type_object_instance_contained':
            # Add Non-contained Objects First
            if show_pbar:
                non_contained_pbar = tqdm(total=len(self.ndds_ann.objects), unit='ann_obj', leave=False)
                non_contained_pbar.set_description('Processing Containers')
            for ann_obj in self.ndds_ann.objects:
                obj_type, obj_name, instance_name, contained_name = ann_obj.parse_obj_info(naming_rule=naming_rule, delimiter=delimiter)
                if contained_name is None: # Non-contained Object
                    process_non_contained(handler=handler, ann_obj=ann_obj)
                if show_pbar:
                    non_contained_pbar.update()
        
            # Add Contained Objects Second
            if show_pbar:
                contained_pbar = tqdm(total=len(self.ndds_ann.objects), unit='ann_obj', leave=False)
                contained_pbar.set_description('Processing Containables')
            for ann_obj in self.ndds_ann.objects:
                obj_type, obj_name, instance_name, contained_name = ann_obj.parse_obj_info(naming_rule=naming_rule, delimiter=delimiter)
                if contained_name is not None: # Contained Object
                    process_contained(handler=handler, ann_obj=ann_obj)
                if show_pbar:
                    contained_pbar.update()
            return handler
        else:
            raise NotImplementedError

class NDDS_Frame_Handler(
    BasicLoadableHandler['NDDS_Frame_Handler', 'NDDS_Frame'],
    BasicHandler['NDDS_Frame_Handler', 'NDDS_Frame']
):
    def __init__(self, frames: List[NDDS_Frame]=None):
        super().__init__(obj_type=NDDS_Frame, obj_list=frames)
        self.frames = self.obj_list
    
    @classmethod
    def from_dict_list(cls, dict_list: List[dict]) -> NDDS_Frame_Handler:
        return NDDS_Frame_Handler([NDDS_Frame.from_dict(item_dict) for item_dict in dict_list])

    def _check_paths_valid(self, src_img_dir: str):
        check_dir_exists(src_img_dir)
        img_filename_list = []
        duplicate_img_filename_list = []
        for frame in self:
            img_filename = get_filename(frame.img_path)
            if img_filename not in img_filename_list:
                img_filename_list.append(frame.img_path)
            else:
                duplicate_img_filename_list.append(frame.img_path)
            img_path = f'{src_img_dir}/{img_filename}'
            check_file_exists(img_path)
            if frame.cs_img_path:
                check_file_exists(f'{src_img_dir}/{get_filename(frame.cs_img_path)}')
            if frame.depth_img_path:
                check_file_exists(f'{src_img_dir}/{get_filename(frame.depth_img_path)}')
            if frame.is_img_path:
                check_file_exists(f'{src_img_dir}/{get_filename(frame.is_img_path)}')
        if len(duplicate_img_filename_list) > 0:
            logger.error(f'Found the following duplicate image filenames in {self.__class__.__name__}:\n{duplicate_img_filename_list}')
            raise Exception

    def save_to_dir(self, json_save_dir: str, src_img_dir: str, overwrite: bool=False, dst_img_dir: str=None, show_pbar: bool=True):
        self._check_paths_valid(src_img_dir=src_img_dir)
        make_dir_if_not_exists(json_save_dir)
        delete_all_files_in_dir(json_save_dir, ask_permission=not overwrite)
        if dst_img_dir is not None:
            make_dir_if_not_exists(dst_img_dir)
            delete_all_files_in_dir(dst_img_dir, ask_permission=not overwrite)
        
        if show_pbar:
            pbar = tqdm(total=len(self), unit='ann(s)', leave=True)
            pbar.set_description(f'Saving {self.__class__.__name__}')
        for frame in self:
            save_path = f'{json_save_dir}/{get_rootname_from_path(frame.img_path)}.json'
            if dst_img_dir is not None:
                copy_file(
                    src_path=f'{src_img_dir}/{get_filename(frame.img_path)}',
                    dest_path=f'{dst_img_dir}/{get_filename(frame.img_path)}',
                    silent=True
                )
                if frame.cs_img_path:
                    copy_file(
                        src_path=f'{src_img_dir}/{get_filename(frame.cs_img_path)}',
                        dest_path=f'{dst_img_dir}/{get_filename(frame.cs_img_path)}',
                        silent=True
                    )
                if frame.depth_img_path:
                    copy_file(
                        src_path=f'{src_img_dir}/{get_filename(frame.depth_img_path)}',
                        dest_path=f'{dst_img_dir}/{get_filename(frame.depth_img_path)}',
                        silent=True
                    )
                if frame.is_img_path:
                    copy_file(
                        src_path=f'{src_img_dir}/{get_filename(frame.is_img_path)}',
                        dest_path=f'{dst_img_dir}/{get_filename(frame.is_img_path)}',
                        silent=True
                    )
                frame.ndds_ann.save_to_path(save_path=save_path)
            else:
                frame.ndds_ann.save_to_path(save_path=save_path)
            if show_pbar:
                pbar.update()

    @classmethod
    def load_from_dir(cls, img_dir: str, json_dir: str, show_pbar: bool=True) -> NDDS_Frame_Handler:
        check_dir_exists(json_dir)
        check_dir_exists(img_dir)

        img_pathlist = get_valid_image_paths(img_dir)
        json_path_list = [path for path in get_all_files_of_extension(dir_path=json_dir, extension='json') if not get_filename(path).startswith('_')]
        handler = NDDS_Frame_Handler()
        if show_pbar:
            pbar = tqdm(total=len(json_path_list), unit='ann(s)', leave=True)
            pbar.set_description(f'Loading {cls.__name__}')
        for json_path in json_path_list:
            check_file_exists(json_path)
            json_rootname = get_rootname_from_path(json_path)
            matching_img_path = None
            matching_cs_img_path = None
            matching_depth_img_path = None
            matching_is_img_path = None
            for img_path in img_pathlist:
                img_rootname = '.'.join(get_filename(img_path).split('.')[:-1])
                if img_rootname == json_rootname:
                    matching_img_path = img_path
                elif img_rootname == f'{json_rootname}.cs':
                    matching_cs_img_path = img_path
                elif img_rootname == f'{json_rootname}.depth':
                    matching_depth_img_path = img_path
                elif img_rootname == f'{json_rootname}.is':
                    matching_is_img_path = img_path
                if matching_img_path and matching_cs_img_path and matching_depth_img_path and matching_is_img_path:
                    break
            if matching_img_path is None:
                logger.error(f"Couldn't find image file that matches rootname of {get_filename(json_path)} in {img_dir}")
                raise FileNotFoundError
            frame = NDDS_Frame(
                img_path=matching_img_path, ndds_ann=NDDS_Annotation.load_from_path(json_path),
                cs_img_path=matching_cs_img_path, depth_img_path=matching_depth_img_path, is_img_path=matching_is_img_path
            )
            handler.append(frame)
            if show_pbar:
                pbar.update()
        return handler