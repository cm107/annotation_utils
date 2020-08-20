from __future__ import annotations
from tqdm import tqdm
from typing import Dict
from common_utils.check_utils import check_dir_exists, check_file_exists, check_required_keys
from common_utils.path_utils import get_dirpath_from_filepath
from common_utils.base.basic import BasicLoadableObject
from logger import logger
from .frame import NDDS_Frame_Handler
from .settings import CameraConfig, ObjectSettings

class NDDS_Dataset(BasicLoadableObject['NDDS_Dataset']):
    """Class object that represents all of the data in an NDDS dataset.

    Arguments:
        BasicLoadableObject {[str]} -- [Abstraction of a loadable class]
    """
    def __init__(self, camera_config: CameraConfig, obj_config: ObjectSettings, frames: NDDS_Frame_Handler=None):
        super().__init__()
        self.camera_config = camera_config
        self.obj_config = obj_config
        self.frames = frames if frames is not None else NDDS_Frame_Handler()

    def _check_valid_object_settings(self):
        frame_labels = []
        for frame in self.frames:
            for ann_obj in frame.ndds_ann.objects:
                if ann_obj.class_name not in frame_labels:
                    frame_labels.append(ann_obj.class_name)
        matched_labels = []
        unused_labels = []
        missing_labels = []
        for label in self.obj_config.exported_object_classes:
            if label in frame_labels:
                matched_labels.append(label)
            else:
                unused_labels.append(label)
        for label in frame_labels:
            if label not in self.obj_config.exported_object_classes:
                missing_labels.append(label)
        if len(missing_labels) > 0:
            logger.error(f"Found some labels in the frames that don't exist in ObjectSettings.")
            logger.error(f'matched_labels: {matched_labels}')
            logger.error(f'unused_labels: {unused_labels}')
            logger.error(f'missing_labels: {missing_labels}')
            raise Exception

    def get_img_dir(self, check_paths: bool=True, show_pbar: bool=True) -> str:
        """Returns the image directory of the dataset if all of the registered images are in the same directory.
        Otherwise, None is returned.

        Keyword Arguments:
            check_paths {bool} -- [Whether or not you want to verify that the images paths exist during the scan.] (default: {True})

        Returns:
            str -- [Path to the dataset's image directory]
        """
        img_dir = None

        if show_pbar:
            pbar = tqdm(total=len(self.frames), unit='frame(s)', leave=True)
            pbar.set_description('Locating Image Directory')
        for frame in self.frames:
            if check_paths:
                check_file_exists(frame.img_path)
                check_file_exists(frame.is_img_path)
                check_file_exists(frame.cs_img_path)
                check_file_exists(frame.depth_img_path)
            pending_img_dir = get_dirpath_from_filepath(frame.img_path)
            pending_is_img_dir = get_dirpath_from_filepath(frame.is_img_path)
            pending_cs_img_dir = get_dirpath_from_filepath(frame.cs_img_path)
            pending_depth_img_dir = get_dirpath_from_filepath(frame.depth_img_path)
            
            frame_has_common_dir = all([pending_dir == pending_img_dir for pending_dir in [pending_is_img_dir, pending_cs_img_dir, pending_depth_img_dir]])
            if frame_has_common_dir:
                if img_dir is None:
                    img_dir = pending_img_dir
                elif img_dir != pending_img_dir:
                    if show_pbar:
                        pbar.close()
                    return None
                else:
                    pass
            else:
                if show_pbar:
                    pbar.close()
                return None
            if show_pbar:
                pbar.update()
        return img_dir

    @classmethod
    def from_dict(cls, item_dict: dict) -> NDDS_Dataset:
        check_required_keys(
            item_dict,
            required_keys=['camera_config', 'obj_config', 'frames']
        )
        return NDDS_Dataset(
            camera_config=CameraConfig.from_dict(item_dict['camera_config']),
            obj_config=ObjectSettings.from_dict(item_dict['obj_config']),
            frames=NDDS_Frame_Handler.from_dict_list(item_dict['frames'])
        )

    @classmethod
    def load_from_dir(
        cls, json_dir: str,
        img_dir: str=None, camera_config_path: str=None, obj_config_path: str=None, show_pbar: bool=False
    ) -> NDDS_Dataset:
        """Loads NDDS_Dataset object from a directory path.

        Arguments:
            json_dir {str} -- [Path to directory with all of the NDDS annotation json files.]

        Keyword Arguments:
            img_dir {str} -- [Path to directory with all of the NDDS image files.] (default: json_dir)
            camera_config_path {str} -- [Path to the camera configuration json file.] (default: f'{json_dir}/_camera_settings.json')
            obj_config_path {str} -- [Path to the object configuration json file.] (default: f'{json_dir}/_object_settings.json')
            show_pbar {bool} -- [Show the progress bar.] (default: {False})

        Returns:
            NDDS_Dataset -- [NDDS_Dataset object]
        """
        check_dir_exists(json_dir)
        if img_dir is None:
            img_dir = json_dir
        else:
            check_dir_exists(img_dir)
        camera_config_path = camera_config_path if camera_config_path is not None else f'{json_dir}/_camera_settings.json'
        check_file_exists(camera_config_path)
        obj_config_path = obj_config_path if obj_config_path is not None else f'{json_dir}/_object_settings.json'
        check_file_exists(obj_config_path)

        return NDDS_Dataset(
            camera_config=CameraConfig.load_from_path(camera_config_path),
            obj_config=ObjectSettings.load_from_path(obj_config_path),
            frames=NDDS_Frame_Handler.load_from_dir(
                img_dir=img_dir,
                json_dir=json_dir,
                show_pbar=show_pbar
            )
        )
    
    def save_to_dir(
        self, json_save_dir: str,
        src_img_dir: str=None, dst_img_dir: str=None, dst_camera_config_path: str=None, dst_obj_config_path: str=None,
        overwrite: bool=False, show_pbar: bool=False
    ):
        """Saves NDDS_Dataset object to a directory path.

        Arguments:
            json_save_dir {str} -- [Path to directory where you want to save the NDDS annotation json files.]

        Keyword Arguments:
            src_img_dir {str} -- [
                Path to directory where the original NDDS images are saved.
                If not specified, the image path will be calculated from the cache of this object,
                but if the paths have changed, then you will need to explicitly provide the source image directory here.
            ] (default: Detected automatically)
            dst_img_dir {str} -- [Path to directory where you want to copy the original NDDS images.] (default: json_save_dir)
            dst_camera_config_path {str} -- [Path to where you would like to save the camera configuration settings json file.] (default: f'{json_save_dir}/_camera_settings.json')
            dst_obj_config_path {str} -- [Path to where you would like to save the object configuration settings json file.] (default: f'{json_save_dir}/_object_settings.json')
            overwrite {bool} -- [Whether or not you would like to overwrite existing files/directories.] (default: {False})
            show_pbar {bool} -- [Whether or not you would like to show the progress bar.] (default: {False})
        """
        src_img_dir = src_img_dir if src_img_dir is not None else self.get_img_dir(check_paths=True, show_pbar=show_pbar)
        if src_img_dir is None:
            logger.error(f'Failed to detect the image directory of this dataset.')
            logger.error(f'Are you sure that all of your image files are in the same folder?')
            raise Exception

        dst_img_dir = dst_img_dir if dst_img_dir is not None else json_save_dir
        self.frames.save_to_dir(
            json_save_dir=json_save_dir,
            src_img_dir=src_img_dir,
            dst_img_dir=dst_img_dir,
            overwrite=overwrite,
            show_pbar=show_pbar
        )
        self.camera_config.save_to_path(
            save_path=dst_camera_config_path if dst_camera_config_path is not None else f'{json_save_dir}/_camera_settings.json',
            overwrite=overwrite
        )
        self.obj_config.save_to_path(
            save_path=dst_obj_config_path if dst_obj_config_path is not None else f'{json_save_dir}/_object_settings.json'
        )
    
    def _check_valid_merge_map(self, merge_map: Dict[str, str]):
        # Check merge_map is not empty.
        if len(merge_map) == 0:
            logger.error(f'len(merge_map) == 0')
            raise Exception

        # Check that the object settings are valid.
        self._check_valid_object_settings()

        # Check that all of the classes in merge_map exist in the object settings.
        invalid_classes = []
        for src_class, dst_class in merge_map.items():
            if src_class not in self.obj_config.exported_object_classes:
                invalid_classes.append(src_class)
            if dst_class not in self.obj_config.exported_object_classes:
                invalid_classes.append(dst_class)
        if len(invalid_classes) > 0:
            logger.error(f'Found invalid classes in merge_map.')
            logger.error(f'Invalid classes: {invalid_classes}')
            logger.error(f'self.obj_config.exported_object_classes: {self.obj_config.exported_object_classes}')
            raise Exception

        # Check that the maps are possible
        availability_map = {}
        erroneous_maps = {}
        for class_name in self.obj_config.exported_object_classes:
            availability_map[class_name] = None
        for src_class, dst_class in merge_map.items():
            if availability_map[src_class] is None:
                if availability_map[dst_class] is not None:
                    logger.error(f'Cannot map {src_class} to {dst_class} because {dst_class} is mapped to {availability_map[dst_class]}')
                    raise Exception
                availability_map[src_class] = dst_class
            else:
                if src_class not in erroneous_maps:
                    erroneous_maps[src_class] = [dst_class]
                else:
                    erroneous_maps[src_class].append(dst_class)
        if len(erroneous_maps) > 0:
            logger.error(f'Invalid merge_map: {merge_map}')
            for class_name, dst_class_names in erroneous_maps.items():
                logger.error(f'{class_name} is already mapped to {availability_map[class_name]}')
                logger.error(f'Failed to map {class_name} to: {erroneous_maps[class_name]}')
            raise Exception

    # def merge_classes(self, merge_map: Dict[str, str]):
    #     self.__check_valid_merge_map(merge_map)
    #     for frame in self.frames:
    #         frame.merge_classes(merge_map=merge_map)