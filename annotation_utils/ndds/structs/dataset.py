from __future__ import annotations
from common_utils.check_utils import check_dir_exists, check_file_exists, check_required_keys
from .frame import NDDS_Frame_Handler
from .settings import CameraConfig, ObjectSettings
from ...base.basic import BasicLoadableObject

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
        self, json_save_dir: str, src_img_dir: str,
        dst_img_dir: str=None, dst_camera_config_path: str=None, dst_obj_config_path: str=None,
        overwrite: bool=False, show_pbar: bool=False
    ):
        """Saves NDDS_Dataset object to a directory path.

        Arguments:
            json_save_dir {str} -- [Path to directory where you want to save the NDDS annotation json files.]
            src_img_dir {str} -- [Path to directory where the original NDDS images are saved.]

        Keyword Arguments:
            dst_img_dir {str} -- [Path to directory where you want to copy the original NDDS images.] (default: json_save_dir)
            dst_camera_config_path {str} -- [Path to where you would like to save the camera configuration settings json file.] (default: f'{json_save_dir}/_camera_settings.json')
            dst_obj_config_path {str} -- [Path to where you would like to save the object configuration settings json file.] (default: f'{json_save_dir}/_object_settings.json')
            overwrite {bool} -- [Whether or not you would like to overwrite existing files/directories.] (default: {False})
            show_pbar {bool} -- [Whether or not you would like to show the progress bar.] (default: {False})
        """
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