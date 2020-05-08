from annotation_utils.ndds.structs.settings import CameraConfig
from logger import logger

config = CameraConfig.load_from_path('/home/clayton/workspace/prj/data_keep/data/ndds/HSR/_camera_settings.json')
logger.purple(config)
assert config == CameraConfig.from_dict(config.to_dict())
