from logger import logger
from annotation_utils.ndds.structs import ObjectSettings

settings = ObjectSettings.load_from_path('/home/clayton/workspace/prj/data_keep/data/ndds/HSR/_object_settings.json')
assert settings == ObjectSettings.from_dict(settings.to_dict())

logger.purple(settings)