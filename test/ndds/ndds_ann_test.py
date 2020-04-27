from logger import logger
from common_utils.path_utils import get_all_files_of_extension, get_filename
from annotation_utils.ndds.structs.annotation import NDDS_Annotation

json_dir = '/home/clayton/workspace/prj/data_keep/data/ndds/HSR'
json_paths = get_all_files_of_extension(dir_path=json_dir, extension='json')

for json_path in json_paths:
    json_path.startswith('_')
    if get_filename(json_path).startswith('_'):
        logger.yellow(f'Skip {json_path}')
        continue
    logger.cyan(json_path)
    ann = NDDS_Annotation.load_from_path(json_path)
    assert ann.to_dict() == NDDS_Annotation.from_dict(ann.to_dict()).to_dict()
    logger.purple(ann)