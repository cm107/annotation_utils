from logger import logger
from annotation_utils.coco.structs import COCO_Info

def reset_type_buffer(obj) -> object:
    return obj

# info = COCO_Info()
# info0 = info.copy()

# info1 = reset_type_buffer(info)
# info1 = COCO_Info.buffer(info1)
# logger.cyan(info1.contributor)

# logger.purple(f'info:\n{info}')
# logger.purple(f'info0:\n{info0}')

# logger.purple(f'info.to_dict():\n{info.to_dict()}')

info2 = COCO_Info.from_dict(
    {
        'description': 'This is a test',
        'url': 'https://test/url.com',
        'version': '1.0',
        'year': '2020',
        'contributor': 'Clayton',
        'date_created': '2020/03/10'
    }
)
logger.purple(f'info2:\n{info2}')

info2.save_to_path('info.json', overwrite=True)

from annotation_utils.coco.structs import COCO_Dataset
dataset = COCO_Dataset.combine_from_config(
    config_path='/home/clayton/workspace/prj/data_keep/data/toyota/dataset/config/yaml/box_hsr_kpt_sim.yaml',
    img_sort_attr_name='file_name',
    show_pbar=True
)
# dataset.display_preview(kpt_idx_offset=-1)
coco_ann = dataset.annotations[123]
logger.purple(coco_ann)
from common_utils.common_types.keypoint import Keypoint3D_List
coco_ann.keypoints_3d = Keypoint3D_List()
logger.purple(f'coco_ann.to_dict():\n{coco_ann.to_dict()}')