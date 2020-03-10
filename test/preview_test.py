from annotation_utils.coco.refactored.structs import COCO_Dataset
from common_utils.common_types.keypoint import Keypoint2D_List
from common_utils.common_types.segmentation import Segmentation

dataset = COCO_Dataset.combine_from_config(
    config_path='/home/clayton/workspace/prj/data_keep/data/toyota/dataset/config/yaml/box_hsr_kpt_trainval.yaml',
    img_sort_attr_name='file_name',
    show_pbar=True
)
dataset.display_preview(kpt_idx_offset=-1)