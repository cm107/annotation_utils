from annotation_utils.coco.refactored.structs import COCO_Dataset
from common_utils.common_types.keypoint import Keypoint2D_List
from common_utils.common_types.segmentation import Segmentation

dataset = COCO_Dataset.load_from_path(
    json_path='split/train/coco/a.json',
    img_dir='split/train/img'
)
dataset.images.sort(attr_name='file_name')
dataset.display_preview(kpt_idx_offset=-1)