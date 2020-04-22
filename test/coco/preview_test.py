from annotation_utils.coco.structs import COCO_Dataset
from common_utils.file_utils import file_exists

ann_save_path = 'preview_dataset.json'
if not file_exists(ann_save_path):
    dataset = COCO_Dataset.combine_from_config(
        config_path='/home/clayton/workspace/prj/data_keep/data/toyota/dataset/config/yaml/box_hsr_kpt_trainval.yaml',
        img_sort_attr_name='file_name',
        show_pbar=True
    )
    dataset.save_to_path(ann_save_path)
else:
    dataset = COCO_Dataset.load_from_path(
        json_path=ann_save_path
    )
dataset.display_preview(kpt_idx_offset=-1, show_details=True)