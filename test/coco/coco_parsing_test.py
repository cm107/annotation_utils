from logger import logger
from annotation_utils.coco.structs import COCO_Dataset
dataset = COCO_Dataset.load_from_path(
    json_path='/home/clayton/workspace/prj/data_keep/data/toyota/dataset/sim/20200228/28_02_2020_11_18_30_coco-data/HSR-coco.json',
    img_dir='/home/clayton/workspace/prj/data_keep/data/toyota/dataset/sim/20200228/28_02_2020_11_18_30_coco-data'
)
for coco_cat in dataset.categories:
    logger.purple(f'name: {coco_cat.name}')
    label_skeleton = coco_cat.get_label_skeleton(skeleton_idx_offset=1)
    logger.cyan(f'label_skeleton: {label_skeleton}')