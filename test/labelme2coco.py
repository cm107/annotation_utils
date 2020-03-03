from logger import logger
from annotation_utils.labelme.refactored import LabelmeAnnotationHandler
from annotation_utils.coco.refactored.structs import COCO_Dataset, COCO_Category, COCO_Category_Handler

img_dir = '/home/clayton/workspace/prj/KeypointPose/test/ann_parsing/test_img'
json_dir = '/home/clayton/workspace/prj/KeypointPose/test/ann_parsing/test_json'

labelme_handler = LabelmeAnnotationHandler.load_from_dir(load_dir=json_dir)

coco_dataset = COCO_Dataset.from_labelme(
    labelme_handler=labelme_handler,
    categories=COCO_Category_Handler.load_from_path('box_hsr_categories.json'),
    img_dir=img_dir
)
coco_dataset.save_to_path(save_path='output.json', overwrite=True)

test_coco_dataset = COCO_Dataset.load_from_path(json_path='output.json')
test_coco_dataset.display_preview(kpt_idx_offset=0)