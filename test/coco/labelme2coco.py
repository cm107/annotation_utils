from annotation_utils.labelme.structs import LabelmeAnnotationHandler
from annotation_utils.coco.structs import COCO_Dataset, COCO_Category_Handler

img_dir = '/home/clayton/workspace/prj/data_keep/data/toyota/dataset/real/phone_videos/new/sampled_data/VID_20200217_161043/img'
json_dir = '/home/clayton/workspace/prj/data_keep/data/toyota/dataset/real/phone_videos/new/sampled_data/VID_20200217_161043/json'

labelme_handler = LabelmeAnnotationHandler.load_from_dir(load_dir=json_dir)

coco_dataset = COCO_Dataset.from_labelme(
    labelme_handler=labelme_handler,
    categories=COCO_Category_Handler.load_from_path('/home/clayton/workspace/prj/data_keep/data/toyota/dataset/config/categories/hsr_categories.json'),
    img_dir=img_dir
)
coco_dataset.save_to_path(save_path='output.json', overwrite=True)

test_coco_dataset = COCO_Dataset.load_from_path(json_path='output.json')
test_coco_dataset.display_preview(kpt_idx_offset=-1)