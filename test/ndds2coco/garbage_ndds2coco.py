from logger import logger
from annotation_utils.ndds.structs import NDDS_Dataset
from annotation_utils.coco.structs import COCO_Dataset, COCO_Category_Handler

# Load NDDS Dataset
ndds_dataset = NDDS_Dataset.load_from_dir(
    json_dir='/home/clayton/workspace/prj/data_keep/data/ndds/garbage_can/combined_instance',
    show_pbar=True
)

# Fix NDDS Dataset naming so that it follows convention. (This is not necessary if the NDDS dataset already follows the naming convention.)
for frame in ndds_dataset.frames:
    # Fix Naming Convention
    for ann_obj in frame.ndds_ann.objects:
        if ann_obj.class_name.startswith('garbage'):
            obj_type, obj_name = 'seg', 'garbage'
            instance_name = ann_obj.class_name.replace('garbage', '')
            ann_obj.class_name = f'{obj_type}_{obj_name}_{instance_name}'
        else:
            logger.error(f'ann_obj.class_name: {ann_obj.class_name}')
            raise Exception

# Convert To COCO Dataset
dataset = COCO_Dataset.from_ndds(
    ndds_dataset=ndds_dataset,
    categories=COCO_Category_Handler.load_from_path('/home/clayton/workspace/prj/data_keep/data/ndds/categories/garbage.json'),
    naming_rule='type_object_instance_contained',
    show_pbar=True,
    bbox_area_threshold=1
)

dataset.save_to_path('garbage_ndds2coco.json', overwrite=True)
dataset.display_preview(show_details=True)