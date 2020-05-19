from logger import logger
from annotation_utils.ndds.structs import NDDS_Dataset
from annotation_utils.coco.structs import COCO_Dataset, COCO_Category_Handler

# Load NDDS Dataset
ndds_dataset = NDDS_Dataset.load_from_dir(
    json_dir='/home/clayton/workspace/prj/data_keep/data/ndds/NewMap2',
    show_pbar=True
)

# Fix NDDS Dataset naming so that it follows convention. (This is not necessary if the NDDS dataset already follows the naming convention.)
for frame in ndds_dataset.frames:
    # Fix Naming Convention
    for ann_obj in frame.ndds_ann.objects:
        # Note: Part numbers should be specified in the obj_type string.
        if ann_obj.class_name == 'colorcone1':
            obj_type, obj_name = 'seg0', 'cone'
            ann_obj.class_name = f'{obj_type}_{obj_name}'
        elif ann_obj.class_name == 'colorcone2':
            obj_type, obj_name = 'seg1', 'cone'
            ann_obj.class_name = f'{obj_type}_{obj_name}'

ndds_dataset.save_to_path(save_path='cone_fixed_ndds.json', overwrite=True)

# Convert To COCO Dataset
dataset = COCO_Dataset.from_ndds(
    ndds_dataset=ndds_dataset,
    categories=COCO_Category_Handler.load_from_path('/home/clayton/workspace/prj/data_keep/data/ndds/categories/cone.json'),
    naming_rule='type_object_instance_contained', delimiter='_',
    ignore_unspecified_categories=True,
    show_pbar=True
)

dataset.save_to_path('cone_ndds2coco_test.json', overwrite=True)
dataset.display_preview(show_details=True)