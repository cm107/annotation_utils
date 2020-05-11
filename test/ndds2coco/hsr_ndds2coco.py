from logger import logger
from annotation_utils.ndds.structs import NDDS_Dataset
from annotation_utils.coco.structs import COCO_Dataset, COCO_Category_Handler

# Load NDDS Dataset
ndds_dataset = NDDS_Dataset.load_from_dir(
    json_dir='/home/clayton/workspace/prj/data_keep/data/ndds/HSR',
    show_pbar=True
)

# Fix NDDS Dataset naming so that it follows convention. (This is not necessary if the NDDS dataset already follows the naming convention.)
for frame in ndds_dataset.frames:
    # Fix Naming Convention
    for ann_obj in frame.ndds_ann.objects:
        if ann_obj.class_name.startswith('hsr'):
            obj_type, obj_name = 'seg', 'hsr'
            instance_name = ann_obj.class_name.replace('hsr', '')
            ann_obj.class_name = f'{obj_type}_{obj_name}_{instance_name}'
        elif ann_obj.class_name.startswith('point'):
            obj_type, obj_name = 'kpt', 'hsr'
            temp = ann_obj.class_name.replace('point', '')
            instance_name, contained_name = temp[1], temp[0]
            ann_obj.class_name = f'{obj_type}_{obj_name}_{instance_name}_{contained_name}'
        else:
            logger.error(f'ann_obj.class_name: {ann_obj.class_name}')
            raise Exception
    
    # Delete Duplicate Objects
    frame.ndds_ann.objects.delete_duplicates(verbose=True, verbose_ref=frame.img_path)

ndds_dataset.save_to_path(save_path='hsr_fixed_ndds.json', overwrite=True)

# Convert To COCO Dataset
dataset = COCO_Dataset.from_ndds(
    ndds_dataset=ndds_dataset,
    categories=COCO_Category_Handler.load_from_path('/home/clayton/workspace/prj/data_keep/data/ndds/categories/hsr.json'),
    naming_rule='type_object_instance_contained',
    show_pbar=True,
    bbox_area_threshold=1
)

dataset.save_to_path('hsr_ndds2coco_test.json', overwrite=True)
dataset.display_preview(show_details=True)