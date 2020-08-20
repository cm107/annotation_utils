from logger import logger
from annotation_utils.ndds.structs import NDDS_Dataset
from annotation_utils.coco.structs import COCO_Category_Handler
from annotation_utils.coco.dataset_specific import Measure_COCO_Dataset
from common_utils.file_utils import make_dir_if_not_exists, delete_all_files_in_dir

# Load NDDS Dataset
ndds_dataset = NDDS_Dataset.load_from_dir(
    json_dir='/home/clayton/workspace/prj/data_keep/data/ndds/measure_kume_map3_1_200',
    show_pbar=True
)

# Fix NDDS Dataset naming so that it follows convention. (This is not necessary if the NDDS dataset already follows the naming convention.)
target_obj_type = 'seg'

for frame in ndds_dataset.frames:
    # Fix Naming Convention
    for ann_obj in frame.ndds_ann.objects:
        # Note: Part numbers should be specified in the obj_type string.
        if ann_obj.class_name == 'measure':
            obj_type, obj_name = target_obj_type, 'measure'
            ann_obj.class_name = f'{obj_type}_{obj_name}'
        elif ann_obj.class_name.startswith('num_'):
            temp = ann_obj.class_name.replace('num_', '')
            temp_parts = temp.split('_')
            if len(temp_parts) == 2:
                obj_type, obj_name, instance_name = target_obj_type, temp_parts[1], temp_parts[0]
                ann_obj.class_name = f'{obj_type}_{obj_name}_{instance_name}'
            elif len(temp_parts) == 3:
                obj_type, obj_name, instance_name = target_obj_type, f'{temp_parts[1]}part{temp_parts[2]}', '0'
                ann_obj.class_name = f'{obj_type}_{obj_name}_{instance_name}'
            else:
                raise Exception

# Convert To COCO Dataset
dataset = Measure_COCO_Dataset.from_ndds(
    ndds_dataset=ndds_dataset,
    categories=COCO_Category_Handler.load_from_path('/home/clayton/workspace/prj/data_keep/data/ndds/categories/measure_all.json'),
    naming_rule='type_object_instance_contained', delimiter='_',
    ignore_unspecified_categories=True,
    show_pbar=True,
    bbox_area_threshold=1,
    default_visibility_threshold=0.10,
    visibility_threshold_dict={'measure': 0.01},
    allow_unfound_seg=True,
    class_merge_map={
        'mark_10th_place': 'seg_measure',
        'marking_bottom': 'seg_measure',
        'marking_top': 'seg_measure',
        'hook': 'seg_measure'
    }
)

# Output Directories
make_dir_if_not_exists('measure_coco')
delete_all_files_in_dir('measure_coco')

measure_dir = 'measure_coco/measure'
whole_number_dir = 'measure_coco/whole_number'
digit_dir = 'measure_coco/digit'
json_output_filename = 'output.json'

measure_dataset, whole_number_dataset, digit_dataset = dataset.split_measure_dataset(
    measure_dir=measure_dir,
    whole_number_dir=whole_number_dir,
    digit_dir=digit_dir,
    allow_no_measures=True,
    allow_missing_parts=True
)

if False: # Change to True if you want to remove all segmentation from the measure dataset.
    from common_utils.common_types.segmentation import Segmentation
    for coco_ann in measure_dataset.annotations:
        coco_ann.seg = Segmentation()

measure_dataset.display_preview(show_details=True, window_name='Measure Dataset Preview')
logger.info(f'Saving Measure Dataset')
measure_dataset.save_to_path(f'{measure_dir}/{json_output_filename}', overwrite=True)

whole_number_dataset.display_preview(show_details=True, window_name='Whole Number Dataset Preview')
logger.info(f'Saving Whole Number Dataset')
whole_number_dataset.save_to_path(f'{whole_number_dir}/{json_output_filename}', overwrite=True)

if False: # For debugging 2-digit digit annotations
    del_ann_id_list = []
    for coco_image in digit_dataset.images:
        anns = digit_dataset.annotations.get_annotations_from_imgIds([coco_image.id])
        if len(anns) == 1:
            del_ann_id_list.append(anns[0].id)
    digit_dataset.annotations.remove(del_ann_id_list)
    digit_dataset.images.remove_if_no_anns(
        ann_handler=digit_dataset.annotations,
        license_handler=digit_dataset.licenses,
        verbose=True
    )

digit_dataset.display_preview(show_details=True, window_name='Digit Dataset Preview')
logger.info(f'Saving Digit Dataset')
digit_dataset.save_to_path(f'{digit_dir}/{json_output_filename}', overwrite=True)