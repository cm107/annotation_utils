from annotation_utils.ndds.structs import NDDS_Dataset
from annotation_utils.coco.structs import COCO_Category_Handler, COCO_Dataset
from common_utils.file_utils import file_exists

result_json = 'conv_test.json'

if not file_exists(result_json):
    # Load NDDS Dataset
    ndds_dataset = NDDS_Dataset.load_from_dir(
        json_dir='/home/clayton/workspace/prj/data_keep/data/ndds/mv_500',
        show_pbar=True
    )

    # Fix NDDS Dataset naming so that it follows convention. (This is not necessary if the NDDS dataset already follows the naming convention.)
    number_spelling_map = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
        'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9
    }

    for frame in ndds_dataset.frames:
        # Fix Naming Convention
        for ann_obj in frame.ndds_ann.objects:
            # Note: Part numbers should be specified in the obj_type string.
            if ann_obj.class_name == 'measure':
                obj_type, obj_name = 'seg', 'measure'
                ann_obj.class_name = f'{obj_type}_{obj_name}'
            elif ann_obj.class_name.startswith('num_'):
                temp = ann_obj.class_name.replace('num_', '')
                obj_type, obj_name, instance_name = 'seg', temp[:-1], temp[-1]
                obj_name = number_spelling_map[obj_name]
                ann_obj.class_name = f'{obj_type}_{obj_name}_{instance_name}'
            elif ann_obj.class_name == 'part_one0':
                obj_type, obj_name, instance_name = 'seg', '10part1', '0'
                ann_obj.class_name = f'{obj_type}_{obj_name}_{instance_name}'
            elif ann_obj.class_name == '10part_zero1':
                obj_type, obj_name, instance_name = 'seg', '10part0', '0'
                ann_obj.class_name = f'{obj_type}_{obj_name}_{instance_name}'
            elif ann_obj.class_name == 'part_two0':
                obj_type, obj_name, instance_name = 'seg', '20part2', '0'
                ann_obj.class_name = f'{obj_type}_{obj_name}_{instance_name}'
            elif ann_obj.class_name == '20part_zero2':
                obj_type, obj_name, instance_name = 'seg', '20part0', '0'
                ann_obj.class_name = f'{obj_type}_{obj_name}_{instance_name}'
            elif ann_obj.class_name == 'part_three0':
                obj_type, obj_name, instance_name = 'seg', '30part3', '0'
                ann_obj.class_name = f'{obj_type}_{obj_name}_{instance_name}'
            elif ann_obj.class_name == '30part_zero3':
                obj_type, obj_name, instance_name = 'seg', '30part0', '0'
                ann_obj.class_name = f'{obj_type}_{obj_name}_{instance_name}'
            elif ann_obj.class_name == 'part_four0':
                obj_type, obj_name, instance_name = 'seg', '40part4', '0'
                ann_obj.class_name = f'{obj_type}_{obj_name}_{instance_name}'
            elif ann_obj.class_name == '40part_zero4':
                obj_type, obj_name, instance_name = 'seg', '40part0', '0'
                ann_obj.class_name = f'{obj_type}_{obj_name}_{instance_name}'
            elif ann_obj.class_name == 'part_five0':
                obj_type, obj_name, instance_name = 'seg', '50part5', '0'
                ann_obj.class_name = f'{obj_type}_{obj_name}_{instance_name}'
            elif ann_obj.class_name == '50part_zero5':
                obj_type, obj_name, instance_name = 'seg', '50part0', '0'
                ann_obj.class_name = f'{obj_type}_{obj_name}_{instance_name}'
            elif ann_obj.class_name == 'part_six0':
                obj_type, obj_name, instance_name = 'seg', '60part6', '0'
                ann_obj.class_name = f'{obj_type}_{obj_name}_{instance_name}'
            elif ann_obj.class_name == '60part_zero6':
                obj_type, obj_name, instance_name = 'seg', '60part0', '0'
                ann_obj.class_name = f'{obj_type}_{obj_name}_{instance_name}'
            elif ann_obj.class_name == 'part_seven0':
                obj_type, obj_name, instance_name = 'seg', '70part7', '0'
                ann_obj.class_name = f'{obj_type}_{obj_name}_{instance_name}'
            elif ann_obj.class_name == '70part_zero7':
                obj_type, obj_name, instance_name = 'seg', '70part0', '0'
                ann_obj.class_name = f'{obj_type}_{obj_name}_{instance_name}'
            elif ann_obj.class_name == 'part_eight0':
                obj_type, obj_name, instance_name = 'seg', '80part8', '0'
                ann_obj.class_name = f'{obj_type}_{obj_name}_{instance_name}'
            elif ann_obj.class_name == '80part_zero8':
                obj_type, obj_name, instance_name = 'seg', '80part0', '0'
                ann_obj.class_name = f'{obj_type}_{obj_name}_{instance_name}'
            elif ann_obj.class_name == 'part_nine0':
                obj_type, obj_name, instance_name = 'seg', '90part9', '0'
                ann_obj.class_name = f'{obj_type}_{obj_name}_{instance_name}'
            elif ann_obj.class_name == '90part_zero9':
                obj_type, obj_name, instance_name = 'seg', '90part0', '0'
                ann_obj.class_name = f'{obj_type}_{obj_name}_{instance_name}'

    # Convert To COCO Dataset
    dataset = COCO_Dataset.from_ndds(
        ndds_dataset=ndds_dataset,
        categories=COCO_Category_Handler.load_from_path('/home/clayton/workspace/prj/data_keep/data/ndds/categories/measure_all.json'),
        naming_rule='type_object_instance_contained', delimiter='_',
        ignore_unspecified_categories=True,
        show_pbar=True,
        bbox_area_threshold=1,
        exclude_invalid_polygons=True
    )
    dataset.save_to_path(result_json)
else:
    dataset = COCO_Dataset.load_from_path(result_json)

dataset.remove_categories_by_name(category_names=['measure'])
dataset.images.sort(attr_name='file_name')
dataset.save_visualization(
    save_dir='measure_test_vis',
    preserve_filenames=True,
    show_details=True,
    bbox_thickness=1, bbox_label_thickness=1, bbox_label_color=[0,255,0], bbox_label_orientation='right',
    show_seg=True, seg_color=[0,0,255], seg_transparent=False
)