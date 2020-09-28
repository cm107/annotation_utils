from logger import logger
from annotation_utils.ndds.structs import NDDS_Dataset
from annotation_utils.coco.structs import COCO_Dataset, COCO_Category_Handler
from common_utils.file_utils import make_dir_if_not_exists, delete_all_files_in_dir

src_dir = '/home/clayton/workspace/prj/data_keep/data/ndds/20200924_yagura5_bright_1-colored'
dst_dir = 'nihonbashi_debug'
make_dir_if_not_exists(dst_dir)
delete_all_files_in_dir(dst_dir, ask_permission=False)

hsr_categories = COCO_Category_Handler.load_from_path('/home/clayton/workspace/prj/data_keep/data/toyota/dataset/config/categories/hsr_categories.json')

# Load NDDS Dataset
ndds_dataset = NDDS_Dataset.load_from_dir(
    json_dir=src_dir,
    show_pbar=True
)

# Fix NDDS Dataset naming so that it follows convention. (This is not necessary if the NDDS dataset already follows the naming convention.)
for frame in ndds_dataset.frames:
    # Fix Naming Convention
    for ann_obj in frame.ndds_ann.objects:
        if ann_obj.class_name.lower() == 'nihonbashi':
            obj_type, obj_name = 'seg', 'hsr'
            instance_name = '0'
            ann_obj.class_name = f'{obj_type}_{obj_name}_{instance_name}'
        elif ann_obj.class_name.lower() in list('abcdefghijkl'):
            obj_type, obj_name = 'kpt', 'hsr'
            instance_name, contained_name = '0', ann_obj.class_name
            ann_obj.class_name = f'{obj_type}_{obj_name}_{instance_name}_{contained_name}'
        else:
            logger.error(f'Unknown ann_obj.class_name: {ann_obj.class_name}')
            raise Exception

# Convert To COCO Dataset
dataset = COCO_Dataset.from_ndds(
    ndds_dataset=ndds_dataset,
    categories=hsr_categories,
    naming_rule='type_object_instance_contained',
    show_pbar=True,
    bbox_area_threshold=1,
    allow_same_instance_for_contained=True,
    color_interval=5
)
dataset.move_images(
    dst_dir,
    preserve_filenames=True,
    update_img_paths=True,
    overwrite=True,
    show_pbar=True
)
dataset.save_to_path(f'{dst_dir}/output.json', overwrite=True)
dataset.save_video(
    save_path=f'{dst_dir}/preview_with_mask.avi',
    fps=5,
    show_details=True,
    kpt_idx_offset=-1,
)
dataset.save_video(
    save_path=f'{dst_dir}/preview.avi',
    fps=5,
    show_details=True,
    kpt_idx_offset=-1,
    show_seg=False
)