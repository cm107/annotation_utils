from logger import logger
from annotation_utils.ndds.structs import NDDS_Dataset
from annotation_utils.coco.structs import COCO_Category_Handler, COCO_Category
from annotation_utils.coco.structs import COCO_Dataset

# Load NDDS Dataset
ndds_dataset = NDDS_Dataset.load_from_dir(
    json_dir='/home/clayton/workspace/prj/data_keep/data/ndds/type1',
    show_pbar=True
)

# Fix NDDS Dataset naming so that it follows convention. (This is not necessary if the NDDS dataset already follows the naming convention.)
for frame in ndds_dataset.frames:
    for ann_obj in frame.ndds_ann.objects:
        if ann_obj.class_name == 'bolt':
            obj_type, obj_name = 'seg', 'bolt-roi'
            ann_obj.class_name = f'{obj_type}_{obj_name}'

# Define COCO Categories
categories = COCO_Category_Handler()
categories.append(
    COCO_Category(
        id=len(categories),
        name='bolt-roi'
    )
)

# Convert To COCO Dataset
dataset = COCO_Dataset.from_ndds(
    ndds_dataset=ndds_dataset,
    categories=categories,
    naming_rule='type_object_instance_contained', delimiter='_',
    ignore_unspecified_categories=True,
    show_pbar=True,
    bbox_area_threshold=1,
    default_visibility_threshold=0.10,
    visibility_threshold_dict={'bolt-roi': 0.01},
    allow_unfound_seg=True,
    class_merge_map={
        'mark1': 'seg_bolt-roi',
        'mark2': 'seg_bolt-roi',
        'mark3': 'seg_bolt-roi'
    }
)

# Save COCO Dataset
dataset.save_to_path('bolt-roi_dataset.json', overwrite=True)

# Preview Dataset
dataset.display_preview(show_details=True)