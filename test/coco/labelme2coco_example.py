from common_utils.file_utils import file_exists
from annotation_utils.coco.structs import COCO_Dataset, COCO_Category_Handler, COCO_Category
from annotation_utils.labelme.structs import LabelmeAnnotationHandler

# Define Labelme Directory Paths
img_dir = '/path/to/labelme/img/dir'
json_dir = '/path/to/labelme/json/dir'

# Load Labelme Handler
labelme_handler = LabelmeAnnotationHandler.load_from_dir(load_dir=json_dir)

# Define COCO Categories Before Conversion
if not file_exists('categories_example.json'): # Save a new categories json if it doesn't already exist.
    categories = COCO_Category_Handler()
    categories.append( # Standard Keypoint Example
        COCO_Category(
            id=len(categories),
            supercategory='pet',
            name='dog',
            keypoints=[ # The keypoint labels are defined here
                'left_eye', 'right_eye', # 0, 1
                'mouth_left', 'mouth_center', 'mouth_right' # 2, 3, 4
            ],
            skeleton=[ # The connections between keypoints are defined with indecies here
                [0, 1],
                [2, 3], [3,4]
            ]
        )
    )
    categories.append( # Simple Keypoint Example
        COCO_Category.from_label_skeleton(
            id=len(categories),
            supercategory='pet',
            name='cat',
            label_skeleton=[
                ['left_eye', 'right_eye'],
                ['mouth_left', 'mouth_center'], ['mouth_center', 'mouth_right']
            ]
        )
    )
    for name in ['duck', 'sparrow', 'pigion']:
        categories.append( # Simple Non-Keypoint Example
            COCO_Category(
                id=len(categories),
                supercategory='bird',
                name=name
            )
        )
    categories.save_to_path('categories_example.json')
else: # Or load from an existing categories json
    categories = COCO_Category_Handler.load_from_path('categories_example.json')

# Convert To COCO
coco_dataset = COCO_Dataset.from_labelme(
    labelme_handler=labelme_handler,
    categories=categories,
    img_dir=img_dir
)
coco_dataset.save_to_path(save_path='converted_coco.json', overwrite=True)
coco_dataset.display_preview(show_details=True) # Optional: Preview your resulting dataset.