from annotation_utils.coco.structs import COCO_Category_Handler, COCO_Category

# Simple Non-Keypoint Example
handler = COCO_Category_Handler(
    [
        COCO_Category(
            id=0,
            supercategory='bird',
            name='duck'
        ),
        COCO_Category(
            id=1,
            supercategory='bird',
            name='sparrow'
        ),
        COCO_Category(
            id=1,
            supercategory='bird',
            name='pigeon'
        )
    ]
)
handler.save_to_path('birds0.json')

# Simple Non-Keypoint Example using for loop
handler = COCO_Category_Handler()

for name in ['duck', 'sparrow', 'pigeon']:
    handler.append(
        COCO_Category(
            id=len(handler),
            supercategory='bird',
            name=name
        )
    )
handler.save_to_path('birds1.json')

# Keypoint Example
handler = COCO_Category_Handler(
    [
        COCO_Category(
            id=0,
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
    ]
)
handler.save_to_path('dog0.json')

# Simple Keypoint Example
handler = COCO_Category_Handler(
    [
        COCO_Category.from_label_skeleton(
            id=0,
            supercategory='pet',
            name='dog',
            label_skeleton=[
                ['left_eye', 'right_eye'],
                ['mouth_left', 'mouth_center'], ['mouth_center', 'mouth_right']
            ]
        )
    ]
)
handler.save_to_path('dog1.json')