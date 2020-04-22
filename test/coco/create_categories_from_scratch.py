from annotation_utils.coco.structs import COCO_Category_Handler, COCO_Category

handler = COCO_Category_Handler()
handler.append(
    COCO_Category(
        supercategory='screw',
        name='screw',
        keypoints=[],
        skeleton=[],
        id=len(handler)
    )
)
handler.append(
    COCO_Category(
        supercategory='hole',
        name='hole',
        keypoints=[],
        skeleton=[],
        id=len(handler)
    )
)
handler.save_to_path('interphone_ng_object_categories.json')