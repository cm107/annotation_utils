from annotation_utils.coco.structs import COCO_Dataset

dataset = COCO_Dataset.load_from_path(
    json_path='/home/clayton/workspace/prj/data_keep/data/sekisui/hook/coco/output.json',
    img_dir='/home/clayton/workspace/prj/data_keep/data/sekisui/hook/img'
)

for coco_image in dataset.images:
    anns = dataset.annotations.get_annotations_from_imgIds([coco_image.id])
    for ann in anns:
        ann.bbox = ann.bbox.scale_about_center(
            scale_factor=1.3,
            frame_shape=[coco_image.height, coco_image.width]
        )

dataset.save_to_path('bbox_resized.json', overwrite=True)
dataset.display_preview(show_details=True, kpt_idx_offset=-1)