from annotation_utils.coco.structs import COCO_Dataset

dataset = COCO_Dataset.load_from_path(
    json_path='/home/clayton/Downloads/screw1_mark-100-_coco-data/screw-coco.json',
    img_dir='/home/clayton/Downloads/screw1_mark-100-_coco-data',
    strict=False
)
screw_dataset, mark_dataset = dataset.copy(), dataset.copy()
screw_dataset.remove_categories_by_name(category_names=['mark'])
mark_dataset.remove_categories_by_name(category_names=['screw'])
screw_dataset.save_to_path(save_path='screw.json', overwrite=True)
mark_dataset.save_to_path(save_path='mark.json', overwrite=True)

screw_dataset.display_preview(
    show_details=True,
    show_seg=False
)
mark_dataset.display_preview(
    show_details=True,
)