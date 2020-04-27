from annotation_utils.coco.structs import COCO_Dataset

# dataset0 = COCO_Dataset.load_from_path(
#     json_path='/home/clayton/workspace/prj/data_keep/data/toyota/dataset/sim/20200228/28_02_2020_11_18_30_coco-data/HSR-coco.json',
#     img_dir='/home/clayton/workspace/prj/data_keep/data/toyota/dataset/sim/20200228/28_02_2020_11_18_30_coco-data',
#     check_paths=True
# )
# dataset1 = COCO_Dataset.load_from_path(
#     json_path='/home/clayton/workspace/prj/data_keep/data/toyota/dataset/sim/20200122/coco-data/new_HSR-coco.json',
#     img_dir='/home/clayton/workspace/prj/data_keep/data/toyota/dataset/sim/20200122/coco-data',
#     check_paths=True
# )
# combined_dataset = COCO_Dataset.combine([dataset0, dataset1])
combined_dataset = COCO_Dataset.combine_from_config(
    config_path='/home/clayton/workspace/prj/data_keep/data/toyota/dataset/config/json/box_hsr_kpt_train.json',
    img_sort_attr_name='file_name',
    show_pbar=True
)
combined_dataset.move_images(
    dst_img_dir='combined_img',
    preserve_filenames=False,
    update_img_paths=True,
    overwrite=True,
    show_pbar=True
)
combined_dataset.save_to_path(save_path='combined.json', overwrite=True)
combined_dataset.display_preview(kpt_idx_offset=-1, start_idx=0)