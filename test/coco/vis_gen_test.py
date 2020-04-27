from annotation_utils.coco.structs import COCO_Dataset

dataset = COCO_Dataset.load_from_path(
    json_path='/home/clayton/workspace/prj/data_keep/data/toyota/dataset/sim/20200228/28_02_2020_11_18_30_coco-data/HSR-coco.json',
    img_dir='/home/clayton/workspace/prj/data_keep/data/toyota/dataset/sim/20200228/28_02_2020_11_18_30_coco-data'
)
dataset.images.sort(attr_name='file_name')
dataset.save_visualization(
    save_dir='test_vis',
    show_preview=True,
    kpt_idx_offset=-1,
    overwrite=True
)