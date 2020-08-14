from annotation_utils.coco.structs import COCO_Dataset

dataset0 = COCO_Dataset.load_from_path(
    '/home/clayton/workspace/prj/data_keep/data/dataset/traffic_light/split/val0/coco/output.json',
    img_dir='/home/clayton/workspace/prj/data_keep/data/dataset/traffic_light/split/val0/img'
)
dataset1 = COCO_Dataset.load_from_path(
    '/home/clayton/workspace/prj/data_keep/data/dataset/traffic_light/split/val1/coco/output.json',
    img_dir='/home/clayton/workspace/prj/data_keep/data/dataset/traffic_light/split/val1/img'
)

combined_dataset = COCO_Dataset.combine([dataset0, dataset1])
combined_dataset.save_to_path('combine_test.json', overwrite=True)