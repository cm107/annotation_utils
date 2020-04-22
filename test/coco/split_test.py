from annotation_utils.coco.structs import COCO_Dataset

dataset = COCO_Dataset.load_from_path(
    json_path='/home/clayton/workspace/prj/data_keep/data/toyota/dataset/sim/20200228/28_02_2020_11_18_30_coco-data/HSR-coco.json',
    img_dir='/home/clayton/workspace/prj/data_keep/data/toyota/dataset/sim/20200228/28_02_2020_11_18_30_coco-data'
)
dataset.images.sort(attr_name='file_name')
dataset_list = dataset.split(
    dest_dir='split',
    split_dirname_list=['train', 'test', 'val', 'other0', 'other1'],
    ratio=[3, 1, 1, 1, 1],
    coco_filename_list=['a.json', 'b.json', 'c.json', 'd.json', 'e.json'],
    shuffle=True,
    preserve_filenames=False,
    overwrite=True
)