from annotation_utils.ndds.structs import NDDS_Dataset

dataset_dir = '/home/clayton/workspace/prj/data_keep/data/ndds/HSR'

ndds_dataset = NDDS_Dataset.load_from_dir(
    json_dir=dataset_dir,
    show_pbar=True
)
assert ndds_dataset == NDDS_Dataset.from_dict(ndds_dataset.to_dict())
ndds_dataset.save_to_dir(
    json_save_dir='test',
    src_img_dir=dataset_dir,
    overwrite=True,
    show_pbar=True
)