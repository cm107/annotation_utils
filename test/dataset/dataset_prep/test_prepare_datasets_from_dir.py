from annotation_utils.dataset.dataset_prep import prepare_datasets_from_dir

prepare_datasets_from_dir(
    scenario_root_dir='/home/clayton/workspace/prj/data_keep/data/dataset/bird/dataset_root_dir',
    dst_root_dir='/home/clayton/workspace/prj/data_keep/data/dataset/bird/combined_datasets',
    annotation_filename='output.json',
    skip_existing=True,
    val_target_proportion=0.05,
    min_val_size=1, max_val_size=20,
    orig_config_save='/home/clayton/workspace/prj/data_keep/data/dataset/bird/orig_config.json',
    reorganized_config_save='/home/clayton/workspace/prj/data_keep/data/dataset/bird/reorganized_config.json'
)