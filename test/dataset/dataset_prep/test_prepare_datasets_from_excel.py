from annotation_utils.dataset.dataset_prep import prepare_datasets_from_excel

prepare_datasets_from_excel(
    xlsx_path='/home/clayton/workspace/prj/data_keep/data/dataset/bird/datasets.xlsx',
    dst_root_dir='/home/clayton/workspace/prj/data_keep/data/dataset/bird/combined_datasets1',
    usecols='A:D', skiprows=None, skipfooter=0, skip_existing=False,
    val_target_proportion=0.05, min_val_size=1, max_val_size=20,
    orig_config_save='/home/clayton/workspace/prj/data_keep/data/dataset/bird/orig_config1.yaml',
    reorganized_config_save='/home/clayton/workspace/prj/data_keep/data/dataset/bird/reorganized_config1.yaml',
    show_pbar=True
)