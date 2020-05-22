from annotation_utils.ndds.structs import NDDS_Dataset

dataset = NDDS_Dataset.load_from_dir('/home/clayton/workspace/prj/data_keep/data/ndds/measure_kume_map3_1_200', show_pbar=True)
dataset.save_to_dir('save_test', show_pbar=True)