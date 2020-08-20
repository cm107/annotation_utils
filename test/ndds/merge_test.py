from annotation_utils.ndds.structs import NDDS_Dataset

dataset = NDDS_Dataset.load_from_dir('/home/clayton/workspace/prj/data_keep/data/ndds/measure_kume_map3_1_200', show_pbar=True)
dataset._check_valid_merge_map(
    merge_map={
        'marking_bottom': 'measure',
        'marking_top': 'measure',
        'hook': 'measure',
        'mark_10th_place': 'measure'
    }
)
print('Success')