from logger import logger
from annotation_utils.coco.structs import COCO_Dataset

dataset = COCO_Dataset.load_from_path(
    json_path='bk_28_02_2020_11_18_30_coco-data/HSR-coco.json',
    img_dir='bk_28_02_2020_11_18_30_coco-data'
)
logger.purple(f'Flag0 len(dataset.images): {len(dataset.images)}, len(dataset.annotations): {len(dataset.annotations)}')
dataset.prune_keypoints(min_num_kpts=11, verbose=True)
logger.purple(f'Flag1 len(dataset.images): {len(dataset.images)}, len(dataset.annotations): {len(dataset.annotations)}')
dataset.move_images(dst_img_dir='test_img', preserve_filenames=True, update_img_paths=True, overwrite=True, show_pbar=True)
logger.purple(f'Flag2 len(dataset.images): {len(dataset.images)}, len(dataset.annotations): {len(dataset.annotations)}')
dataset.save_to_path(save_path='prune_test.json', overwrite=True)
dataset.display_preview(kpt_idx_offset=-1)