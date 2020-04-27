from common_utils.file_utils import make_dir_if_not_exists, delete_all_files_in_dir
from annotation_utils.coco.structs import COCO_Dataset

dataset = COCO_Dataset.load_from_path(json_path='output.json')
# dataset.display_preview()
dump_dir = 'ann_dump'
make_dir_if_not_exists(dump_dir)
delete_all_files_in_dir(dump_dir, ask_permission=False)

dataset.info.save_to_path(f'{dump_dir}/info.json')
dataset.images.save_to_path(f'{dump_dir}/images.json')
dataset.annotations.save_to_path(f'{dump_dir}/annotations.json')
dataset.categories.save_to_path(f'{dump_dir}/categories.json')
dataset.categories.save_to_path('box_hsr_categories.json')