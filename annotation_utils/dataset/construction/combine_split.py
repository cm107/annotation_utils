from logger import logger

from common_utils.check_utils import check_value_from_list
from common_utils.file_utils import delete_file
from ...coco.combiner import COCO_Annotations_Combiner
from ...coco.splitter import COCO_Splitter
from ..config import DatasetPathConfig

class DatasetCombineSplitWrapper:
    def __init__(
        self,
        dataset_path_config: str, src_root_dir: str, output_dir: str,
        ratio: list=[4, 0, 1], ann_format: str='coco'
    ):
        self.dataset_path_config = dataset_path_config
        self.src_root_dir = src_root_dir
        self.output_dir = output_dir
        self.ratio = ratio
        self.ann_format = ann_format
        self.combined_ann_output_path = f"/tmp/combined_ann_output.json"

    def run(self):
        # Load Dataset Paths From Config
        dataset_path_config = DatasetPathConfig.from_load(target=self.dataset_path_config)
        dataset_dir_list, img_dir_list, ann_path_list, ann_format_list = dataset_path_config.get_paths()
        check_value_from_list(item_list=ann_format_list, valid_value_list=[self.ann_format])

        # Combine Annotations
        worker = COCO_Annotations_Combiner(
            img_dir_list=img_dir_list,
            ann_path_list=ann_path_list
        )
        worker.load_combined(verbose=False, detailed_verbose=False)
        worker.write_combined(output_path=self.combined_ann_output_path, verbose=True)
        logger.purple(f"len(worker.licenses.license_list): {len(worker.buffer.licenses.license_list)}")
        logger.purple(f"len(worker.images.image_list): {len(worker.buffer.images.image_list)}")
        logger.purple(f"len(worker.annotations.annotation_list): {len(worker.buffer.annotations.annotation_list)}")
        logger.purple(f"len(worker.categories.category_list): {len(worker.buffer.categories.category_list)}")

        # Split Annotations In New Directory
        splitter = COCO_Splitter(
            ann_path=self.combined_ann_output_path,
            src_root_dir=self.src_root_dir,
            dest_dir=self.output_dir,
            ratio=self.ratio
        )
        delete_file(self.combined_ann_output_path)
        splitter.load_split(preserve_filenames=False, verbose=False)
        splitter.write_split(verbose=True)