from common_utils.iter_utils import ImageIterator
from ..structs.handlers import COCO_Image_Handler

class CocoImageIterator(ImageIterator):
    def __init__(
        self, coco_images: COCO_Image_Handler, check_paths: bool=True,
        show_pbar: bool=False, leave_pbar: bool=False
    ):
        if not isinstance(coco_images, COCO_Image_Handler):
            raise TypeError(f'Invalid type for images: {type(coco_images).__name__}. Expected COCO_Image_Handler.')
        self.coco_images = coco_images
        img_paths = [coco_image.coco_url for coco_image in self.coco_images]
        super().__init__(
            img_paths=img_paths,
            check_paths=check_paths,
            show_pbar=show_pbar, leave_pbar=leave_pbar,
            pbar_desc_mode='filename'
        )
    
    @property
    def next_img_path(self) -> str:
        return self.coco_images[self.n].coco_url

    @property
    def next_img_filename(self) -> str:
        return self.coco_images[self.n].file_name

    @property
    def current_img_path(self) -> str:
        return self.coco_images[self.n-1].coco_url
    
    @property
    def current_img_filename(self) -> str:
        return self.coco_images[self.n-1].file_name

    @classmethod
    def from_dir(self, img_dir: str):
        raise Exception('Not applicable to CocoImageIterator')