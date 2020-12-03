from __future__ import annotations
import cv2
import numpy as np
from typing import cast, Tuple

from common_utils.common_types.bbox import BBox
from common_utils.common_types.point import Point2D
from common_utils.base.basic import BasicLoadableObject

from .dataset import COCO_Dataset
from .objects import COCO_Image, COCO_Annotation

class COCO_Zoom(BasicLoadableObject['COCO_Zoom']):
    def __init__(self, magnitude: float=1.0, center: Point2D=None):
        super().__init__()
        self._magnitude = magnitude
        self._center = center

        # Image Buffers
        self._src_img = cast(np.ndarray, None)
        self._zoom_img = cast(np.ndarray, None)

        # COCO Exclusive
        self._src_coco_img = cast(COCO_Image, None)
        self._src_coco_ann = cast(COCO_Annotation, None)
        self._zoom_coco_ann = cast(COCO_Annotation, None)
    
    def to_dict(self) -> dict:
        result = {
            'magnitude': self.magnitude
        }
        if self.center is not None:
            result['center'] = self.center.to_list()
        return result
    
    @classmethod
    def from_dict(cls, item_dict: dict) -> COCO_Zoom:
        return COCO_Zoom(
            magnitude=item_dict['magnitude'],
            center=Point2D.from_list(item_dict['center']) if 'center' in item_dict else None
        )

    @property
    def magnitude(self) -> float:
        return self._magnitude
    
    @magnitude.setter
    def magnitude(self, magnitude: float):
        self._magnitude = magnitude
        self.update_zoom()

    @property
    def center(self) -> Point2D:
        return self._center
    
    def update_center(self, center: Point2D):
        self._center = center

    @center.setter
    def center(self, center: Point2D):
        self.update_center(center=center)
        self.update_zoom()

    def center_to_cener_of(self, img: np.ndarray):
        img_h, img_w = img.shape[:2]
        self.update_center(Point2D(x=int(img_w / 2), y=int(img_h / 2)))

    def adjust_center_to_src_img(self):
        h, w = self.src_img.shape[:2]
        if self.center is None:
            self.center_to_cener_of(self.src_img)
        cx, cy = self.center.to_list()
        cx = cx if cx < w else w - 1
        cy = cy if cy < h else h - 1
        self.update_center(Point2D(x=cx, y=cy))

    @property
    def src_img(self) -> np.ndarray:
        return self._src_img

    def update_src_img(self, src_img: np.ndarray):
        self._src_img = src_img

    @src_img.setter
    def src_img(self, src_img: np.ndarray):
        self.update_src_img(src_img=src_img)
        self.update_src_coco_ann(None)
        self.adjust_center_to_src_img()
        self.update_zoom()

    @property
    def src_coco_img(self) -> COCO_Image:
        return self._src_coco_img
    
    def update_src_coco_img(self, src_coco_img: COCO_Image):
        self._src_coco_img = src_coco_img

    @src_coco_img.setter
    def src_coco_img(self, src_coco_img: COCO_Image):
        self.update_src_coco_img(src_coco_img=src_coco_img)
        self.update_src_img(src_img=cv2.imread(self.src_coco_img.coco_url))
        self.update_center(center=Point2D(x=int(src_coco_img.width / 2), y=int(src_coco_img.height / 2)))
        self.adjust_center_to_src_img()

    @property
    def src_coco_ann(self) -> COCO_Annotation:
        return self._src_coco_ann
    
    def update_src_coco_ann(self, src_coco_ann: COCO_Annotation):
        self._src_coco_ann = src_coco_ann

    @src_coco_ann.setter
    def src_coco_ann(self, src_coco_ann: COCO_Annotation):
        self.update_src_coco_ann(src_coco_ann=src_coco_ann)

    def check_ids_are_valid(self):
        if self.src_coco_img is not None and self.src_coco_ann is not None:
            assert self.src_coco_ann.image_id == self.src_coco_img.id

    def update_src_coco(self, src_coco_img: COCO_Image, src_coco_ann: COCO_Annotation):
        self.update_src_coco_img(src_coco_img=src_coco_img)
        self.update_src_img(src_img=cv2.imread(self.src_coco_img.coco_url))
        self.update_src_coco_ann(src_coco_ann=src_coco_ann)
        self.update_center(center=Point2D.from_list(src_coco_ann.bbox.center()))
        self.adjust_center_to_src_img()
        self.check_ids_are_valid()
        self.update_zoom()

    @property
    def zoom_img(self) -> np.ndarray:
        return self._zoom_img

    @property
    def zoom_coco_ann(self) -> COCO_Annotation:
        return self._zoom_coco_ann

    def resize(self, dsize: Tuple[int]):
        target_w, target_h = dsize
        h, w = self.zoom_img.shape[:2]
        self._zoom_img = cv2.resize(self._zoom_img, dsize=dsize)
        self._zoom_coco_ann.bbox = self._zoom_coco_ann.bbox.resize(orig_frame_shape=[h, w], new_frame_shape=[target_h, target_w])
        self._zoom_coco_ann.segmentation = self._zoom_coco_ann.segmentation.resize(
            orig_frame_shape=[h, w],
            new_frame_shape=[target_h, target_w]
        )
        self._zoom_coco_ann.keypoints = self._zoom_coco_ann.keypoints.resize(orig_frame_shape=[h, w], new_frame_shape=[target_h, target_w])        

    def _adjust_for_zoom(self, target_zoom_c: int, target_crop_length: int, image_side_length: int) -> (int, int ,int):
        if target_zoom_c - int(target_crop_length / 2) < 0:
            zoom_c = int(target_crop_length / 2)
            crop_lower = 0
            crop_upper = crop_lower + target_crop_length - 1
        elif target_zoom_c + int(target_crop_length / 2) >= image_side_length:
            zoom_c = image_side_length - int(target_crop_length / 2) - 1
            crop_upper = image_side_length - 1
            crop_lower = crop_upper - target_crop_length + 1
        else:
            zoom_c = target_zoom_c
            crop_lower = zoom_c - int(target_crop_length / 2)
            crop_upper = crop_lower + target_crop_length
        return zoom_c, crop_lower, crop_upper
    
    def adjust_for_zoom(self) -> (np.ndarray, COCO_Annotation):
        assert self.src_img is not None and self.magnitude is not None
        if self.center is None:
            self.center_to_cener_of(img=self.src_img)
        img_h, img_w = self.src_img.shape[:2]
        target_crop_h, target_crop_w = int(img_h / self.magnitude), int(img_w / self.magnitude)
        zoom_cx, crop_xmin, crop_xmax = self._adjust_for_zoom(target_zoom_c=self.center.x, target_crop_length=target_crop_w, image_side_length=img_w)
        zoom_cy, crop_ymin, crop_ymax = self._adjust_for_zoom(target_zoom_c=self.center.y, target_crop_length=target_crop_h, image_side_length=img_h)
        crop_bbox = BBox(xmin=crop_xmin, ymin=crop_ymin, xmax=crop_xmax, ymax=crop_ymax)

        cropped_img = crop_bbox.crop_from(img=self.src_img)
        if self.src_coco_ann is not None:
            new_coco_ann = self.src_coco_ann.copy()
            new_coco_ann.bbox = self.src_coco_ann.bbox - crop_bbox.pmin
            new_coco_ann.segmentation = self.src_coco_ann.segmentation - crop_bbox.pmin
            new_coco_ann.keypoints = self.src_coco_ann.keypoints - crop_bbox.pmin
            return cropped_img, new_coco_ann
        else:
            return cropped_img, None

    def update_zoom(self, show_seg: bool=False, show_bbox: bool=True, show_kpt: bool=True):
        if self.src_img is not None:
            cropped_img, new_coco_ann = self.adjust_for_zoom()
            if new_coco_ann is not None:
                cropped_img = COCO_Dataset.draw_ann(img=cropped_img, coco_ann=new_coco_ann, show_seg=show_seg, show_bbox=show_bbox, show_kpt=show_kpt)
            self._zoom_img = cropped_img
            self._zoom_coco_ann = new_coco_ann
        else:
            self._zoom_img = None
            self._zoom_coco_ann = None