import cv2
import numpy as np
import operator
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
import matplotlib.patches as patches
from pycocotools.coco import COCO
from logger import logger
from common_utils.file_utils import delete_dir_if_exists, make_dir, file_exists
from common_utils.path_utils import get_rootname_from_filename, get_extension_from_filename
from common_utils.cv_drawing_utils import draw_bbox, draw_segmentation, draw_skeleton, draw_keypoints
from common_utils.common_types.bbox import BBox
from common_utils.common_types.segmentation import Segmentation

from ..coco_annotation import COCO_AnnotationFileParser
from ...structs import COCO_Image, COCO_Annotation

class COCO_API_COCOAnnotationVisualizer:
    """
    Obsolete. Please use COCOAnnotationVisualizer instead.
    """
    def __init__(
        self, img_dir: str, coco_annotation_path: str, visualization_dump_dir: str, included_categories: list
    ):
        self.img_dir = img_dir
        self.coco_annotation_path = coco_annotation_path
        self.visualization_dump_dir = visualization_dump_dir
        self.included_categories = included_categories

    def save(self, coco: COCO, img: dict, anns: list, show_bbox: bool=False, filename_key: str='file_name'):
        I = mpimage.imread(f"{self.img_dir}/{img[filename_key]}")
        plt.axis('off')
        plt.imshow(I)
        ax = plt.gca()
        ax.set_axis_off()
        ax.margins(0,0)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
        coco.showAnns(anns)

        if show_bbox:
            for ann in anns:
                xmin, ymin, width, height = ann['bbox']
                # Create a Rectangle patch
                rect = patches.Rectangle((xmin, ymin), width, height, linewidth=1, edgecolor='yellow', facecolor='none')
                # Add the patch to the Axes
                ax.add_patch(rect)

        rootname = str(img['id'])
        while len(rootname) < 6:
            rootname = '0' + rootname
        img_filename = f"{rootname}.jpg"
        plt.savefig(f'{self.visualization_dump_dir}/{img_filename}', bbox_inches='tight', pad_inches=0)
        logger.info(f'Created {self.visualization_dump_dir}/{img_filename}')
        plt.clf()

    def get_data(self, coco: COCO):
        catIds = coco.getCatIds(catNms=[self.included_categories])
        imgIds = coco.getImgIds(catIds=catIds)
        annIds = coco.getAnnIds()
        imgs = coco.loadImgs(imgIds)
        return catIds, annIds, imgs

    def generate_visualizations(self, show_bbox: bool=False, filename_key: str='file_name', limit: int=None):
        delete_dir_if_exists(self.visualization_dump_dir)
        make_dir(self.visualization_dump_dir)
        coco = COCO(annotation_file=self.coco_annotation_path)
        catIds, annIds, imgs = self.get_data(coco)

        for i, img in enumerate(imgs):
            if limit is not None and i >= limit:
                break
            annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
            anns = coco.loadAnns(annIds)

            has_keypoints = False
            for ann in anns:
                if ann['num_keypoints'] > 0:
                    has_keypoints = True
                    break
            if not has_keypoints and not show_bbox:
                continue

            self.save(coco, img, anns, show_bbox=show_bbox, filename_key=filename_key)

class COCOAnnotationVisualizer:
    """
    Visualize COCO Annotations given the annotation json path and corresponding image directory.
    """
    def __init__(
        self, img_dir: str, coco_annotation_path: str, visualization_dump_dir: str, included_categories: list,
        bbox_color: list=[0, 255, 255], kpt_color: list=[0, 0, 255], skeleton_color: list=[255, 0, 0], seg_color: list=[255, 255, 0],
        transparent_seg: bool=True, kpt_radius: int=10, bbox_thickness: int=2, skeleton_thickness: int=2, show_cat_name: bool=False,
        skeleton_index_offset: int=0,
        show_bbox: bool=True, show_kpts: bool=True, show_skeleton: bool=True, show_seg: bool=True,
        show_bbox_labels: bool=False, show_kpt_labels: bool=False, bbox_label_thickness: int=None, kpt_label_thickness: int=None,
        show_bbox_labels_only: bool=False, show_kpt_labels_only: bool=False,
        render_order: list=['bbox', 'seg', 'skeleton', 'kpt'],
        viz_limit: int=None
    ):
        self.img_dir = img_dir
        self.coco_annotation_path = coco_annotation_path
        self.visualization_dump_dir = visualization_dump_dir
        self.included_categories = included_categories

        self.parser = COCO_AnnotationFileParser(coco_annotation_path)
        self.parser.load()

        self.bbox_color = bbox_color
        self.kpt_color = kpt_color
        self.skeleton_color = skeleton_color
        self.seg_color = seg_color

        self.transparent_seg = transparent_seg
        self.kpt_radius = kpt_radius
        self.bbox_thickness = bbox_thickness
        self.skeleton_thickness = skeleton_thickness
        self.show_cat_name = show_cat_name
        self.skeleton_index_offset = skeleton_index_offset

        self.show_bbox = show_bbox
        self.show_kpts = show_kpts
        self.show_skeleton = show_skeleton
        self.show_seg = show_seg

        self.show_bbox_labels = show_bbox_labels
        self.show_kpt_labels = show_kpt_labels
        self.bbox_label_thickness = bbox_label_thickness
        self.kpt_label_thickness = kpt_label_thickness
        self.show_bbox_labels_only = show_bbox_labels_only
        self.show_kpt_labels_only = show_kpt_labels_only

        self.render_order = render_order

        self.viz_limit = viz_limit
        self.viz_count = 0

    def _draw(self, img: np.ndarray, bbox: BBox, kpt_list: list, kpt_skeleton: list, kpt_label_list: list, seg: Segmentation, cat_name: str) -> np.ndarray:
        for render_target in self.render_order:
            if render_target.lower() in ['seg', 'segmentation']:
                if self.show_seg:
                    img = draw_segmentation(img=img, segmentation=seg, color=self.seg_color, transparent=self.transparent_seg)
            elif render_target.lower() in ['skel', 'skeleton']:
                if self.show_skeleton:
                    img = draw_skeleton(
                        img=img, keypoints=kpt_list, keypoint_skeleton=kpt_skeleton, index_offset=self.skeleton_index_offset,
                        thickness=self.skeleton_thickness, color=self.skeleton_color
                    )
            elif render_target.lower() in ['kpt', 'kpts', 'keypoint', 'keypoints']:
                if self.show_kpts:
                    if self.kpt_label_thickness is not None:
                        kpt_label_thickness = self.kpt_label_thickness
                    else:
                        kpt_label_thickness = self.kpt_radius - 1 if self.kpt_radius > 1 else 1
                    img = draw_keypoints(
                        img=img, keypoints=kpt_list, radius=self.kpt_radius, color=self.kpt_color, keypoint_labels=kpt_label_list,
                        show_keypoints_labels=self.show_kpt_labels, label_thickness=kpt_label_thickness, label_only=self.show_kpt_labels_only
                    )
            elif render_target.lower() in ['bbox', 'bounding_box', 'bounding box']:
                if self.show_bbox:
                    bbox_text = cat_name if self.show_bbox_labels or self.show_bbox_labels_only else None
                    bbox_label_thickness = self.bbox_label_thickness if self.bbox_label_thickness is not None else self.bbox_thickness
                    img = draw_bbox(
                        img=img, bbox=bbox, color=self.bbox_color, thickness=self.bbox_thickness,
                        text=bbox_text, label_thickness=bbox_label_thickness, label_only=self.show_bbox_labels_only
                    )
            else:
                logger.error(f"Invalid render_target: {render_target}")
                logger.error(f"Options: seg, skeleton, kpt, bbox")
                raise Exception
        return img

    def save(
        self, img: np.ndarray,
        bbox_list: list,
        kpt_list_list: list, kpt_skeleton_list: list, kpt_label_list_list: list,
        seg_list: list, cat_name_list: list,
        file_name: str
    ):
        result = img.copy()
        save_path = f"{self.visualization_dump_dir}/{file_name}"
        retry_count = 0
        while file_exists(save_path):
            rootname = get_rootname_from_filename(file_name)
            extension = get_extension_from_filename(file_name)
            save_path = f"{self.visualization_dump_dir}/{rootname}_{retry_count}.{extension}"
            if retry_count == 9:
                logger.error(f"Can't resolve save_path.")
                raise Exception
            retry_count += 1

        for bbox, kpt_list, kpt_skeleton, kpt_label_list, seg, cat_name in \
            zip(bbox_list, kpt_list_list, kpt_skeleton_list, kpt_label_list_list, seg_list, cat_name_list):
            result = self._draw(
                img=result,
                bbox=bbox, kpt_list=kpt_list,
                kpt_skeleton=kpt_skeleton, kpt_label_list=kpt_label_list,
                seg=seg, cat_name=cat_name
            )
        cv2.imwrite(filename=save_path, img=result)
        logger.info(f"Wrote {save_path}")
        self.viz_count += 1

    def generate_visualizations(self, do_sort: bool=False):
        delete_dir_if_exists(self.visualization_dump_dir)
        make_dir(self.visualization_dump_dir)

        if do_sort:
            self.parser.images.image_list.sort(key=operator.attrgetter('file_name'))

        for coco_image in self.parser.images.image_list:
            coco_image = COCO_Image.buffer(coco_image)
            img_path = f"{self.img_dir}/{coco_image.file_name}"
            img = cv2.imread(img_path)
            coco_annotation_list = self.parser.annotations.get_annotations_from_imgIds(imgIds=[coco_image.id])
            bbox_list = []
            kpt_list_list = []
            kpt_skeleton_list = []
            kpt_label_list_list = []
            seg_list = []
            cat_name_list = []
            for coco_annotation in coco_annotation_list:
                coco_annotation = COCO_Annotation.buffer(coco_annotation)
                bbox = coco_annotation.bounding_box
                keypoint_list = coco_annotation.get_keypoint_list()
                kpt_list = [[kpt.x, kpt.y] for kpt in keypoint_list]
                seg = Segmentation.from_list(points_list=coco_annotation.segmentation)
                bbox_list.append(bbox)
                kpt_list_list.append(kpt_list)
                seg_list.append(seg)
                coco_category = self.parser.categories.get_category_from_id(id=coco_annotation.category_id)
                cat_name_list.append(coco_category.name)
                kpt_skeleton_list.append(coco_category.skeleton)
                kpt_label_list_list.append(coco_category.keypoints)
            self.save(
                img=img,
                bbox_list=bbox_list,
                kpt_list_list=kpt_list_list,
                kpt_skeleton_list=kpt_skeleton_list,
                kpt_label_list_list=kpt_label_list_list,
                seg_list=seg_list,
                cat_name_list=cat_name_list,
                file_name=coco_image.file_name
            )
            if self.viz_limit is not None and self.viz_count >= self.viz_limit:
                break