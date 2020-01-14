import matplotlib.pyplot as plt
import matplotlib.image as mpimage
import matplotlib.patches as patches
from pycocotools.coco import COCO
from logger import logger

class COCOAnnotationVisualizer:
    def __init__(self, img_dir: str, coco_annotation_path: str, visualization_dump_dir: str, included_categories: list):
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
        img_filename = f"{rootname}.png"
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
        coco = COCO(annotation_file=self.coco_annotation_path)
        catIds, annIds, imgs = self.get_data(coco)

        for i, img in zip(range(len(imgs)), imgs):
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