import matplotlib.pyplot as plt
import matplotlib.image as mpimage
from pycocotools.coco import COCO
from ..logger.logger_handler import logger

class COCOAnnotationVisualizer:
    def __init__(self, img_dir: str, coco_annotation_path: str, visualization_dump_dir: str, included_categories: list):
        self.img_dir = img_dir
        self.coco_annotation_path = coco_annotation_path
        self.visualization_dump_dir = visualization_dump_dir
        self.included_categories = included_categories

    def save(self, coco: COCO, img: dict, anns: list):
        I = mpimage.imread(f"{self.img_dir}/{img['file_name']}")
        plt.axis('off')
        plt.imshow(I)
        ax = plt.gca()
        ax.set_axis_off()
        ax.margins(0,0)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
        coco.showAnns(anns)
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

    def generate_visualizations(self):
        coco = COCO(annotation_file=self.coco_annotation_path)
        catIds, annIds, imgs = self.get_data(coco)

        for img in imgs:
            annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
            anns = coco.loadAnns(annIds)

            has_keypoints = False
            for ann in anns:
                if ann['num_keypoints'] > 0:
                    has_keypoints = True
                    break
            if not has_keypoints:
                continue

            self.save(coco, img, anns)