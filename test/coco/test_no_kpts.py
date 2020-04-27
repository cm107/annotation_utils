from annotation_utils.coco.structs import COCO_Dataset
from common_utils.common_types.segmentation import Segmentation
from common_utils.common_types.keypoint import Keypoint2D_List

dataset = COCO_Dataset.load_from_path(
    json_path='/home/clayton/workspace/prj/data_keep/data/toyota/dataset/sim/20200228/28_02_2020_11_18_30_coco-data/HSR-coco.json',
    img_dir='/home/clayton/workspace/prj/data_keep/data/toyota/dataset/sim/20200228/28_02_2020_11_18_30_coco-data'
)
# for i, coco_ann in enumerate(dataset.annotations):
#     if i % 2 == 0:
#         coco_ann.segmentation = Segmentation()
#     if i % 3 == 0:
#         coco_ann.keypoints = Keypoint2D_List()
#         coco_ann.num_keypoints = 0
for coco_ann in dataset.annotations:
    coco_ann.segmentation = Segmentation()
    coco_ann.keypoints = Keypoint2D_List()
    coco_ann.num_keypoints = 0
    coco_ann.keypoints_3d = None
    coco_ann.camera = None
for coco_cat in dataset.categories:
    coco_cat.keypoints = []
    coco_cat.skeleton = []
dataset.save_to_path('non_strict_dataset.json', overwrite=True, strict=False)
dataset0 = COCO_Dataset.load_from_path('non_strict_dataset.json', strict=False)
dataset0.images.sort(attr_name='file_name')
dataset0.save_visualization(
    save_dir='test_vis',
    show_preview=True,
    kpt_idx_offset=-1,
    overwrite=True,
    show_details=True
)