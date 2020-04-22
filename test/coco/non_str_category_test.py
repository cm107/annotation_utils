from logger import logger
from annotation_utils.coco.structs import COCO_Category

coco_cat = COCO_Category(
    id=0,
    supercategory=1.0,
    name=2
)
assert type(coco_cat.supercategory) is str
assert type(coco_cat.name) is str
coco_cat0 = COCO_Category.from_dict(coco_cat.to_dict())
assert type(coco_cat0.supercategory) is str
assert type(coco_cat0.name) is str

coco_cat = COCO_Category(
    id=0,
    supercategory='a',
    name='b'
)
assert type(coco_cat.supercategory) is str
assert type(coco_cat.name) is str
coco_cat0 = COCO_Category.from_dict(coco_cat.to_dict())
assert type(coco_cat0.supercategory) is str
assert type(coco_cat0.name) is str

logger.good(f'Test passed')