from logger import logger
from annotation_utils.coco.structs import COCO_License_Handler, COCO_License

license0 = COCO_License(url='url_a', id=0, name='license_a')
license1 = COCO_License(url='url_b', id=1, name='license_b')
license2 = COCO_License(url='url_c', id=2, name='license_c')
license_handler = COCO_License_Handler([license0, license1, license2])

license_handler.append(COCO_License(url='url_d', id=3, name='license_d'))
logger.purple(license_handler.license_list)
license_handler0 = license_handler.copy()
del license_handler0[1]
license_handler0[1] = COCO_License(url='url_x', id=99, name='license_x')
for coco_license in license_handler0:
    logger.cyan(coco_license)
logger.blue(len(license_handler0))
license_handler0.sort(attr_name='name')
for coco_license in license_handler0:
    logger.cyan(coco_license)

logger.info('Shuffle')
license_handler0.shuffle()
for coco_license in license_handler0:
    logger.cyan(coco_license)

coco_license = license_handler0.get_obj_from_id(3)
logger.purple(f'coco_license: {coco_license}')

logger.purple(f'license_handler0.to_dict_list():\n{license_handler0.to_dict_list()}')
license_handler0.save_to_path('license_handler.json', overwrite=True)