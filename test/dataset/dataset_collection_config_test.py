from logger import logger
from annotation_utils.dataset.config import DatasetConfigCollectionHandler

handler = DatasetConfigCollectionHandler.load_from_path('/home/clayton/workspace/prj/data_keep/data/toyota/dataset/config/json/box_hsr_kpt_train.json')
logger.purple(handler.to_dict_list())
for collection in handler:
    for config in collection:
        logger.blue(config)
handler.save_to_path('test.yaml', overwrite=True)
handler0 = DatasetConfigCollectionHandler.load_from_path('test.yaml')
handler0.save_to_path('test0.yaml', overwrite=True)

fp, fp0 = open('test.yaml'), open('test0.yaml')
line, line0 = fp.readline(), fp0.readline()
for i, [line, line0] in enumerate(zip(fp, fp0)):
    logger.white(f'{i}: {line.strip()}')
    logger.white(f'{i}: {line0.strip()}')
    assert line.strip() == line0.strip()