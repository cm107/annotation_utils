from logger import logger
from annotation_utils.ndds.structs.frame import NDDS_Frame_Handler

json_dir = '/home/clayton/workspace/prj/data_keep/data/ndds/HSR'
img_dir = json_dir

handler = NDDS_Frame_Handler.load_from_dir(img_dir=img_dir, json_dir=json_dir)
# logger.purple(handler)
handler.save_to_path('handler_save.json', overwrite=True)
handler.save_to_dir(
    json_save_dir='json',
    src_img_dir=img_dir,
    overwrite=True,
    dst_img_dir='img',
    show_pbar=True
)