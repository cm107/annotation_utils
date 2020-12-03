import cv2
from annotation_utils.ndds.structs import NDDS_Dataset
from common_utils.file_utils import make_dir_if_not_exists, delete_all_files_in_dir
from common_utils.cv_drawing_utils import draw_bbox
from streamer.cv_viewer import cv_simple_image_viewer

# Load NDDS Dataset
ndds_dataset = NDDS_Dataset.load_from_dir(
    json_dir='/home/clayton/workspace/prj/data_keep/data/ndds/m1_200',
    show_pbar=True
)

vis_dump_dir = 'vis_dump'
make_dir_if_not_exists(vis_dump_dir)
delete_all_files_in_dir(vis_dump_dir)

for frame in ndds_dataset.frames:
    img = cv2.imread(frame.img_path)
    for ndds_obj in frame.ndds_ann.objects:
        img = draw_bbox(img=img, bbox=ndds_obj.bounding_box)
    quit_flag = cv_simple_image_viewer(img=img, preview_width=1000)
    if quit_flag:
        break