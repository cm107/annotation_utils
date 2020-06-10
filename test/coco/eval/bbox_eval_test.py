import json
from annotation_utils.coco.eval.bbox import BBoxResult, BBoxResultHandler
from annotation_utils.coco.eval.keypoint import KeypointResult, KeypointResultHandler
from common_utils.common_types.bbox import BBox
from common_utils.common_types.keypoint import Keypoint2D_List

bbox_result_handler = BBoxResultHandler()
kpt_result_handler = KeypointResultHandler()

stat_path = '/home/clayton/workspace/prj/data_keep/data/toyota/test_cases/statistics/20200430/condition_D/morning_fixed.json'
stat_data = json.load(open(stat_path, 'r'))

for frame_name, frame_data in stat_data.items():
    score_list = frame_data['score_list']
    bbox_list = frame_data['bbox_list']
    bbox_list = []
    vis_keypoints_list = frame_data['vis_keypoints_list']
    kpt_confidences_list = frame_data['kpt_confidences_list']