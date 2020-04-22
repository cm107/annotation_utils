from annotation_utils.labelme.structs import LabelmeAnnotationHandler
from common_utils.common_types.bbox import BBox
from common_utils.common_types.point import Point2D_List

handler = LabelmeAnnotationHandler.load_from_dir('/home/clayton/workspace/prj/data_keep/data/toyota/dataset/real/phone_videos/new/sampled_data/VID_20200217_161043/json')
bbox_scale_factor = 1.4
for ann in handler:
    for shape in ann.shapes:
        if shape.shape_type == 'rectangle':
            bbox = BBox.from_p0p1(shape.points.to_list(demarcation=2))
            bbox = bbox.scale_about_center(scale_factor=bbox_scale_factor, frame_shape=[ann.img_h, ann.img_w])
            shape.points = Point2D_List.from_list(value_list=bbox.to_list(), demarcation=False)

handler.save_to_dir(
    json_save_dir='/home/clayton/workspace/test/labelme_testing/temp/json',
    src_img_dir='/home/clayton/workspace/prj/data_keep/data/toyota/dataset/real/phone_videos/new/sampled_data/VID_20200217_161043/img',
    dst_img_dir='/home/clayton/workspace/test/labelme_testing/temp/img',
    overwrite=True
)