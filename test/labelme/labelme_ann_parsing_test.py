from logger import logger
from common_utils.common_types.point import Point2D_List
from annotation_utils.labelme.structs import LabelmeAnnotation, LabelmeShape

ann = LabelmeAnnotation.load_from_path('/home/clayton/workspace/test/labelme_testing/orig_cat.json')
ann.shapes.append(
    shape=LabelmeShape(
        label='test_bbox',
        points=Point2D_List.from_list([[50, 50], [100, 100]], demarcation=True),
        shape_type='rectangle'
    )
)

for shape in ann.shapes:
    logger.purple(f'shape.label: {shape.label}')
    logger.purple(f'shape.shape_type: {shape.shape_type}')
    logger.cyan(f'shape.points.to_numpy().shape: {shape.points.to_numpy().shape}')

ann.save_to_path('/home/clayton/workspace/test/labelme_testing/cat.json')