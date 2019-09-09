import cv2
from ...logger.logger_handler import logger
from ...common_utils.check_utils import check_input_path_and_output_dir, check_dir_exists, \
    check_file_exists
from ...common_utils.image_utils import resize_img
from ...common_utils.path_utils import get_dirpath_from_filepath
from ..labelme_annotation import LabelMeAnnotation, LabelMeAnnotationParser
from ..labelme_annotation_writer import LabelMeAnnotationWriter
from ...common_utils.common_types import Point, Rectangle, Polygon, Size, Resize

def write_resized_image(input_path: str, output_path: str, target_size: Size, silent: bool=False):
    check_input_path_and_output_dir(input_path=input_path, output_path=output_path)
    img = cv2.imread(input_path)
    resized_img = resize_img(img=img, size=target_size)
    cv2.imwrite(filename=output_path, img=resized_img)
    if not silent:
        logger.info(f"Wrote resized image to {output_path}")

def write_resized_json(
    input_img_path: str, input_json_path: str, output_img_path: str, output_json_path: str,
    target_size: Size, bound_type: str, silent: bool=False
):
    check_input_path_and_output_dir(input_path=input_img_path, output_path=output_img_path)
    check_input_path_and_output_dir(input_path=input_json_path, output_path=output_json_path)

    output_img_dir = get_dirpath_from_filepath(output_img_path)

    annotation = LabelMeAnnotation(
        annotation_path=input_json_path,
        img_dir=output_img_dir,
        bound_type=bound_type
    )
    parser = LabelMeAnnotationParser(annotation_path=input_json_path)
    parser.load()
    img = cv2.imread(filename=input_img_path)
    orig_size = Size.from_cv2_shape(img.shape)
    resize = Resize(old_size=orig_size, new_size=target_size)

    for shape in parser.shape_handler.points:
        shape.points = resize.on_point(
            Point.from_labelme_point_list(shape.points)
        ).to_labelme_format()
    for shape in parser.shape_handler.rectangles:
        shape.points = resize.on_rectangle(
            Rectangle.from_labelme_point_list(shape.points)
        ).to_labelme_format()
    for shape in parser.shape_handler.polygons:
        shape.points = resize.on_polygon(
            Polygon.from_labelme_point_list(shape.points)
        ).to_labelme_format()

    # Update Shapes
    parser.shape_handler2shapes()

    # Get Info From Resized Image
    check_file_exists(output_img_path)
    parser.img_path = output_img_path
    img = cv2.imread(output_img_path)
    parser.img_height, parser.img_width = img.shape[:2]

    annotation.copy_from_parser(
        parser=parser,
        annotation_path=output_json_path,
        img_dir=output_img_dir,
        bound_type='rect'
    )
    
    annotation_writer = LabelMeAnnotationWriter(labelme_annotation=annotation)
    annotation_writer.write()
    if not silent:
        logger.info(f"Wrote resized labelme annotation to {output_json_path}")