import cv2
import numpy as np
from logger import logger
from common_utils.check_utils import check_input_path_and_output_dir, \
    check_file_exists
from common_utils.image_utils import resize_img
from common_utils.path_utils import get_dirpath_from_filepath, get_rootname_from_path, get_extension_from_path
from common_utils.file_utils import delete_file
from ...labelme.annotation import LabelMeAnnotation, LabelMeAnnotationParser, Shape, ShapeHandler
from ...labelme.writer import LabelMeAnnotationWriter
from common_utils.common_types import Point, Rectangle, Polygon, Size, Resize
from common_utils.common_types.bbox import BBox
from ...labelme.writer import LabelMeAnnotationWriter

def write_cropped_image(src_path: str, dst_path: str, bbox: BBox, verbose: bool=False):
    check_input_path_and_output_dir(input_path=src_path, output_path=dst_path)
    img = cv2.imread(src_path)
    int_bbox = bbox.to_int()
    cropped_img = img[int_bbox.ymin:int_bbox.ymax, int_bbox.xmin:int_bbox.xmax, :]
    cv2.imwrite(filename=dst_path, img=cropped_img)

def write_cropped_json(src_img_path: str, src_json_path: str, dst_img_path: str, dst_json_path: str, bound_type='rect', verbose: bool=False):
    def process_shape(shape: Shape, bbox: BBox, new_shape_handler: ShapeHandler):
        points = [Point.from_list(point) for point in shape.points]
        
        contained_count = 0
        for point in points:
            if bbox.contains(point):
                contained_count += 1
        if contained_count == 0:
            return
        elif contained_count == len(points):
            pass
        else:
            logger.error(f"Found a shape that is only partially contained by a bbox.")
            logger.error(f"Shape: {shape}")
            logger.error(f"BBox: {bbox}")

        cropped_points = [Point(x=point.x-bbox.xmin, y=point.y-bbox.ymin) for point in points]
        for point in cropped_points:
            if point.x < 0 or point.y < 0:
                logger.error(f"Encountered negative point after crop: {point}")
                raise Exception
        new_shape = shape.copy()
        new_shape.points = [cropped_point.to_list() for cropped_point in cropped_points]
        new_shape_handler.add(new_shape)
    
    check_input_path_and_output_dir(input_path=src_img_path, output_path=dst_img_path)
    check_input_path_and_output_dir(input_path=src_json_path, output_path=dst_json_path)
    output_img_dir = get_dirpath_from_filepath(dst_img_path)

    annotation = LabelMeAnnotation(
        annotation_path=src_img_path,
        img_dir=dst_img_path,
        bound_type=bound_type
    )
    parser = LabelMeAnnotationParser(annotation_path=src_json_path)
    parser.load()

    bbox_list = []
    for rect in parser.shape_handler.rectangles:
        numpy_array = np.array(rect.points)
        if numpy_array.shape != (2, 2):
            logger.error(f"Encountered rectangle with invalid shape: {numpy_array.shape}")
            logger.error(f"rect: {rect}")
            raise Exception
        xmin, xmax = numpy_array.T[0].min(), numpy_array.T[0].max()
        ymin, ymax = numpy_array.T[1].min(), numpy_array.T[1].max()
        bbox_list.append(BBox.from_list([xmin, ymin, xmax, ymax]))

    img = cv2.imread(src_img_path)
    img_h, img_w = img.shape[:2]

    for i, bbox in enumerate(bbox_list):
        bbox = BBox.buffer(bbox)
        new_shape_handler = ShapeHandler()
        for shape_group in [parser.shape_handler.points, parser.shape_handler.rectangles, parser.shape_handler.polygons]:
            for shape in shape_group:
                process_shape(shape=shape, bbox=bbox, new_shape_handler=new_shape_handler)
        new_shape_list = new_shape_handler.to_shape_list()
        if len(new_shape_list) > 0:
            img_rootname, json_rootname = get_rootname_from_path(dst_img_path), get_rootname_from_path(dst_json_path)
            dst_img_dir, dst_json_dir = get_dirpath_from_filepath(dst_img_path), get_dirpath_from_filepath(dst_json_path)
            dst_img_extension = get_extension_from_path(dst_img_path)
            dst_cropped_img_path = f"{dst_img_dir}/{img_rootname}_{i}.{dst_img_extension}"
            dst_cropped_json_path = f"{dst_json_dir}/{json_rootname}_{i}.json"
            write_cropped_image(src_path=src_img_path, dst_path=dst_cropped_img_path, bbox=bbox, verbose=verbose)
            cropped_labelme_ann = annotation.copy()
            cropped_labelme_ann.annotation_path = dst_cropped_json_path
            cropped_labelme_ann.img_dir = dst_img_dir
            cropped_labelme_ann.img_path = dst_cropped_img_path
            cropped_img = cv2.imread(dst_cropped_img_path)
            cropped_img_h, cropped_img_w = cropped_img.shape[:2]
            cropped_labelme_ann.img_height = cropped_img_h
            cropped_labelme_ann.img_width = cropped_img_w
            cropped_labelme_ann.shapes = new_shape_list
            cropped_labelme_ann.shape_handler = new_shape_handler
            writer = LabelMeAnnotationWriter(cropped_labelme_ann)
            writer.write()
            if verbose:
                logger.info(f"Wrote {dst_cropped_json_path}")

def write_resized_image(input_path: str, output_path: str, target_size: Size, silent: bool=False):
    check_input_path_and_output_dir(input_path=input_path, output_path=output_path)
    img = cv2.imread(input_path)
    resized_img = resize_img(img=img, size=target_size)
    cv2.imwrite(filename=output_path, img=resized_img)
    if not silent:
        logger.info(f"Wrote resized image to {output_path}")

def write_resized_json(
    input_img_path: str, input_json_path: str, output_img_path: str, output_json_path: str,
    target_size: Size, bound_type: str='rect', silent: bool=False
):
    # Note: bound_type doesn't have any significance right now.
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

def copy_annotation(src_img_path: str, src_json_path: str, dst_img_path: str, dst_json_path: str, bound_type: str):
    src_img_dir = get_dirpath_from_filepath(src_img_path)
    dst_img_dir = get_dirpath_from_filepath(dst_img_path)
    labelme_annotation = LabelMeAnnotation(
        annotation_path=src_json_path,
        img_dir=src_img_dir,
        bound_type=bound_type
    )
    labelme_annotation.load()
    labelme_annotation.img_dir = dst_img_dir
    labelme_annotation.img_path = dst_img_path
    labelme_annotation.annotation_path = dst_json_path
    writer = LabelMeAnnotationWriter(labelme_annotation)
    writer.write()

def move_annotation(src_img_path: str, src_json_path: str, dst_img_path: str, dst_json_path: str, bound_type: str):
    copy_annotation(
        src_img_path=src_img_path,
        src_json_path=src_json_path,
        dst_img_path=dst_img_path,
        dst_json_path=dst_json_path,
        bound_type=bound_type
    )
    delete_file(src_json_path)