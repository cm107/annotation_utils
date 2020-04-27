import numpy as np
from logger import logger
from ...labelme.annotation import Shape
from ..checks import check_shape_type
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from ..common_types import BBox

def polygon2segmentation(polygon: Shape):
    check_shape_type(shape=polygon, shape_type='polygon')
    segmentation = []
    for point in polygon.points:
        segmentation.append(round(point[0], 2))
        segmentation.append(round(point[1], 2))
    return segmentation

def polygon2bbox(polygon: Shape) -> BBox:
    check_shape_type(shape=polygon, shape_type='polygon')
    x_values = np.array(polygon.points).T[0]
    y_values = np.array(polygon.points).T[1]
    xmin = round(x_values.min(), 2)
    ymin = round(y_values.min(), 2)
    xmax = round(x_values.max(), 2)
    ymax = round(y_values.max(), 2)
    return BBox(x=xmin, y=ymin, width=xmax-xmin, height=ymax-ymin)

def point_inside_polygon(point: Shape, polygon: Shape):
    check_shape_type(shape=point, shape_type='point')
    check_shape_type(shape=polygon, shape_type='polygon')
    shapely_point = Point(point.points[0])
    shapely_polygon = Polygon(polygon.points)
    return shapely_point.within(shapely_polygon)

def int_skeleton2str_skeleton(int_skeleton: list, keypoint_labels: list):
    str_skeleton = []
    for int_bone in int_skeleton:
        bone_start = keypoint_labels[int_bone[0]-1] 
        bone_end = keypoint_labels[int_bone[1]-1]
        str_bone = [bone_start, bone_end]
        str_skeleton.append(str_bone)
    return str_skeleton

def labeled_points2keypoints(keypoint_labels: list, labeled_points: list, img_path: str, annotation_path: str) -> list:
    """
    Note that keypoint entries are of the format (x, y, v), where
    v = 0: not labeled (in which case x=y=0)
    v = 1: labeled but not visible
    v = 2: labeled and visible
    """
    keypoint_registry = {}
    for keypoint_label in keypoint_labels:
        keypoint_registry[keypoint_label] = None
    
    for point in labeled_points:
        if point.label not in keypoint_labels:
            logger.error(f"Point label {point.label} not found in keypoint labels.")
            logger.error(f"Keypoint Labels: {keypoint_labels}")
            logger.error(f"Please modify the following image/annotation pair:")
            logger.error(f"\tImage: {img_path}")
            logger.error(f"\tAnnotation: {annotation_path}")
            raise Exception
        if keypoint_registry[point.label] is not None:
            logger.error(f"{point.label} point {point.points[0]} already exists in keypoint_registry.")
            logger.error(f"keypoint_registry[{point.label}]: {keypoint_registry[point.label].points[0]}")
            logger.error(f"Please modify the following image/annotation pair:")
            logger.error(f"\tImage: {img_path}")
            logger.error(f"\tAnnotation: {annotation_path}")
            raise Exception
        keypoint_registry[point.label] = point

    keypoints = []
    for point in keypoint_registry.values():
        if point is None:
            entry = [0, 0, 0] # not labeled
            keypoints.extend(entry)
        elif point.hidden:
            entry = [point.points[0][0], point.points[0][1], 1] # labeled but not visible
            keypoints.extend(entry)
        else:
            entry = [point.points[0][0], point.points[0][1], 2] # labeled and visible
            keypoints.extend(entry)
    return keypoints

def rectangle2bbox(rectangle: Shape) -> BBox:
    check_shape_type(shape=rectangle, shape_type='rectangle')
    p0, p1 = rectangle.points
    xmin = round(min(p0[0], p1[0]), 2)
    xmax = round(max(p0[0], p1[0]), 2)
    ymin = round(min(p0[1], p1[1]), 2)
    ymax = round(max(p0[1], p1[1]), 2)
    return BBox(x=xmin, y=ymin, width=xmax-xmin, height=ymax-ymin)

def point_inside_rectangle(point: Shape, rectangle: Shape):
    check_shape_type(shape=point, shape_type='point')
    check_shape_type(shape=rectangle, shape_type='rectangle')
    px = point.points[0][0]
    py = point.points[0][1]
    bbox = rectangle2bbox(rectangle)
    if px >= bbox.xmin and px <= bbox.xmax and py >= bbox.ymin and py <= bbox.ymax:
        return True
    else:
        return False