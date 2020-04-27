from logger import logger
from ...labelme.annotation import Shape

def check_shape_type(shape: Shape, shape_type: str):
    if shape.shape_type != shape_type:
        logger.error(f"The given shape is not a {shape_type}.")
        logger.error(f"label: {shape.label}, shape_type: {shape.shape_type}")
        raise Exception