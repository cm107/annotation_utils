from __future__ import annotations
import json
from logger import logger

class Shape:
    def __init__(self, label: str, group_id: int, points: list, shape_type: str, flags: str):
        if shape_type == 'point':
            self.hidden = True if label[0] == '&' else False
            self.label = label[1:] if self.hidden else label
        else:
            self.hidden = False
            self.label = label

        self.group_id = group_id
        self.points = points
        self.shape_type = shape_type
        self.flags = flags

    def __str__(self):
        return f"shape_type: {self.shape_type}, label: {self.label}, len(points): {len(self.points)}"

    def __repr__(self):
        return self.__str__()

    @classmethod
    def buffer(self, shape: Shape) -> Shape:
        return shape

    def copy(self) -> Shape:
        return Shape(
            label=self.label,
            group_id=self.group_id,
            points=self.points,
            shape_type=self.shape_type,
            flags=self.flags
        )

    def show_full(self):
        return f"shape_type: {self.shape_type}, label: {self.label}, points: {self.points}"

    def to_dict(self) -> dict:
        shape_dict = {}
        shape_dict['label'] = self.label if not self.hidden else f"&{self.label}"
        shape_dict['group_id'] = self.group_id
        shape_dict['points'] = self.points
        shape_dict['shape_type'] = self.shape_type
        shape_dict['flags'] = self.flags
        return shape_dict

class PolygonShape(Shape):
    def __init__(self, label: str, group_id: int, points: list, flags: str):
        super().__init__(
            label=label,
            group_id=group_id,
            points=points,
            shape_type='polygon',
            flags=flags
        )
        raise NotImplementedError

class RectangleShape(Shape):
    def __init__(self, label: str, group_id: int, points: list, flags: str):
        super().__init__(
            label=label,
            group_id=group_id,
            points=points,
            shape_type='rectangle',
            flags=flags
        )
        raise NotImplementedError

class CircleShape(Shape):
    def __init__(self, label: str, group_id: int, points: list, flags: str):
        super().__init__(
            label=label,
            group_id=group_id,
            points=points,
            shape_type='circle',
            flags=flags
        )
        raise NotImplementedError

class LineShape(Shape):
    def __init__(self, label: str, group_id: int, points: list, flags: str):
        super().__init__(
            label=label,
            group_id=group_id,
            points=points,
            shape_type='line',
            flags=flags
        )
        raise NotImplementedError

class PointShape(Shape):
    def __init__(self, label: str, group_id: int, points: list, flags: str):
        super().__init__(
            label=label,
            group_id=group_id,
            points=points,
            shape_type='point',
            flags=flags
        )
        raise NotImplementedError

class LinestripShape(Shape):
    def __init__(self, label: str, group_id: int, points: list, flags: str):
        super().__init__(
            label=label,
            group_id=group_id,
            points=points,
            shape_type='linestrip',
            flags=flags
        )
        raise NotImplementedError

class ShapeHandler:
    def __init__(self):
        self.polygons = []
        self.rectangles = []
        self.circles = []
        self.lines = []
        self.points = []
        self.linestrips = []

    def __str__(self):
        print_str = f"polygons:\n"
        for polygon in self.polygons:
            print_str += f"\t{polygon}\n"
        print_str += f"rectangles:\n"
        for rectangle in self.rectangles:
            print_str += f"\t{rectangle}\n"
        print_str += f"circles:\n"
        for circle in self.circles:
            print_str += f"\t{circle}\n"
        print_str += f"lines:\n"
        for line in self.lines:
            print_str += f"\t{line}\n"
        print_str += f"points:\n"
        for point in self.points:
            print_str += f"\t{point}\n"
        print_str += f"linestrips:\n"
        for linestrip in self.linestrips:
            print_str += f"\t{linestrip}\n"
        return print_str

    def __repr__(self):
        return self.__str__()

    def show_full(self):
        print_str = f"polygons:\n"
        for polygon in self.polygons:
            print_str += f"\t{polygon.show_full()}\n"
        print_str += f"rectangles:\n"
        for rectangle in self.rectangles:
            print_str += f"\t{rectangle.show_full()}\n"
        print_str += f"circles:\n"
        for circle in self.circles:
            print_str += f"\t{circle.show_full()}\n"
        print_str += f"lines:\n"
        for line in self.lines:
            print_str += f"\t{line.show_full()}\n"
        print_str += f"points:\n"
        for point in self.points:
            print_str += f"\t{point.show_full()}\n"
        print_str += f"linestrips:\n"
        for linestrip in self.linestrips:
            print_str += f"\t{linestrip.show_full()}\n"
        return print_str

    def add(self, shape: Shape):
        if shape.shape_type == 'polygon':
            self.polygons.append(shape)
        elif shape.shape_type == 'rectangle':
            self.rectangles.append(shape)
        elif shape.shape_type == 'circle':
            self.circles.append(shape)
        elif shape.shape_type == 'line':
            self.lines.append(shape)
        elif shape.shape_type == 'point':
            self.points.append(shape)
        elif shape.shape_type == 'linestrip':
            self.linestrips.append(shape)
        else:
            logger.error(f"Invalid shape_type: {shape.shape_type}")
            raise Exception

    def to_shape_list(self):
        shape_list = []
        for shape in self.polygons:
            shape_list.append(shape.to_dict())
        for shape in self.rectangles:
            shape_list.append(shape.to_dict())
        for shape in self.circles:
            shape_list.append(shape.to_dict())
        for shape in self.lines:
            shape_list.append(shape.to_dict())
        for shape in self.points:
            shape_list.append(shape.to_dict())
        for shape in self.linestrips:
            shape_list.append(shape.to_dict())
        return shape_list

class LabelMeAnnotationParser:
    def __init__(self, annotation_path: str):
        self.annotation_path = annotation_path

        self.version = None
        self.flags = None
        self.shapes = None
        self.img_path = None
        self.img_height = None
        self.img_width = None

        self.shape_handler = None

    def load(self):
        logger.purple(f"Loading: {self.annotation_path}")
        self.load_data()
        self.load_shapes()
        logger.info(f"LabelMe Annotation Loaded: {self.annotation_path}")

    def load_data(self):
        data = json.load(open(self.annotation_path, 'r'))
        self.version = data['version']
        self.flags = data['flags']
        self.shapes = data['shapes']
        line_color = data['lineColor'] if 'lineColor' in data else None
        fill_color = data['fillColor'] if 'fillColor' in data else None
        if line_color is not None or fill_color is not None:
            logger.warning(f"As of version 4.2.5, 'lineColor' and 'fillColor' keys in the global dictionary are decapritated.")
        img_path = data['imagePath']
        # get_dirpath_from_filepath(self.annotation_path)
        # self.img_path = rel_to_abs_path(f"{}/{img_path}")
        self.img_path = img_path
        self.img_height = data['imageHeight']
        self.img_width = data['imageWidth']

    def load_shapes(self):
        self.shape_handler = ShapeHandler()
        for shape in self.shapes:
            label = shape['label']
            line_color = shape['line_color'] if 'line_color' in shape else None # removed as of version 4.2.5
            fill_color = shape['fill_color'] if 'fill_color' in shape else None # removed as of version 4.2.5
            points = shape['points']
            group_id = shape['group_id'] if 'group_id' in shape else None # added as of version 4.2.5
            shape_type = shape['shape_type']
            flags = shape['flags'] if 'flags' in shape else None

            if line_color is not None or fill_color is not None:
                logger.warning(f"As of version 4.2.5, 'lineColor' and 'fillColor' keys in the shape dictionary are decapritated.")
            if flags is None:
                logger.warning(f"'flags' field not found in {self.annotation_path}")
                logger.warning(f"Ignoring flags field. Modify the json if you want to use flags.")
            shape_object = Shape( # TODO: Finish fixing
                label=label,
                group_id=group_id,
                points=points,
                shape_type=shape_type,
                flags=flags
            )
            self.shape_handler.add(shape_object)

    def shape_handler2shapes(self):
        self.shapes = self.shape_handler.to_shape_list()


class LabelMeAnnotation:
    def __init__(self, annotation_path: str, img_dir: str, bound_type: str):
        self.annotation_path = annotation_path
        self.img_dir = img_dir
        self.bound_type = bound_type

        self.version = None
        self.flags = None
        self.shapes = None
        self.img_path = None
        self.img_height = None
        self.img_width = None
        self.shape_handler = None

    def __str__(self):
        print_str = f"annotation_path: {self.annotation_path}"
        print_str += f"\nimg_dir: {self.img_dir}"
        print_str += f"\nbound_type: {self.bound_type}"
        print_str += f"\nversion: {self.version}"
        print_str += f"\nflags: {self.flags}"
        print_str += f"\nshapes: {self.shapes}"
        print_str += f"\nimg_path: {self.img_path}"
        print_str += f"\nimg_height: {self.img_height}"
        print_str += f"\nimg_width: {self.img_width}"
        if self.shape_handler is not None:
            print_str += f"\nshape_handler:\n{self.shape_handler.show_full()}"
        else:
            print_str += f"\nshape_handler:\n{self.shape_handler}"
        return print_str

    def __repr__(self):
        return self.__str__()

    def copy(self) -> LabelMeAnnotation:
        labelme_annotation = LabelMeAnnotation(
            annotation_path=self.annotation_path,
            img_dir=self.img_dir,
            bound_type=self.bound_type
        )
        labelme_annotation.version = self.version
        labelme_annotation.flags = self.flags
        labelme_annotation.shapes = self.shapes
        labelme_annotation.img_path = self.img_path
        labelme_annotation.img_height = self.img_height
        labelme_annotation.img_width = self.img_width
        labelme_annotation.shape_handler = self.shape_handler
        return labelme_annotation

    def load(self):
        parser = LabelMeAnnotationParser(self.annotation_path)
        parser.load()
        self.copy_from_parser(parser=parser)

    def copy_from_parser(self, parser: LabelMeAnnotationParser,
        annotation_path: str=None, img_dir: str=None, bound_type: str=None
    ):
        self.annotation_path = annotation_path if annotation_path is not None else self.annotation_path
        self.img_dir = img_dir if img_dir is not None else self.img_dir
        self.bound_type = bound_type if bound_type is not None else self.bound_type

        self.version = parser.version
        self.flags = parser.flags
        self.shapes = parser.shapes
        self.img_path = parser.img_path
        self.img_height = parser.img_height
        self.img_width = parser.img_width
        self.shape_handler = parser.shape_handler

    def unload(self):
        self.version = None
        self.flags = None
        self.shapes = None
        self.img_path = None
        self.img_height = None
        self.img_width = None
        self.shape_handler = None

class LabelMeAnnotationHandler:
    def __init__(self):
        self.annotations = {}
        self.added_keys = []
        self.loaded_keys = []
        self.unloaded_keys = []

    def add(self, key: str, annotation_path: str, img_dir: str, bound_type: str):
        if key in self.annotations:
            logger.error(f"key={key} already exists in labelme annotations dictionary")
            raise Exception
        self.annotations[key] = LabelMeAnnotation(annotation_path, img_dir, bound_type)
        self.added_keys.append(key)

    def delete(self, key: str):
        if key not in self.annotations:
            logger.error(f"key={key} doesn't exist in labelme annotations dictionary")
            raise Exception
        del self.annotations[key]
        self.added_keys.remove(key)
        if key in self.loaded_keys:
            self.loaded_keys.remove(key)
        if key in self.unloaded_keys:
            self.unloaded_keys.remove(key)

    def load(self, key: str):
        if key not in self.annotations:
            logger.error(f"key={key} doesn't exist in labelme annotations dictionary")
            raise Exception
        if key in self.loaded_keys:
            logger.error(f"key={key} has already been loaded in labelme annotations dictionary")
            raise Exception
        self.annotations[key].load()
        self.loaded_keys.append(key)
        if key in self.unloaded_keys:
            self.unloaded_keys.remove(key)

    def unload(self, key: str):
        if key in self.annotations:
            logger.error(f"key={key} already exists in labelme annotations dictionary")
            raise Exception
        if key not in self.loaded_keys:
            logger.error(f"key={key} hasn't been loaded in labelme annotations dictionary yet")
            raise Exception
        self.annotations[key].unload()
        self.unloaded_keys.append(key)
        self.loaded_keys.remove(key)

    def load_list(self, key_list: list):
        for key in key_list:
            self.load(key)

    def unload_list(self, key_list: list):
        for key in key_list:
            self.unload(key)

    def load_remaining(self):
        key_list = [key for key in self.added_keys if key not in self.loaded_keys and key not in self.unloaded_keys]
        self.load_list(key_list)

    def unload_all(self):
        self.unload_list(self.loaded_keys)

    def load_batch(self, batch_size: int):
        remaining_keys = [key for key in self.added_keys if key not in self.loaded_keys and key not in self.unloaded_keys]
        if batch_size < len(remaining_keys):
            self.load_list(remaining_keys[:batch_size])
        else:
            self.load_list(remaining_keys)

    def finished_processing(self) -> bool:
        return len(self.added_keys) > 0 and self.unloaded_keys == self.added_keys