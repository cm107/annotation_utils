from common_utils.path_utils import get_all_files_of_extension, \
    get_rootname_from_path
from ..labelimg.annotation import LabelImgAnnotationHandler, LabelImgAnnotation
from ..labelme.annotation import LabelMeAnnotationHandler, LabelMeAnnotation, Shape, ShapeHandler
import labelme

class LabelImgLabelMeConverter:
    def __init__(
        self, labelimg_annotation_dir: str, labelme_annotation_dir: str, img_dir: str
    ):
        self.labelimg_annotation_dir = labelimg_annotation_dir
        self.labelme_annotation_dir = labelme_annotation_dir
        self.img_dir = img_dir

        # LabelImg Handler
        self.labelimg_annotation_handler = LabelImgAnnotationHandler()

    def load_annotation_paths(self):
        annotation_paths = get_all_files_of_extension(dir_path=self.labelimg_annotation_dir, extension='xml')
        for annotation_path in annotation_paths:
            rootname = get_rootname_from_path(path=annotation_path)
            img_path = f"{self.img_dir}/{rootname}.png"
            xml_path = f"{self.labelimg_annotation_dir}/{rootname}.xml"
            self.labelimg_annotation_handler.add(
                key=len(self.labelimg_annotation_handler.annotations),
                annotation_path=xml_path,
                img_path=img_path
            )

    def load_annotation_data(self):
        self.labelimg_annotation_handler.load_remaining()

    def load_labelimg(self):
        self.load_annotation_paths()
        self.load_annotation_data()

    def get_labelme_annotation_handler(self) -> LabelMeAnnotationHandler:
        labelme_annotation_handler = LabelMeAnnotationHandler()
        
        for labelimg_annotation in self.labelimg_annotation_handler.annotations.values():
            rootname = get_rootname_from_path(path=labelimg_annotation.annotation_path)
            labelme_annotation_path = f"{self.labelme_annotation_dir}/{rootname}.json"
            labelme_annotation = LabelMeAnnotation(
                annotation_path=labelme_annotation_path,
                img_dir=self.img_dir,
                bound_type='rect'
            )
            shape_handler = self.get_shape_handler(labelimg_annotation)
            shapes = self.get_shapes(shape_handler=shape_handler)

            labelme_annotation.version = labelme.__version__
            labelme_annotation.flags = {}
            labelme_annotation.shapes = shapes
            labelme_annotation.line_color = [0, 255, 0, 128]
            labelme_annotation.fill_color = [66, 255, 33, 128]
            labelme_annotation.img_path = labelimg_annotation.img_path
            labelme_annotation.img_height = labelimg_annotation.size.height
            labelme_annotation.img_width = labelimg_annotation.size.width
            labelme_annotation.shape_handler = shape_handler
        
            labelme_annotation_handler.annotations[
                len(labelme_annotation_handler.annotations)
            ] = labelme_annotation

        return labelme_annotation_handler

    def get_shape_handler(self, labelimg_annotation: LabelImgAnnotation) -> ShapeHandler:
        shape_handler = ShapeHandler()
        for rectangle in labelimg_annotation.rectangle_handler.rectangles:
            name = rectangle.name
            bounding_box = rectangle.bounding_box

            points = [
                [bounding_box.xmin, bounding_box.ymin],
                [bounding_box.xmax, bounding_box.ymax]
            ]
            shape = Shape(
                label=name,
                line_color=None,
                fill_color=None,
                points=points,
                shape_type='rectangle',
                flags=""
            )
            shape_handler.add(shape)
        return shape_handler

    def get_shapes(self, shape_handler: ShapeHandler):
        shapes = []
        for rectangle in shape_handler.rectangles:
            shape_dict = {}
            shape_dict['label'] = rectangle.label
            shape_dict['line_color'] = rectangle.line_color
            shape_dict['fill_color'] = rectangle.fill_color
            shape_dict['points'] = rectangle.points
            shape_dict['shape_type'] = rectangle.shape_type
            shape_dict['flags'] = {}
            shapes.append(shape_dict)
        return shapes
            