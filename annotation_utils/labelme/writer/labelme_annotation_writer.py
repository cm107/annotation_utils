import base64
import json
from logger import logger
from ..annotation import LabelMeAnnotation, LabelMeAnnotationHandler

class LabelMeAnnotationWriter:
    def __init__(self, labelme_annotation: LabelMeAnnotation):
        self.labelme_annotation = labelme_annotation

    def build_json_dict(self):
        version = self.labelme_annotation.version
        flags = self.labelme_annotation.flags
        shapes = self.labelme_annotation.shapes
        img_path = self.labelme_annotation.img_path
        img_data = open(img_path, 'rb').read()
        img_data = base64.b64encode(img_data).decode('utf-8')
        img_height = self.labelme_annotation.img_height
        img_width = self.labelme_annotation.img_width

        json_dict = {}
        json_dict['version'] = version
        json_dict['flags'] = flags
        json_dict['shapes'] = shapes
        json_dict['imagePath'] = img_path
        json_dict['imageData'] = img_data
        json_dict['imageHeight'] = img_height
        json_dict['imageWidth'] = img_width
        return json_dict

    def write_json_dict(self, json_dict: dict):
        json.dump(json_dict, open(self.labelme_annotation.annotation_path, 'w'))
        logger.info(f"JSON dict has been written to:\n{self.labelme_annotation.annotation_path}")

    def write(self):
        json_dict = self.build_json_dict()
        self.write_json_dict(json_dict)

class LabelMeAnnotationHandlerWriter:
    def __init__(self, labelme_annotation_handler: LabelMeAnnotationHandler):
        self.labelme_annotation_handler = labelme_annotation_handler

    def write_all(self):
        for labelme_annotation in self.labelme_annotation_handler.annotations.values():
            labelme_annotation_writer = LabelMeAnnotationWriter(labelme_annotation)
            labelme_annotation_writer.write()