from logger import logger
from xml.etree import ElementTree
from common_utils.path_utils import get_all_files_of_extension, get_rootname_from_path

class Source:
    def __init__(self, database: str):
        self.database = database

class Size:
    def __init__(self, width: int, height: int, depth: int):
        self.width = width
        self.height = height
        self.depth = depth

class BoundingBox:
    def __init__(self, xmin: int, ymin: int, xmax: int, ymax: int):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

class Rectangle:
    def __init__(self, name: str, pose: str, truncated: int, difficult: int, bounding_box: BoundingBox):
        self.name = name
        self.pose = pose
        self.truncated = truncated
        self.difficult = difficult
        self.bounding_box = bounding_box

class RectangleHandler:
    def __init__(self):
        self.rectangles = []

    def add(self, rectangle: Rectangle):
        self.rectangles.append(rectangle)

class LabelImgAnnotationParser:
    def __init__(self, annotation_path: str):
        self.annotation_path = annotation_path

        self.folder = None
        self.filename = None
        self.path = None
        self.source = None
        self.size = None
        self.segmented = None
        self.rectangle_handler = None

    def load(self):
        self.load_data()
        logger.info(f"LabelImg Annotation Loaded: {self.annotation_path}")

    def load_data(self):
        tree = ElementTree.parse(source=self.annotation_path)
        root = tree.getroot()

        self.folder = root.find('folder').text
        self.filename = root.find('filename').text
        self.path = root.find('path').text
        source_tree = root.find('source')
        database = source_tree.find('database').text
        self.source = Source(database=database)
        size_tree = root.find('size')
        width = int(size_tree.find('width').text)
        height = int(size_tree.find('height').text)
        depth = int(size_tree.find('depth').text)
        self.size = Size(width=width, height=height, depth=depth)
        self.segmented = int(root.find('segmented').text)
        object_trees = root.findall('object')
        self.rectangle_handler = RectangleHandler()
        for object_tree in object_trees:
            name = object_tree.find('name').text
            pose = object_tree.find('pose').text
            truncated = int(object_tree.find('truncated').text)
            difficult = int(object_tree.find('difficult').text)
            bndbox_tree = object_tree.find('bndbox')
            xmin = int(bndbox_tree.find('xmin').text)
            ymin = int(bndbox_tree.find('ymin').text)
            xmax = int(bndbox_tree.find('xmax').text)
            ymax = int(bndbox_tree.find('ymax').text)
            bounding_box = BoundingBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
            rectangle = Rectangle(
                name=name,
                pose=pose,
                truncated=truncated,
                difficult=difficult,
                bounding_box=bounding_box
            )
            self.rectangle_handler.add(rectangle)

class LabelImgAnnotation:
    def __init__(
        self, annotation_path: str, img_path: str
    ):
        self.annotation_path = annotation_path
        self.img_path = img_path

        self.folder = None
        self.filename = None
        self.path = None
        self.source = None
        self.size = None
        self.segmented = None
        self.rectangle_handler = None

    def load(self):
        parser = LabelImgAnnotationParser(self.annotation_path)
        parser.load()

        self.folder = parser.folder
        self.filename = parser.filename
        self.path = parser.path
        self.source = parser.source
        self.size = parser.size
        self.segmented = parser.segmented
        self.rectangle_handler = parser.rectangle_handler

    def unload(self):
        self.folder = None
        self.filename = None
        self.path = None
        self.source = None
        self.size = None
        self.segmented = None
        self.rectangle_handler = None

class LabelImgAnnotationHandler:
    def __init__(self):
        self.annotations = {}
        self.added_keys = []
        self.loaded_keys = []
        self.unloaded_keys = []

    def add(self, key, annotation_path: str, img_path: str):
        if key in self.annotations:
            logger.error(f"key={key} already exists in labelimg annotations dictionary")
            raise Exception
        self.annotations[key] = LabelImgAnnotation(annotation_path, img_path)
        self.added_keys.append(key)

    def delete(self, key):
        if key not in self.annotations:
            logger.error(f"key={key} doesn't exist in labelimg annotations dictionary")
            raise Exception
        del self.annotations[key]
        self.added_keys.remove(key)
        if key in self.loaded_keys:
            self.loaded_keys.remove(key)
        if key in self.unloaded_keys:
            self.unloaded_keys.remove(key)

    def load(self, key):
        if key not in self.annotations:
            logger.error(f"key={key} doesn't exist in labelimg annotations dictionary")
            raise Exception
        if key in self.loaded_keys:
            logger.error(f"key={key} has already been loaded in labelimg annotations dictionary")
            raise Exception
        self.annotations[key].load()
        self.loaded_keys.append(key)
        if key in self.unloaded_keys:
            self.unloaded_keys.remove(key)

    def unload(self, key):
        if key in self.annotations:
            logger.error(f"key={key} already exists in labelimg annotations dictionary")
            raise Exception
        if key not in self.loaded_keys:
            logger.error(f"key={key} hasn't been loaded in labelimg annotations dictionary yet")
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

class LabelImgAnnotationHandlerTest:
    def __init__(self):
        self.handler = LabelImgAnnotationHandler()
        self.annotation_dir = "/home/clayton/workspace/prj/data_keep/youtube_promo/xml"
        self.img_dir = "/home/clayton/workspace/prj/data_keep/youtube_promo/img"

    def run(self):
        img_dirs = get_all_files_of_extension(dir_path=self.img_dir, extension='png')
        annotation_paths = get_all_files_of_extension(dir_path=self.annotation_dir, extension='xml')

        for i, annotation_path in zip(range(len(annotation_paths)), annotation_paths):
            rootname = get_rootname_from_path(path=annotation_path)
            img_path = f"{self.img_dir}/{rootname}.png"
            xml_path = f"{self.annotation_dir}/{rootname}.xml"
            self.handler.add(key=len(self.handler.annotations), annotation_path=xml_path, img_path=img_path)

        self.handler.load_remaining()
