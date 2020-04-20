from logger import logger
from common_utils.path_utils import get_all_files_of_extension, get_rootname_from_path, \
    get_filename, get_dirpath_from_filepath
from common_utils.file_utils import file_exists
from common_utils.time_utils import get_present_year, get_present_time_Ymd, \
    get_ctime
from common_utils.user_utils import get_username
from ..labelme.annotation import LabelMeAnnotationHandler
from ..coco.structs import COCO_Info, COCO_License_Handler, COCO_License, \
    COCO_Image_Handler, COCO_Image, COCO_Annotation_Handler, COCO_Annotation, \
    COCO_Category_Handler, COCO_Category
from ..util.utils import polygon2segmentation, polygon2bbox, point_inside_polygon, \
    labeled_points2keypoints, rectangle2bbox, point_inside_rectangle

class LabelMeCOCOConverter:
    def __init__(
        self, labelme_img_dir_list: list, labelme_annotation_dir_list: list, labelme_index_range_list: list=None,
        bound_type_list: str=None, category_dict: dict=None
    ):
        # Image Directory List (for getting dates modified)
        self.labelme_img_dir_list = labelme_img_dir_list

        # Category Dictionary (for handling keypoints)
        self.category_dict = category_dict

        # Bound type used for grouping keypoints
        self.bound_type_list = bound_type_list if bound_type_list is not None else ['poly'] * len(self.labelme_img_dir_list)

        # Annotation Paths List
        self.labelme_annotation_dir_list = labelme_annotation_dir_list
        self.annotation_pathlist_list = []
        for labelme_annotation_dir in labelme_annotation_dir_list:
            annotation_pathlist = get_all_files_of_extension(
                dir_path=labelme_annotation_dir,
                extension="json"
            )
            self.annotation_pathlist_list.append(annotation_pathlist)

        # Index ranges to be loaded
        self.labelme_index_range_list = labelme_index_range_list if labelme_index_range_list is not None else ['all'] * len(self.labelme_img_dir_list)

        # LabelMe Handler
        self.labelme_annotation_handler = LabelMeAnnotationHandler()

        # COCO Data
        self.info = None
        self.licenses = None
        self.images = None
        self.annotations = None
        self.categories = None

        # For keeping the ids unique
        self.img_path2id_dict = {}

    def load_annotation_paths(self):
        for i, annotation_pathlist in zip(range(len(self.annotation_pathlist_list)), self.annotation_pathlist_list):
            index_range = self.labelme_index_range_list[i]
            if index_range == 'none':
                continue
            elif index_range == 'all':
                for annotation_path in annotation_pathlist:
                    self.labelme_annotation_handler.add(
                        key=len(self.labelme_annotation_handler.annotations), annotation_path=annotation_path, img_dir=self.labelme_img_dir_list[i],
                        bound_type=self.bound_type_list[i]
                    )
            else:
                for annotation_path in annotation_pathlist:
                    annotation_index = int(get_rootname_from_path(annotation_path))
                    for start_index, end_index in index_range:
                        if annotation_index >= start_index and annotation_index <= end_index:
                            self.labelme_annotation_handler.add(
                                key=len(self.labelme_annotation_handler.annotations), annotation_path=annotation_path, img_dir=self.labelme_img_dir_list[i],
                                bound_type=self.bound_type_list[i]
                            )
                            break

    def load_annotation_data(self):
        self.labelme_annotation_handler.load_remaining()

    def load_labelme(self):
        self.load_annotation_paths()
        self.load_annotation_data()

    def determine_coco_fields(self):
        self.info = self.get_info()
        self.licenses = self.get_licenses()
        self.images = self.get_images()
        self.categories = self.get_categories()
        self.annotations = self.get_annotations()

    def get_info(self) -> COCO_Info:
        return COCO_Info(
            description="COCO Dataset generated from labelme tool",
            url="https://github.com/wkentaro/labelme",
            version="1.0",
            year=get_present_year(),
            contributor=get_username(),
            date_created=get_present_time_Ymd()
        )

    def get_licenses(self) -> COCO_License_Handler:
        coco_license_handler = COCO_License_Handler()
        labelme_license = COCO_License(
            url="https://github.com/wkentaro/labelme/blob/master/LICENSE",
            id=1,
            name="GNU General Public License"
        )
        coco_license_handler.add(labelme_license)
        return coco_license_handler

    def get_images(self) -> COCO_Image_Handler:
        coco_image_handler = COCO_Image_Handler()
        for annotation in self.labelme_annotation_handler.annotations.values():
            height = annotation.img_height
            filename = get_filename(annotation.img_path)
            logger.yellow(f"annotation.img_dir: {annotation.img_dir}")
            logger.purple(f"annotation.img_path: {annotation.img_path}")
            img_path = f"{annotation.img_dir}/{filename}"
            if file_exists(img_path):
                date_captured = get_ctime(img_path)
            else:
                assumed_img_dir = get_dirpath_from_filepath(annotation.img_path)
                logger.warning(f"Image directory not specified.")
                logger.warning(f"Assuming img_dir = {assumed_img_dir}")
                date_captured = get_ctime(annotation.img_path)

            id = len(self.img_path2id_dict)
            self.img_path2id_dict[img_path] = id
            coco_image = COCO_Image(
                license_id=1,
                file_name=filename,
                coco_url=img_path,
                height=annotation.img_height,
                width=annotation.img_width,
                date_captured=date_captured,
                flickr_url=None,
                id=id
            )
            coco_image_handler.add(coco_image)
        return coco_image_handler

    def get_annotations(self) -> COCO_Annotation_Handler:
        """
        Note: There is no indication of whether two separate segmentations
        correspond to the same object in labelme, so for now I will just
        make it so that all segmentations are independent.
        In other words, each segmentation list will only have at most one
        element.
        TODO: Figure out how to resolve the above mentioned issue.
        """
        coco_annotation_handler = COCO_Annotation_Handler()
        for annotation in self.labelme_annotation_handler.annotations.values():
            img_filename = get_filename(annotation.img_path)
            img_path = f"{annotation.img_dir}/{img_filename}"
            img_id = self.img_path2id_dict[img_path]
            logger.info(f"Retrieving Annotation: {img_id}")
            # here
            # circles = annotation.shape_handler.circles
            # lines = annotation.shape_handler.lines
            # linestrips = annotation.shape_handler.linestrips

            seg_dict = {}
            if annotation.bound_type == 'poly':
                polygons = annotation.shape_handler.polygons
                if len(polygons) > 0:
                    for polygon in polygons:
                        segmentation_list = []
                        segmentation = polygon2segmentation(polygon)
                        segmentation_list.append(segmentation)
                        seg_bbox = polygon2bbox(polygon)
                        seg_dict[len(seg_dict)] = {
                            'segmentation_list': segmentation_list,
                            'seg_bbox_area': seg_bbox.area,
                            'seg_bbox': seg_bbox,
                            'shape_obj': polygon,
                            'labeled_points': [],
                            'label': polygon.label
                        }
            elif annotation.bound_type == 'rect':
                rectangles = annotation.shape_handler.rectangles
                if len(rectangles) > 0:
                    for rectangle in rectangles:
                        bbox = rectangle2bbox(rectangle=rectangle)
                        seg_dict[len(seg_dict)] = {
                            'segmentation_list': [],
                            'seg_bbox_area': bbox.area,
                            'seg_bbox': bbox,
                            'shape_obj': rectangle,
                            'labeled_points': [],
                            'label': rectangle.label
                        }

            points = annotation.shape_handler.points
            if len(points) > 0:
                for point in points:
                    for seg_item in seg_dict.values():
                        if annotation.bound_type == 'poly':
                            if point_inside_polygon(
                                point=point,
                                polygon=seg_item['shape_obj']
                            ):
                                seg_item['labeled_points'].append(point)
                        elif annotation.bound_type == 'rect':
                            if point_inside_rectangle(
                                point=point,
                                rectangle=seg_item['shape_obj']
                            ):
                                seg_item['labeled_points'].append(point)
                        else:
                            logger.error(f"Invalid bound_type: {annotation.bound_type}")
                            raise Exception

            for seg_item in seg_dict.values():
                num_keypoints = len(seg_item['labeled_points'])
                keypoint_labels = None
                category_found = False
                category_id = None
                for category in self.categories.category_list:
                    if category.name == seg_item['label']:
                        keypoint_labels = category.keypoints
                        category_found = True
                        category_id = category.id
                        break
                if not category_found:
                    logger.warning(f"Warning: Category label {seg_item['label']} not found.")
                    logger.warning(f"Ignoring {len(seg_item['labeled_points'])} labeled points.")

                keypoints = labeled_points2keypoints(
                    keypoint_labels=keypoint_labels,
                    labeled_points=seg_item['labeled_points'],
                    img_path=img_path,
                    annotation_path=annotation.annotation_path
                ) if keypoint_labels is not None else None
                if num_keypoints > 0 and keypoint_labels is None:
                    logger.warning(f"Warning: No keypoint_labels specified.")
                    logger.warning(f"Cannot generate keypoints without keypoint_labels.")
                    logger.warning(f"{num_keypoints} labeled_points ignored.")
                
                image_id = img_id
                seg_bbox = seg_item['seg_bbox']
                coco_annotation = COCO_Annotation(
                    segmentation=seg_item['segmentation_list'],
                    num_keypoints=num_keypoints,
                    area=seg_item['seg_bbox_area'],
                    iscrowd=0,
                    keypoints=keypoints,
                    image_id=image_id,
                    bbox=[seg_bbox.xmin, seg_bbox.ymin, seg_bbox.width, seg_bbox.height],
                    category_id=category_id,
                    id=len(coco_annotation_handler.annotation_list)
                )
                coco_annotation_handler.add(coco_annotation)
        return coco_annotation_handler

    def get_categories(self) -> COCO_Category_Handler:
        coco_category_handler = COCO_Category_Handler()        
        if self.category_dict is None:
            logger.warning("Warning: No category_dict provided.")
            logger.warning("All keypoints will be ignored.")
            return coco_category_handler
        for key, data in self.category_dict.items():
            coco_category = COCO_Category(
                supercategory=key,
                id=len(coco_category_handler.category_list),
                name=key,
                keypoints=data['keypoints'],
                skeleton=data['skeleton']
            )
            coco_category_handler.add(coco_category)
        return coco_category_handler

    def test(self):
        self.load_labelme()
        self.determine_coco_fields()
        logger.cyan("Info")
        logger.blue(self.info)
        logger.cyan("Licenses")
        logger.blue(self.licenses)
        logger.cyan("Images")
        logger.blue(self.images)
        logger.cyan("Annotations")
        logger.blue(self.annotations)
        logger.cyan("Categories")
        logger.blue(self.categories)
