from __future__ import annotations
from typing import List
import cv2
import numpy as np
from logger import logger
from annotation_utils.ndds.structs import NDDS_Dataset
from annotation_utils.coco.structs import COCO_Dataset, \
    COCO_License_Handler, COCO_Image_Handler, COCO_Annotation_Handler, COCO_Category_Handler, \
    COCO_Info, COCO_Category, COCO_License, COCO_Image, COCO_Annotation
from common_utils.file_utils import delete_all_files_in_dir, make_dir_if_not_exists, file_exists
from common_utils.path_utils import get_rootname_from_filename, get_extension_from_filename
from common_utils.check_utils import check_file_exists
from common_utils.common_types.point import Point2D

class Measure_COCO_Dataset(COCO_Dataset):
    def __init__(self, info: COCO_Info, licenses: COCO_License_Handler, images: COCO_Image_Handler, annotations: COCO_Annotation_Handler, categories: COCO_Category_Handler):
        super().__init__(
            info=info,
            licenses=licenses,
            images=images,
            annotations=annotations,
            categories=categories
        )
        self.measure_dataset = COCO_Dataset.new(description='Measure Dataset converted from NDDS')
        self.whole_number_dataset = COCO_Dataset.new(description='Whole Number Dataset converted from NDDS')
        self.digit_dataset = COCO_Dataset.new(description='Digit Dataset converted from NDDS')

    @classmethod
    def _from_base(self, dataset: COCO_Dataset) -> Measure_COCO_Dataset:
        # return Measure_COCO_Dataset(*dataset.__dict__.values())
        return Measure_COCO_Dataset(
            info=dataset.info,
            licenses=dataset.licenses,
            images=dataset.images,
            annotations=dataset.annotations,
            categories=dataset.categories
        )

    @classmethod
    def from_ndds(self, *args, **kwargs) -> Measure_COCO_Dataset:
        return Measure_COCO_Dataset._from_base(super().from_ndds(*args, **kwargs))

    def _prep_output_dir(self, measure_dir: str, whole_number_dir: str, digit_dir: str):
        for output_dir in [measure_dir, whole_number_dir, digit_dir]:
            make_dir_if_not_exists(output_dir)
            delete_all_files_in_dir(output_dir, ask_permission=False)

    def _save_image(self, img: np.ndarray, save_path: str):
        if not file_exists(save_path):
            cv2.imwrite(filename=save_path, img=img)
        else:
            logger.error(f'File already exists: {save_path}')
            raise Exception

    def _load_licenses(self):
        # Load Licenses
        self.measure_dataset.licenses = self.licenses.copy()
        self.whole_number_dataset.licenses.append(
            COCO_License(
                url='https://github.com/cm107/annotation_utils',
                id=0,
                name='Free License'
            )
        )
        self.digit_dataset.licenses.append(
            COCO_License(
                url='https://github.com/cm107/annotation_utils',
                id=0,
                name='Free License'
            )
        )

    def _load_categories(self):
        # Load Categories
        self.measure_dataset.categories.append(
            COCO_Category(
                id=len(self.measure_dataset.categories),
                name='measure'
            )
        )
        self.whole_number_dataset.categories.append(
            COCO_Category(
                id=len(self.whole_number_dataset.categories),
                name='whole_number'
            )
        )
        for i in range(10):
            self.digit_dataset.categories.append(
                COCO_Category(
                    id=len(self.digit_dataset.categories),
                    supercategory='digit',
                    name=str(i)
                )
            )

    def _get_measure_annotations(self, frame_anns: List[COCO_Annotation], orig_image: COCO_Image) -> List[COCO_Annotation]:
        measure_ann_list = []
        for coco_ann in frame_anns:
            coco_cat = self.categories.get_obj_from_id(coco_ann.category_id)
            if coco_cat.name == 'measure':
                # Measure Annotation Update
                new_ann = coco_ann.copy()
                new_ann.id = len(self.measure_dataset.annotations)
                new_ann.category_id = self.measure_dataset.categories.get_unique_category_from_name('measure').id
                new_ann.bbox = new_ann.bbox.to_int()
                # self.measure_dataset.annotations.append(new_ann)
                measure_ann_list.append(new_ann)
            else:
                continue
        if len(measure_ann_list) == 0:
            logger.error(f"Couldn't find any measure annotations in {orig_image.coco_url}")
            raise Exception

        return measure_ann_list
    
    def _load_measure_annotations(self, measure_ann_list: List[COCO_Annotation]):
        for ann in measure_ann_list:
            self.measure_dataset.annotations.append(ann)

    def _process_whole_number_images(self, frame_img: np.ndarray, orig_image: COCO_Image, whole_number_dir: str, measure_ann_list: List[COCO_Annotation]) -> List[COCO_Image]:
        whole_number_coco_image_list = []
        for i, measure_ann in enumerate(measure_ann_list):
            measure_bbox_region = frame_img[measure_ann.bbox.ymin:measure_ann.bbox.ymax, measure_ann.bbox.xmin:measure_ann.bbox.xmax, :]
            measure_img_save_path = f'{whole_number_dir}/{get_rootname_from_filename(orig_image.file_name)}_{i}.{get_extension_from_filename(orig_image.file_name)}'
            self._save_image(img=measure_bbox_region, save_path=measure_img_save_path)

            whole_number_coco_image = COCO_Image.from_img_path(
                img_path=measure_img_save_path,
                license_id=0,
                image_id=len(self.whole_number_dataset.images)
            )
            whole_number_coco_image_list.append(whole_number_coco_image)
            self.whole_number_dataset.images.append(whole_number_coco_image)
        return whole_number_coco_image_list

    def _get_number_anns_list(self, frame_anns: List[COCO_Annotation], measure_ann_list: List[COCO_Annotation], orig_image: COCO_Image, supercategory: str) -> List[COCO_Annotation]:
        anns_list = [[]]*len(measure_ann_list)
        for coco_ann in frame_anns:
            coco_cat = self.categories.get_obj_from_id(coco_ann.category_id)
            if coco_cat.supercategory == supercategory:
                found = False
                for i in range(len(measure_ann_list)):
                    if coco_ann.bbox.within(measure_ann_list[i].bbox):
                        anns_list[i].append(coco_ann)
                        found = True
                        break
                if not found:
                    logger.error(f"Couldn't find matching measure bbox for the given coco_ann.")
                    logger.error(f"coco_ann:\n{coco_ann}")
                    logger.error(f"coco_cat:\n{coco_cat}")
                    logger.error(f"orig_image:\n{orig_image}")
                    raise Exception
        return anns_list
    
    def _process_single_digit_ann(self, frame_img: np.ndarray, whole_number_coco_image: COCO_Image, whole_ann: COCO_Annotation, measure_ann: COCO_Annotation, whole_number_count: int):
        coco_cat = self.categories.get_obj_from_id(whole_ann.category_id)
        whole_number_cat = self.whole_number_dataset.categories.get_unique_category_from_name('whole_number')

        # Update Digit Image Handler
        orig_bbox = whole_ann.bbox.to_int()
        whole_number_bbox_region = frame_img[orig_bbox.ymin:orig_bbox.ymax, orig_bbox.xmin:orig_bbox.xmax, :]
        whole_number_img_save_path = f'{digit_dir}/{get_rootname_from_filename(whole_number_coco_image.file_name)}_{whole_number_count}.{get_extension_from_filename(whole_number_coco_image.file_name)}'
        self._save_image(img=whole_number_bbox_region, save_path=whole_number_img_save_path)
        digit_coco_image = COCO_Image.from_img_path(
            img_path=whole_number_img_save_path,
            license_id=0,
            image_id=len(self.digit_dataset.images)
        )
        self.digit_dataset.images.append(digit_coco_image)

        # Update Whole Number Annotation Handler
        measure_bbox = measure_ann.bbox
        measure_bbox_ref_point = Point2D(x=measure_bbox.xmin, y=measure_bbox.ymin)
        whole_number_seg = whole_ann.segmentation-measure_bbox_ref_point
        whole_number_bbox = whole_number_seg.to_bbox()
        whole_number_coco_ann = COCO_Annotation(
            id=len(self.whole_number_dataset.annotations),
            category_id=whole_number_cat.id,
            image_id=whole_number_coco_image.id,
            segmentation=whole_number_seg,
            bbox=whole_number_bbox,
            area=whole_number_bbox.area()
        )
        self.whole_number_dataset.annotations.append(whole_number_coco_ann)

        # Update Digit Annotation Handler
        digit_cat = self.digit_dataset.categories.get_unique_category_from_name(coco_cat.name)
        whole_number_bbox_ref_point = measure_bbox_ref_point + Point2D(x=whole_number_bbox.xmin, y=whole_number_bbox.ymin)
        digit_seg = whole_ann.segmentation-whole_number_bbox_ref_point
        digit_bbox = digit_seg.to_bbox()
        digit_coco_ann = COCO_Annotation(
            id=len(self.digit_dataset.annotations),
            category_id=digit_cat.id,
            image_id=digit_coco_image.id,
            segmentation=digit_seg,
            bbox=digit_bbox,
            area=digit_bbox.area()
        )
        self.digit_dataset.annotations.append(digit_coco_ann)

    def _get_organized_parts(self, part_anns: List[COCO_Annotation]) -> List[dict]:
        organized_parts = []
        for part_ann in part_anns:
            coco_cat = self.categories.get_obj_from_id(part_ann.category_id)
            whole_name, part_name = coco_cat.name.split('part')
            organized_part = {
                'whole_name': whole_name,
                'part_names': [part_name],
                'anns': [part_ann]
            }
            organized_parts.append(organized_part)
        
        part_del_idx_list = []
        for j in list(range(len(organized_parts)))[::-1]:
            for i in range(j):
                if organized_parts[i]['whole_name'] == organized_parts[j]['whole_name']:
                    organized_parts[i]['part_names'].extend(organized_parts[j]['part_names'])
                    organized_parts[i]['anns'].extend(organized_parts[j]['anns'])
                    part_del_idx_list.append(j)
                    break
        for part_del_idx in part_del_idx_list:
            del organized_parts[part_del_idx]

        return organized_parts

    def _process_organized_part(self, organized_part: dict, frame_img: np.ndarray, whole_number_coco_image: COCO_Image, measure_ann: COCO_Annotation, whole_number_count: int):
        whole_number_cat = self.whole_number_dataset.categories.get_unique_category_from_name('whole_number')
        whole_number_seg = None
        for part_ann in organized_part['anns']:
            whole_number_seg = whole_number_seg + part_ann.segmentation if whole_number_seg is not None else part_ann.segmentation
        
        whole_number_abs_bbox = whole_number_seg.to_bbox().to_int()

        whole_number_bbox_region = frame_img[whole_number_abs_bbox.ymin:whole_number_abs_bbox.ymax, whole_number_abs_bbox.xmin:whole_number_abs_bbox.xmax, :]
        whole_number_img_save_path = f'{digit_dir}/{get_rootname_from_filename(whole_number_coco_image.file_name)}_{whole_number_count}.{get_extension_from_filename(whole_number_coco_image.file_name)}'
        self._save_image(img=whole_number_bbox_region, save_path=whole_number_img_save_path)
        digit_coco_image = COCO_Image.from_img_path(
            img_path=whole_number_img_save_path,
            license_id=0,
            image_id=len(self.digit_dataset.images)
        )
        self.digit_dataset.images.append(digit_coco_image)

        measure_bbox_ref_point = Point2D(x=measure_ann.bbox.xmin, y=measure_ann.bbox.ymin)
        whole_number_seg = whole_number_seg - measure_bbox_ref_point # relative to measure bbox
        whole_number_bbox = whole_number_seg.to_bbox() # relative to measure bbox

        whole_number_coco_ann = COCO_Annotation(
            id=len(self.whole_number_dataset.annotations),
            category_id=whole_number_cat.id,
            image_id=whole_number_coco_image.id,
            segmentation=whole_number_seg,
            bbox=whole_number_bbox,
            area=whole_number_bbox.area()
        )
        self.whole_number_dataset.annotations.append(whole_number_coco_ann)

        for part_ann in organized_part['anns']:
            part_ann = COCO_Annotation.buffer(part_ann)
            coco_cat = self.categories.get_obj_from_id(part_ann.category_id)
            digit_cat = self.digit_dataset.categories.get_unique_category_from_name(coco_cat.name.split('part')[-1])
            whole_number_bbox_ref_point = measure_bbox_ref_point + Point2D(x=whole_number_bbox.xmin, y=whole_number_bbox.ymin)
            digit_seg = part_ann.segmentation - whole_number_bbox_ref_point
            digit_bbox = digit_seg.to_bbox()
            digit_coco_ann = COCO_Annotation(
                id=len(self.digit_dataset.annotations),
                category_id=digit_cat.id,
                image_id=digit_coco_image.id,
                segmentation=digit_seg,
                bbox=digit_bbox,
                area=digit_bbox.area()
            )
            self.digit_dataset.annotations.append(digit_coco_ann)

    def _convert_frame(self, orig_image: COCO_Image, whole_number_dir: str, digit_dir: str):
        whole_number_cat = self.whole_number_dataset.categories.get_unique_category_from_name('whole_number')

        # Get Frame Data
        check_file_exists(orig_image.coco_url)
        frame_img = cv2.imread(orig_image.coco_url)
        frame_anns = self.annotations.get_annotations_from_imgIds([orig_image.id])

        # Process Images and Annotations For Measure Dataset
        self.measure_dataset.images.append(orig_image)
        measure_ann_list = self._get_measure_annotations(frame_anns=frame_anns, orig_image=orig_image)
        self._load_measure_annotations(measure_ann_list=measure_ann_list)

        # Process Whole Number Images
        whole_number_coco_image_list = self._process_whole_number_images(frame_img=frame_img, orig_image=orig_image, whole_number_dir=whole_number_dir, measure_ann_list=measure_ann_list)

        # Process Single Digit Cases
        whole_number_count = 0
        whole_anns_list = self._get_number_anns_list(frame_anns=frame_anns, measure_ann_list=measure_ann_list, orig_image=orig_image, supercategory='whole_number')
        
        for whole_anns, measure_ann, whole_number_coco_image in zip(whole_anns_list, measure_ann_list, whole_number_coco_image_list):
            for whole_ann in whole_anns:
                self._process_single_digit_ann(
                    frame_img=frame_img, whole_number_coco_image=whole_number_coco_image,
                    whole_ann=whole_ann, measure_ann=measure_ann, whole_number_count=whole_number_count
                )
                whole_number_count += 1
        
        # Process Multiple Digit Cases
        part_anns_list = self._get_number_anns_list(frame_anns=frame_anns, measure_ann_list=measure_ann_list, orig_image=orig_image, supercategory='part_number')
        
        for part_anns, measure_ann, whole_number_coco_image in zip(part_anns_list, measure_ann_list, whole_number_coco_image_list):
            organized_parts = self._get_organized_parts(part_anns=part_anns)
        
            for organized_part in organized_parts:
                if len(organized_part['anns']) == 1:
                    logger.error(f"Found only 1 part for {organized_part['whole_name']}: {organized_part['part_names']}")
                    raise Exception
                self._process_organized_part(
                    organized_part=organized_part,
                    frame_img=frame_img, whole_number_coco_image=whole_number_coco_image,
                    measure_ann=measure_ann, whole_number_count=whole_number_count
                )
                whole_number_count += 1

    def _convert(self, measure_dir: str, whole_number_dir: str, digit_dir: str):
        for coco_image in self.images:
            self._convert_frame(orig_image=coco_image, whole_number_dir=whole_number_dir, digit_dir=digit_dir)
        self.measure_dataset.move_images(dst_img_dir=measure_dir, preserve_filenames=True, overwrite=True, show_pbar=True, update_img_paths=True)

    def split_measure_dataset(
        self, measure_dir: str='measure', whole_number_dir: str='whole_number', digit_dir: str='digit'
    ) -> (COCO_Dataset, COCO_Dataset, COCO_Dataset):
        self._prep_output_dir(measure_dir=measure_dir, whole_number_dir=whole_number_dir, digit_dir=digit_dir)
        self._load_licenses()
        self._load_categories()
        self._convert(
            measure_dir=measure_dir,
            whole_number_dir=whole_number_dir,
            digit_dir=digit_dir
        )
        return self.measure_dataset, self.whole_number_dataset, self.digit_dataset

# Load NDDS Dataset
ndds_dataset = NDDS_Dataset.load_from_dir(
    json_dir='/home/clayton/workspace/prj/data_keep/data/ndds/measure_5',
    show_pbar=True
)

# Fix NDDS Dataset naming so that it follows convention. (This is not necessary if the NDDS dataset already follows the naming convention.)
number_spelling_map = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
    'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9
}

for frame in ndds_dataset.frames:
    # Fix Naming Convention
    for ann_obj in frame.ndds_ann.objects:
        # Note: Part numbers should be specified in the obj_type string.
        if ann_obj.class_name == 'measure':
            obj_type, obj_name = 'seg', 'measure'
            ann_obj.class_name = f'{obj_type}_{obj_name}'
        elif ann_obj.class_name.startswith('num_'):
            temp = ann_obj.class_name.replace('num_', '')
            obj_type, obj_name, instance_name = 'seg', temp[:-1], temp[-1]
            obj_name = number_spelling_map[obj_name]
            ann_obj.class_name = f'{obj_type}_{obj_name}_{instance_name}'
        elif ann_obj.class_name == 'part_one0':
            obj_type, obj_name, instance_name = 'seg', '10part1', '0'
            ann_obj.class_name = f'{obj_type}_{obj_name}_{instance_name}'
        elif ann_obj.class_name == '10part_zero1':
            obj_type, obj_name, instance_name = 'seg', '10part0', '0'
            ann_obj.class_name = f'{obj_type}_{obj_name}_{instance_name}'
        elif ann_obj.class_name == 'part_two0':
            obj_type, obj_name, instance_name = 'seg', '20part2', '0'
            ann_obj.class_name = f'{obj_type}_{obj_name}_{instance_name}'
        elif ann_obj.class_name == '20part_zero2':
            obj_type, obj_name, instance_name = 'seg', '20part0', '0'
            ann_obj.class_name = f'{obj_type}_{obj_name}_{instance_name}'
        elif ann_obj.class_name == 'part_three0':
            obj_type, obj_name, instance_name = 'seg', '30part3', '0'
            ann_obj.class_name = f'{obj_type}_{obj_name}_{instance_name}'
        elif ann_obj.class_name == '30part_zero3':
            obj_type, obj_name, instance_name = 'seg', '30part0', '0'
            ann_obj.class_name = f'{obj_type}_{obj_name}_{instance_name}'
        elif ann_obj.class_name == 'part_four0':
            obj_type, obj_name, instance_name = 'seg', '40part4', '0'
            ann_obj.class_name = f'{obj_type}_{obj_name}_{instance_name}'
        elif ann_obj.class_name == '40part_zero4':
            obj_type, obj_name, instance_name = 'seg', '40part0', '0'
            ann_obj.class_name = f'{obj_type}_{obj_name}_{instance_name}'
        elif ann_obj.class_name == 'part_five0':
            obj_type, obj_name, instance_name = 'seg', '50part5', '0'
            ann_obj.class_name = f'{obj_type}_{obj_name}_{instance_name}'
        elif ann_obj.class_name == '50part_zero5':
            obj_type, obj_name, instance_name = 'seg', '50part0', '0'
            ann_obj.class_name = f'{obj_type}_{obj_name}_{instance_name}'
        elif ann_obj.class_name == 'part_six0':
            obj_type, obj_name, instance_name = 'seg', '60part6', '0'
            ann_obj.class_name = f'{obj_type}_{obj_name}_{instance_name}'
        elif ann_obj.class_name == '60part_zero6':
            obj_type, obj_name, instance_name = 'seg', '60part0', '0'
            ann_obj.class_name = f'{obj_type}_{obj_name}_{instance_name}'
        elif ann_obj.class_name == 'part_seven0':
            obj_type, obj_name, instance_name = 'seg', '70part7', '0'
            ann_obj.class_name = f'{obj_type}_{obj_name}_{instance_name}'
        elif ann_obj.class_name == '70part_zero7':
            obj_type, obj_name, instance_name = 'seg', '70part0', '0'
            ann_obj.class_name = f'{obj_type}_{obj_name}_{instance_name}'
        elif ann_obj.class_name == 'part_eight0':
            obj_type, obj_name, instance_name = 'seg', '80part8', '0'
            ann_obj.class_name = f'{obj_type}_{obj_name}_{instance_name}'
        elif ann_obj.class_name == '80part_zero8':
            obj_type, obj_name, instance_name = 'seg', '80part0', '0'
            ann_obj.class_name = f'{obj_type}_{obj_name}_{instance_name}'
        elif ann_obj.class_name == 'part_nine0':
            obj_type, obj_name, instance_name = 'seg', '90part9', '0'
            ann_obj.class_name = f'{obj_type}_{obj_name}_{instance_name}'
        elif ann_obj.class_name == '90part_zero9':
            obj_type, obj_name, instance_name = 'seg', '90part0', '0'
            ann_obj.class_name = f'{obj_type}_{obj_name}_{instance_name}'

# Convert To COCO Dataset
dataset = Measure_COCO_Dataset.from_ndds(
    ndds_dataset=ndds_dataset,
    categories=COCO_Category_Handler.load_from_path('/home/clayton/workspace/prj/data_keep/data/ndds/categories/measure_all.json'),
    naming_rule='type_object_instance_contained', delimiter='_',
    ignore_unspecified_categories=True,
    show_pbar=True
)

# Output Directories
measure_dir = 'measure'
whole_number_dir = 'whole_number'
digit_dir = 'digit'
json_output_filename = 'output.json'

measure_dataset, whole_number_dataset, digit_dataset = dataset.split_measure_dataset(
    measure_dir=measure_dir,
    whole_number_dir=whole_number_dir,
    digit_dir=digit_dir
)

measure_dataset.display_preview(show_details=True, window_name='Measure Dataset Preview')
measure_dataset.save_to_path(f'{measure_dir}/{json_output_filename}', overwrite=True)

whole_number_dataset.display_preview(show_details=True, window_name='Whole Number Dataset Preview')
whole_number_dataset.save_to_path(f'{whole_number_dir}/{json_output_filename}', overwrite=True)

if False: # For debugging 2-digit digit annotations
    del_ann_id_list = []
    for coco_image in digit_dataset.images:
        anns = digit_dataset.annotations.get_annotations_from_imgIds([coco_image.id])
        if len(anns) == 1:
            del_ann_id_list.append(anns[0].id)
    digit_dataset.annotations.remove(del_ann_id_list)
    digit_dataset.images.remove_if_no_anns(
        ann_handler=digit_dataset.annotations,
        license_handler=digit_dataset.licenses,
        verbose=True
    )

digit_dataset.display_preview(show_details=True, window_name='Digit Dataset Preview')
digit_dataset.save_to_path(f'{digit_dir}/{json_output_filename}', overwrite=True)