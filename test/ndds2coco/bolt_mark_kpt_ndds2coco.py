from annotation_utils.ndds.structs import NDDS_Dataset
from annotation_utils.coco.structs import COCO_Category_Handler, COCO_Category
from annotation_utils.coco.structs import COCO_Dataset
from common_utils.file_utils import make_dir_if_not_exists, delete_all_files_in_dir
from typing import cast
from annotation_utils.coco.structs import COCO_Image, COCO_Annotation
from common_utils.path_utils import get_rootname_from_filename, get_extension_from_filename
from common_utils.common_types.point import Point2D
from tqdm import tqdm
import cv2
from logger import logger
from typing import List

target_src_dir = '/home/clayton/workspace/prj/data_keep/data/ndds/bolt_markMap_2020.08.18-13.03.35'
target_dst_dir = 'bolt_kpt'
make_dir_if_not_exists(target_dst_dir)
delete_all_files_in_dir(target_dst_dir)

# Load NDDS Dataset
logger.info('Loading NDDS Dataset')
ndds_dataset = NDDS_Dataset.load_from_dir(
    json_dir=target_src_dir,
    show_pbar=True
)
delete_idx_list = []


# Fix NDDS Dataset naming so that it follows convention. (This is not necessary if the NDDS dataset already follows the naming convention.)
for i, frame in enumerate(ndds_dataset.frames):
    for ann_obj in frame.ndds_ann.objects:
        if ann_obj.class_name.startswith('bolt'):
            if ann_obj.visibility == 0 :
                delete_idx_list.append(i)
            obj_type, obj_name = 'seg', 'bolt-roi'
            instance_name = ann_obj.class_name.replace('bolt', '')
            ann_obj.class_name = f'{obj_type}_{obj_name}_{instance_name}'
        elif ann_obj.class_name=='mark-inner':
            obj_type, obj_name = 'seg', 'mark-inner'
            instance_name = str(0)
            ann_obj.class_name = f'{obj_type}_{obj_name}_{instance_name}'
        elif ann_obj.class_name=='mark-middle':
            obj_type, obj_name = 'seg', 'mark-middle'
            instance_name = str(0)
            ann_obj.class_name = f'{obj_type}_{obj_name}_{instance_name}'
        elif ann_obj.class_name=='mark-outer':
            obj_type, obj_name = 'seg', 'mark-outer'
            instance_name = str(0)
            ann_obj.class_name = f'{obj_type}_{obj_name}_{instance_name}'
        # keypoints
        elif ann_obj.class_name.startswith('kpt-ia'):
            obj_type, obj_name = 'kpt', 'mark-inner'
            contained_name = 'ia'
            instance_name = str(0)
            ann_obj.class_name = f'{obj_type}_{obj_name}_{instance_name}_{contained_name}' 
        elif ann_obj.class_name.startswith('kpt-ib'):
            obj_type, obj_name = 'kpt', 'mark-inner'
            contained_name = 'ib'
            instance_name = str(0)
            ann_obj.class_name = f'{obj_type}_{obj_name}_{instance_name}_{contained_name}' 

        elif ann_obj.class_name.startswith('kpt-oa'):
            obj_type, obj_name = 'kpt', 'mark-outer'
            contained_name = 'oa'
            instance_name = str(0)
            ann_obj.class_name = f'{obj_type}_{obj_name}_{instance_name}_{contained_name}' 
        
        elif ann_obj.class_name.startswith('kpt-ob'):
            obj_type, obj_name = 'kpt', 'mark-outer'
            contained_name = 'ob'
            instance_name = str(0)
            ann_obj.class_name = f'{obj_type}_{obj_name}_{instance_name}_{contained_name}' 
            

        
for idx in delete_idx_list[::-1]:
    del ndds_dataset.frames[idx]
    print(f"ID deleted {idx}")

# Bolt ROI Dataset Creation
logger.info('Creating Bolt ROI Dataset')
bolt_roi_categories = COCO_Category_Handler()
print(f"Bolt_roi_categories :{bolt_roi_categories}")
bolt_roi_categories.append(
    COCO_Category(
        id=len(bolt_roi_categories),
        name='bolt-roi'
      
    )
)
print(f"Bolt_roi_categories :{bolt_roi_categories}")
bolt_roi_dataset = COCO_Dataset.from_ndds(
    ndds_dataset=ndds_dataset,
    categories=bolt_roi_categories,
    naming_rule='type_object_instance_contained', delimiter='_',
    ignore_unspecified_categories=True,
    show_pbar=True,
    bbox_area_threshold=1,
    default_visibility_threshold=0.01,
    visibility_threshold_dict={'bolt-roi': 0.01},
    allow_unfound_seg=False,
    class_merge_map={
        'seg_mark-inner_0': 'seg_bolt-roi_0',
        'seg_mark-middle_0': 'seg_bolt-roi_0',
        'seg_mark-outer_0': 'seg_bolt-roi_0'
    }
)
bolt_roi_dst_dir = f'{target_dst_dir}/bolt_roi'
make_dir_if_not_exists(bolt_roi_dst_dir)
bolt_roi_dataset.move_images(
    dst_img_dir=bolt_roi_dst_dir,
    preserve_filenames=True, overwrite_duplicates=False, update_img_paths=True, overwrite=True,
    show_pbar=True
)
bolt_roi_dataset.save_to_path(f'{bolt_roi_dst_dir}/output.json', overwrite=True)

#preview
# bolt_roi_dataset.display_preview(show_details=True, window_name='Bolt ROI')

# Mark (Not Cropped) Dataset Creation
logger.info('Creating Mark Dataset (Not Cropped Version)')
mark_categories = COCO_Category_Handler()
mark_categories.append(
    COCO_Category(
        id=len(mark_categories),
        name='mark-inner',
        keypoints=["ia","ib"],
        skeleton=[[0,1]]
    )
)
mark_categories.append(
    COCO_Category(
        id=len(mark_categories),
        name='mark-outer',
        keypoints=["oa","ob"],
        skeleton=[[0,1]]
    )
)
mark_dataset = COCO_Dataset.from_ndds(
    ndds_dataset=ndds_dataset,
    categories=mark_categories,
    naming_rule='type_object_instance_contained', delimiter='_',
    ignore_unspecified_categories=True,
    show_pbar=True,
    bbox_area_threshold=1,
    default_visibility_threshold=0.10,
    visibility_threshold_dict={'bolt-roi': 0.01},
    allow_unfound_seg=False,
    class_merge_map={
        'seg_mark-middle_0': 'seg_mark-inner_0',
        #'seg_mark_2': 'seg_mark_0'
    }
)
# mark_dataset.save_to_path('uncropped_mark.json', overwrite=True)
mark_dataset.display_preview(show_details=True, window_name='Mark')

# Mark (Cropped) Dataset Creation
logger.info('Creating Mark Dataset (Cropped Version)')
marker_dst_dir = f'{target_dst_dir}/marker'
make_dir_if_not_exists(marker_dst_dir)

cropped_mark_dataset = COCO_Dataset.new()
cropped_mark_dataset.categories = mark_categories.copy()
cropped_mark_dataset.licenses = mark_dataset.licenses.copy()

crop_pbar = tqdm(total=len(bolt_roi_dataset.images), unit='image(s)')
crop_pbar.set_description('Cropping')
for roi_image in bolt_roi_dataset.images:
    orig_img = cv2.imread(roi_image.coco_url)
    roi_anns = bolt_roi_dataset.annotations.get_annotations_from_imgIds([roi_image.id])
    mark_images = mark_dataset.images.get_images_from_file_name(roi_image.file_name)
    assert len(mark_images) == 1
    mark_image = mark_images[0]
    mark_anns = mark_dataset.annotations.get_annotations_from_imgIds(mark_image.id)
    #assert len(roi_anns) == len(mark_anns)

    for i, roi_ann in enumerate(roi_anns):
        roi_img = roi_ann.bbox.crop_from(orig_img)
        img_rootname = get_rootname_from_filename(roi_image.file_name)
        img_extension = get_extension_from_filename(roi_image.file_name)
        save_filename = f'{img_rootname}_{i}.{img_extension}'
        save_path = f'{marker_dst_dir}/{save_filename}'
        cv2.imwrite(save_path, roi_img)
        cropped_coco_image = COCO_Image.from_img_path(
            img_path=save_path,
            license_id=cropped_mark_dataset.licenses[0].id,
            image_id=len(cropped_mark_dataset.images)
        )
        cropped_mark_dataset.images.append(cropped_coco_image)

        mark_ann_found = False
        mark_ann_list = cast(List[COCO_Annotation], [])
        for i in list(range(len(mark_anns)))[::-1]:
            if roi_ann.bbox.contains(mark_anns[i].bbox):
                mark_ann_found = True
                # mark_ann = mark_anns[i].copy()
                mark_ann_list.append( mark_anns[i].copy())
                del mark_anns[i]
                
        if not mark_ann_found:
            raise Exception
        for mark_ann in mark_ann_list:
            mark_ann.segmentation = mark_ann.segmentation - Point2D(x=roi_ann.bbox.xmin, y=roi_ann.bbox.ymin)
            mark_ann.bbox = mark_ann.segmentation.to_bbox()
            mark_ann.keypoints = mark_ann.keypoints - Point2D(x=roi_ann.bbox.xmin, y=roi_ann.bbox.ymin)
            cropped_coco_ann = COCO_Annotation(
                id=len(cropped_mark_dataset.annotations),
                category_id=mark_ann.category_id,
                image_id=cropped_coco_image.id,
                segmentation=mark_ann.segmentation,
                bbox=mark_ann.bbox,
                area=mark_ann.bbox.area(),
                keypoints=mark_ann.keypoints,
                num_keypoints=len(mark_ann.keypoints),
                keypoints_3d=mark_ann.keypoints_3d
            )
            cropped_mark_dataset.annotations.append(cropped_coco_ann)
    crop_pbar.update()

cropped_mark_dataset.save_to_path(f'{marker_dst_dir}/output.json')
cropped_mark_dataset.display_preview(show_details=True)
cropped_mark_dataset.save_video(
    save_path=f'{marker_dst_dir}/preview.mp4',
    fps=5,
    show_details=True
)