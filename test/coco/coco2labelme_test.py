from annotation_utils.coco.structs import COCO_Dataset

dataset = COCO_Dataset.load_from_path(
    json_path='/home/clayton/workspace/prj/data_keep/data/toyota/dataset/sim/20200122/coco-data/new_HSR-coco.json',
    img_dir='/home/clayton/workspace/prj/data_keep/data/toyota/dataset/sim/20200122/coco-data'
)

for coco_ann in dataset.annotations:
    coco_image = dataset.images.get_images_from_imgIds([coco_ann.image_id])[0]
    coco_ann.bbox = coco_ann.bbox.scale_about_center(
        scale_factor=1.4,
        frame_shape=[coco_image.height, coco_image.width]
    )

# dataset.display_preview()
labelme_handler = dataset.to_labelme(priority='bbox')

labelme_handler.save_to_dir(
    json_save_dir='test_json',
    src_img_dir='/home/clayton/workspace/prj/data_keep/data/toyota/dataset/sim/20200122/coco-data',
    dst_img_dir='test_img'
)