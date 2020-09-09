from annotation_utils.coco.structs import COCO_Dataset

dataset = COCO_Dataset.load_from_path(
    '/home/clayton/workspace/prj/data_keep/data/dataset/bird/img/output.json',
    img_dir='/home/clayton/workspace/prj/data_keep/data/dataset/bird/img'
)

# split_image_handlers = dataset.images.split(ratio=[1, 3, 1], shuffle=True)
# for image_handler in split_image_handlers:
#     print(f'len(image_handler): {len(image_handler)}')
#     print(f'\tfilenames: {[coco_image.file_name for coco_image in image_handler]}')

parts = dataset.split_into_parts(ratio=[1,2,3], shuffle=True)
part_count = -1
for part in parts:
    part_count += 1
    print(f'len(part.images): {len(part.images)}')
    part.save_video(save_path=f'part{part_count}.avi', fps=5, show_details=True)