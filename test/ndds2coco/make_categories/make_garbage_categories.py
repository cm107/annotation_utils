from annotation_utils.coco.structs import COCO_Category_Handler, COCO_Category

handler = COCO_Category_Handler()
handler.append(
    COCO_Category(
        id=len(handler),
        name='garbage'
    )
)

output_dir = '/home/clayton/workspace/prj/data_keep/data/ndds/categories'
handler.save_to_path(f'{output_dir}/garbage.json', overwrite=True)