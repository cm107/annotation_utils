from annotation_utils.coco.structs import COCO_Category_Handler, COCO_Category

handler_simple = COCO_Category_Handler()
handler = COCO_Category_Handler()

for name in ['measure', 'ticks-top', 'ticks-bottom', '10s-rectangles', 'hook']:
    category = COCO_Category(
        id=len(handler),
        supercategory='measure',
        name=name
    )
    if name in ['measure']:
        handler_simple.append(category)
    handler.append(category)

for i in range(10):
    category = COCO_Category(
        id=len(handler),
        supercategory='whole_number',
        name=str(i)
    )
    handler_simple.append(category)
    handler.append(category)

for i in range(1, 10):
    category_part0 = COCO_Category(
        id=len(handler),
        supercategory='part_number',
        name=f'{i}0part{i}'
    )
    category_part1 = COCO_Category(
        id=len(handler),
        supercategory='part_number',
        name=f'{i}0part0'
    )
    handler_simple.append(category_part0)
    handler_simple.append(category_part1)
    handler.append(category_part0)
    handler.append(category_part1)


output_dir = '/home/clayton/workspace/prj/data_keep/data/ndds/categories'
handler_simple.save_to_path(f'{output_dir}/measure_all.json', overwrite=True)
handler.save_to_path(f'{output_dir}/measure_all_adv.json', overwrite=True)