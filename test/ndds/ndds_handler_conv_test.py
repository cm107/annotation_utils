from logger import logger
from annotation_utils.ndds.structs.frame import NDDS_Frame_Handler

json_dir = '/home/clayton/workspace/prj/data_keep/data/ndds/HSR'
img_dir = json_dir

handler = NDDS_Frame_Handler.load_from_dir(img_dir=img_dir, json_dir=json_dir)
for frame in handler:
    # logger.cyan(frame.img_path)
    # logger.purple([ann_obj.class_name for ann_obj in frame.ndds_ann.objects])
    for ann_obj in frame.ndds_ann.objects:
        if ann_obj.class_name.startswith('hsr'):
            # obj_type, obj_name, instance_name, contained_name
            obj_type, obj_name = 'seg', 'hsr'
            instance_name = ann_obj.class_name.replace('hsr', '')
            ann_obj.class_name = f'{obj_type}_{obj_name}_{instance_name}'
        elif ann_obj.class_name.startswith('point'):
            obj_type, obj_name = 'kpt', 'hsr'
            temp = ann_obj.class_name.replace('point', '')
            instance_name, contained_name = temp[1], temp[0]
            ann_obj.class_name = f'{obj_type}_{obj_name}_{instance_name}_{contained_name}'
        else:
            logger.error(f'ann_obj.class_name: {ann_obj.class_name}')
            raise Exception
    del_idx_list = []
    for i in range(len(frame.ndds_ann.objects)):
        for j in range(i+1, len(frame.ndds_ann.objects)):
            if frame.ndds_ann.objects[i].class_name == frame.ndds_ann.objects[j].class_name:
                if j not in del_idx_list:
                    del_idx_list.append(j)
    for i in del_idx_list:
        logger.info(f'frame.img_path: {frame.img_path}')
        logger.info(f'Deleted duplicate of {frame.ndds_ann.objects[i].class_name}')
        del frame.ndds_ann.objects[i]

# for frame in handler:
#     logger.cyan(frame.img_path)
#     logger.purple([ann_obj.class_name for ann_obj in frame.ndds_ann.objects])

# import sys
# sys.exit()

for frame in handler:
    logger.green(f'frame.img_path: {frame.img_path}')
    class_names = [ann_obj.class_name for ann_obj in frame.ndds_ann.objects]
    
    labeled_obj_handler = frame.to_labeled_obj_handler()
    # logger.purple(f'labeled_obj_handler:\n{labeled_obj_handler}')
    logger.yellow(f'len(labeled_obj_handler): {len(labeled_obj_handler)}')
    for labeled_obj in labeled_obj_handler:
        logger.cyan(f'len(labeled_obj.instances): {len(labeled_obj.instances)}')
        for instance in labeled_obj.instances:
            logger.blue(f'len(instance.contained_instance_list): {len(instance.contained_instance_list)}')
            for contained_instance in instance.contained_instance_list:
                logger.white(f'{contained_instance.instance_type}: {contained_instance.instance_name} ')