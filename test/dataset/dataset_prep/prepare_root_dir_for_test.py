from common_utils.file_utils import make_dir_if_not_exists
from annotation_utils.coco.structs import COCO_Dataset

src_img_dir = '/home/clayton/workspace/prj/data_keep/data/dataset/bird/img'
dst_dataset_root_dir = '/home/clayton/workspace/prj/data_keep/data/dataset/bird/dataset_root_dir'
make_dir_if_not_exists(dst_dataset_root_dir)

dataset = COCO_Dataset.load_from_path( # 18 images -> 2 scenarios x 3 datasets / scenario x 3 images / dataset -> 2 train, 1 val
    json_path=f'{src_img_dir}/output.json',
    img_dir=src_img_dir
)
scenario_names = [f'scenario{i}' for i in range(2)]
scenario_datasets = dataset.split_into_parts(ratio=[9, 9], shuffle=True)
for i in range(len(scenario_datasets)):
    scenario_name = f'scenario{i}'
    dst_scenario_dir = f'{dst_dataset_root_dir}/{scenario_name}'
    make_dir_if_not_exists(dst_scenario_dir)
    part_datasets = scenario_datasets[i].split_into_parts(ratio=[3, 3, 3], shuffle=True)
    for j in range(len(part_datasets)):
        part_name = f'part{j}'
        dst_part_dir = f'{dst_scenario_dir}/{part_name}'
        make_dir_if_not_exists(dst_part_dir)
        part_datasets[j].move_images(
            dst_img_dir=dst_part_dir,
            preserve_filenames=True,
            update_img_paths=True,
            overwrite=True,
            show_pbar=False
        )
        part_datasets[j].save_to_path(
            save_path=f'{dst_part_dir}/output.json',
            overwrite=True
        )
        part_datasets[j].save_video(
            save_path=f'{dst_part_dir}/preview.avi',
            fps=5,
            overwrite=True,
            show_details=True
        )
        print(len(part_datasets[j].images))