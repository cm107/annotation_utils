from common_utils.file_utils import make_dir_if_not_exists
from annotation_utils.dataset.config import DatasetConfigCollectionHandler, DatasetConfigCollection, DatasetConfig

handler = DatasetConfigCollectionHandler()
data_dir = '/home/user/arbitrary/data/dir'

handler.append(
    DatasetConfigCollection(
        [
            DatasetConfig(
                img_dir=f'{data_dir}/real/set0/img',
                ann_path=f'{data_dir}/real/set0/coco/output.json',
                ann_format='coco',
                tag='old'
            ),
            DatasetConfig(
                img_dir=f'{data_dir}/real/set1/img',
                ann_path=f'{data_dir}/real/set1/coco/output.json',
                ann_format='coco',
                tag='new'
            ),
            DatasetConfig(
                img_dir=f'{data_dir}/real/set2/img',
                ann_path=f'{data_dir}/real/set2/coco/output.json',
                ann_format='coco',
                tag='new'
            ),
            DatasetConfig(
                img_dir=f'{data_dir}/real/set3/img',
                ann_path=f'{data_dir}/real/set3/coco/output.json',
                ann_format='coco',
                tag='priority'
            ),
            DatasetConfig(
                img_dir=f'{data_dir}/real/set4/img',
                ann_path=f'{data_dir}/real/set4/coco/output.json',
                ann_format='coco',
                tag='obsolete'
            )
        ],
        tag='real'
    )
)

handler.append(
    DatasetConfigCollection(
        [
            DatasetConfig(
                img_dir=f'{data_dir}/sim/set0/img',
                ann_path=f'{data_dir}/sim/set0/coco/output.json',
                ann_format='coco',
                tag='old'
            ),
            DatasetConfig(
                img_dir=f'{data_dir}/sim/set1/img',
                ann_path=f'{data_dir}/sim/set1/coco/output.json',
                ann_format='coco',
                tag='invalid'
            ),
            DatasetConfig(
                img_dir=f'{data_dir}/sim/set2/img',
                ann_path=f'{data_dir}/sim/set2/coco/output.json',
                ann_format='coco',
                tag='invalid'
            ),
            DatasetConfig(
                img_dir=f'{data_dir}/sim/set3/img',
                ann_path=f'{data_dir}/sim/set3/coco/output.json',
                ann_format='coco'
            ),
            DatasetConfig(
                img_dir=f'{data_dir}/sim/set4/img',
                ann_path=f'{data_dir}/sim/set4/coco/output.json',
                ann_format='coco'
            ),
            DatasetConfig(
                img_dir=f'{data_dir}/sim/set5/img',
                ann_path=f'{data_dir}/sim/set5/coco/output.json',
                ann_format='coco',
                tag='valid'
            ),
            DatasetConfig(
                img_dir=f'{data_dir}/sim/set6/img',
                ann_path=f'{data_dir}/sim/set6/coco/output.json',
                ann_format='coco',
                tag='obsolete'
            ),
            DatasetConfig(
                img_dir=f'{data_dir}/sim/set7/img',
                ann_path=f'{data_dir}/sim/set7/coco/output.json',
                ann_format='coco',
                tag='new'
            )
        ],
        tag='sim'
    )
)

sim_handler = handler.filter_by_collection_tag(['sim'])
real_handler = handler.filter_by_collection_tag('real')
used_handler = handler.filter_by_dataset_tag(['new', 'valid', 'priority'])
unused_handler = handler.filter_by_dataset_tag([None, 'invalid', 'obsolete', 'old'])
untagged_handler = handler.filter_by_dataset_tag(None)

config_dump_dir = 'config_dump'
make_dir_if_not_exists(config_dump_dir)
sim_handler.save_to_path(f'{config_dump_dir}/sim_datasets.json', overwrite=True)
real_handler.save_to_path(f'{config_dump_dir}/real_datasets.json', overwrite=True)
used_handler.save_to_path(f'{config_dump_dir}/used_datasets.json', overwrite=True)
unused_handler.save_to_path(f'{config_dump_dir}/unused_datasets.json', overwrite=True)
untagged_handler.save_to_path(f'{config_dump_dir}/untagged_datasets.json', overwrite=True)