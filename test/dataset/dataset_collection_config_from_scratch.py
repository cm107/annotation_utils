from annotation_utils.dataset.config import \
    DatasetConfigCollectionHandler, DatasetConfigCollection, DatasetConfig

handler = DatasetConfigCollectionHandler()
handler.append(
    DatasetConfigCollection(
        [
            DatasetConfig(
                img_dir='/a/b/c/d/img',
                ann_path='/a/b/ann/path.json',
                ann_format='coco'
            ),
            DatasetConfig(
                img_dir='/a/b/c/d/e/f/img',
                ann_path='/a/b/c/d/e/ann_folder/output.json',
                ann_format='custom'
            ),
            DatasetConfig(
                img_dir='/a/b/c/D/img',
                ann_path='/a/b/c/D/img/output.json'
            )
        ]
    )
)

handler.append(
    DatasetConfigCollection(
        [
            DatasetConfig(
                img_dir='/x/y/a/img',
                ann_path='/x/y/a/ann/output.json',
                ann_format='coco'
            ),
            DatasetConfig(
                img_dir='/x/y/b/img',
                ann_path='/x/y/b/ann/output.json',
                ann_format='coco'
            ),
            DatasetConfig(
                img_dir='/x/y/c/img',
                ann_path='/x/y/c/ann/output.json',
                ann_format='coco'
            ),
        ]
    )
)

handler.append(
    DatasetConfigCollection(
        [
            DatasetConfig(
                img_dir='/A/B/C/D/E/F/G/img',
                ann_path='/A/B/ann/path.json',
                ann_format='coco'
            ),
            DatasetConfig(
                img_dir='/A/B/C/D/E/F/img',
                ann_path='/A/B/C/D/E/ann_folder/output.json',
                ann_format='custom'
            ),
            DatasetConfig(
                img_dir='/A/B/C/D/E/img',
                ann_path='/A/B/C/D/E/img/output.json'
            )
        ]
    )
)

handler.save_to_path('scratch_config.yaml', overwrite=True)