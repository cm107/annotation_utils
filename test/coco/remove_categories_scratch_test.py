from annotation_utils.coco.structs import COCO_Dataset, \
    COCO_License, COCO_Image, COCO_Annotation, COCO_Category
from common_utils.common_types.bbox import BBox
from logger import logger

dataset = COCO_Dataset.new(description='Test')
dataset.categories.append(
    COCO_Category(
        id=len(dataset.categories),
        supercategory='test_category',
        name='category_a'
    )
)
dataset.categories.append(
    COCO_Category(
        id=len(dataset.categories),
        supercategory='test_category',
        name='category_b'
    )
)
dataset.categories.append(
    COCO_Category(
        id=len(dataset.categories),
        supercategory='test_category',
        name='category_c'
    )
)

for i in range(10):
    dataset.licenses.append(
        COCO_License(url=f'test_license_{i}', name=f'Test License {i}', id=len(dataset.licenses))
    )
for i in range(20):
    dataset.images.append(
        COCO_Image(
            license_id=i%len(dataset.licenses),
            file_name=f'{i}.jpg',
            coco_url=f'/path/to/{i}.jpg',
            height=500,
            width=500,
            date_captured='N/A',
            flickr_url=None,
            id=len(dataset.images)
        )
    )
for i in range(len(dataset.images)):
    if i % 2 == 0:
        dataset.annotations.append(
            COCO_Annotation(
                category_id=dataset.categories.get_unique_category_from_name('category_a').id,
                image_id=dataset.images[i].id,
                bbox=BBox(xmin=0, ymin=0, xmax=100, ymax=100),
                id=len(dataset.annotations)
            )
        )
        dataset.annotations.append(
            COCO_Annotation(
                category_id=dataset.categories.get_unique_category_from_name('category_b').id,
                image_id=dataset.images[i].id,
                bbox=BBox(xmin=0, ymin=0, xmax=100, ymax=100),
                id=len(dataset.annotations)
            )
        )
    else:
        dataset.annotations.append(
            COCO_Annotation(
                category_id=dataset.categories.get_unique_category_from_name('category_b').id,
                image_id=dataset.images[i].id,
                bbox=BBox(xmin=0, ymin=0, xmax=100, ymax=100),
                id=len(dataset.annotations)
            )
        )
        dataset.annotations.append(
            COCO_Annotation(
                category_id=dataset.categories.get_unique_category_from_name('category_c').id,
                image_id=dataset.images[i].id,
                bbox=BBox(xmin=0, ymin=0, xmax=100, ymax=100),
                id=len(dataset.annotations)
            )
        )

logger.purple('Before:')
dataset.print_handler_lengths()

dataset.remove_categories_by_name(category_names=['category_b', 'category_c'], verbose=True)

logger.purple('After:')
dataset.print_handler_lengths()

dataset.save_to_path(save_path='remove_test.json', overwrite=True)