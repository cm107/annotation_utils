from annotation_utils.coco.structs import COCO_Dataset

dataset = COCO_Dataset.load_from_path('measure_total_test.json')

dataset.remove_all_categories_except(['measure'])

dataset.display_preview(
    show_details=True
)