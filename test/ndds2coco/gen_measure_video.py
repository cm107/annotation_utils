from annotation_utils.coco.structs import COCO_Dataset

dataset = COCO_Dataset.load_from_path('measure_coco/measure/output.json')
dataset.save_video(
    save_path='merged_mask_measure_viz.mp4',
    show_details=True,
    fps=3
)