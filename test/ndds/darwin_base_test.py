import datetime
import json
from logger import logger

from annotation_utils.ndds.base import CocoNDDSConverter

# data location
data_dir = f"/home/clayton/workspace/prj/data_keep/data/ndds/HSR"

# save location
save_location = "hsr_coco/annot-coco.json"

# initiate coco info, license, and category
today = datetime.datetime.now().strftime("%Y/%m/%d")
year = datetime.datetime.now().strftime("%Y")

info_dict = {
    "description": "HSR 2020 Dataset",
    "url": "",
    "version": "1.0",
    "year": int(year),
    "contributor": "Pasonatech",
    "date_created": today
}
license_dict_list = [{
    "url": "",
    "id": 1,
    "name": "Private License"
}]
# change this to desired object
category_dict_list = [{
    "supercategory": "hsr",
    "id": 1,
    "name": "hsr",
    "keypoints": ["A","B","C","D","E","F","G","H","I","J","K","L"],
    "skeleton": [[1,2],[2,3],[3,4],[4,1],[1,5],[2,6],[3,7],[4,8],[5,6],[6,7],[7,8],[8,5],[5,9],[6,10],[7,11],[8,12],[9,10],[10,11],[11,12],[12,9]]
    }
    ]

coco_convert = CocoNDDSConverter(data_dir=data_dir, info_dict=info_dict, license_dict_list=license_dict_list, category_dict_list=category_dict_list, save_path=save_location)

# process data, return json and saved directory
coco_dataset, img_dir = coco_convert.process()
coco_dataset.save_to_path(save_path=save_location)

#  check saved annotation
json_dict_list = json.load(open(save_location, 'r'))
logger.yellow(json_dict_list)