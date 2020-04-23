from __future__ import annotations
from typing import List, Generic, TypeVar
import json
import os
from pathlib import Path

T = TypeVar('T')

class CocoBase():
    def __init__(self, *args,**kwargs):
        pass
    
    def check_data_validity(self):
        pass

    def to_dict(self):
        return self.__dict__

    
    def from_dict(self, coco_dict: dict):
        for k, v in coco_dict.items():
            setattr(self, k, v)
        return self

    def save_dict_to_path(self, path: str):
        json_dict = self.to_dict()
        json.dump(json_dict, open(path, 'w'), indent=2, ensure_ascii=False)

    def __repr__(self):
        return f'{str(self.__dict__)}'
    

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)


class CocoInfo(CocoBase):

    def __init__(
        self, 
        description: str = "",
        url: str = "",
        version: str = "",
        year: int = 0,
        contributor: str = "",
        date_created: str = ""
    ):
        self.description = description
        self.url = url
        self.version = version
        self.year = year
        self.contributor = contributor
        self.date_created = date_created

    def __call__(
        self, 
        description: str,
        url: str,
        version: str,
        year: int,
        contributor: str,
        date_created: str
    ):
        self.description = description
        self.url = url
        self.version = version
        self.year = year
        self.contributor = contributor
        self.date_created = date_created

    def __str__(self):
        return str(self.__dict__)
    



    # def from_dict(self):
class CocoLicense(CocoBase):


    def __init__(
        self,
        url: str = "",
        id: int = 0,
        name: str = ""
    ):
        self.url = url
        self.id = id
        self.name = name

    def __call__(
        self,
        url: str,
        id: int,
        name: str
    ):
        self.url = url
        self.id = id
        self.name = name
    

    def __str__(self):
        return str(self.__dict__)

class CocoImage(CocoBase):


    def __init__(
        self,
        license: int = 0,
        file_name: str = "",
        coco_url: str = "",
        height: int = 0,
        width: int = 0,
        date_captured: str = "",
        flickr_url: str = None,
        id: int= 0
    ):
        self.license = license
        self.file_name = file_name
        self.coco_url = coco_url
        self.height = height
        self.width = width
        self.date_captured = date_captured
        self.flickr_url = flickr_url
        self.id = id

    def __call__(
        self,
        license: int,
        file_name: str,
        coco_url: str,
        height: int,
        width: int,
        date_captured: str,
        flickr_url: None,
        id: int
    ):
        self.license = license
        self.file_name = file_name
        self.coco_url = coco_url
        self.height = height
        self.width = width
        self.date_captured = date_captured
        self.flickr_url = flickr_url
        self.id = id
    
    def __str__(self):
        return str(self.__dict__)

class CocoAnnotation(CocoBase):

    def __init__(
        self,
        bbox: List[float] = [0,0,0,0],
        image_id: int = 0,
        category_id: int = 0,
        iscrowd: int = 0,
        id: int = 0,
        *args,
        **kwargs
    ):
        self.bbox = bbox
        self.image_id = image_id
        self.category_id = category_id
        self.iscrowd = iscrowd
        self.id = id

        for key, value in kwargs.items():
            setattr(self, key, value)
        self.check_data_validity()

    def check_data_validity(self):
        if "keypoints" in self.__dict__.keys():
            if len(self.keypoints) % 3 != 0 or min(self.keypoints) < 0:
                raise TypeError("Keypoints is not valid") 
            if max(self.keypoints[2::3]) > 2 or max(self.keypoints[2::3]) < 0:
                raise TypeError("Keypoints visibility is not valid") 
        if "keypoints_3d" in self.__dict__.keys():
            if len(self.keypoints_3d) % 4 != 0:
                print("here")
                raise TypeError("Keypoints_3d is not valid") 
            if max(self.keypoints_3d[3::4]) > 2 or max(self.keypoints_3d[3::4]) < 0:
                raise TypeError("Keypoints 3d visibility is not valid") 


    def __call__(
        self,
        bbox: List[float],
        image_id: int,
        category_id: int,
        iscrowd: int,
        id: int,
        *args,
        **kwargs
    ):
        self.bbox = bbox
        self.image_id = image_id
        self.category_id = category_id
        self.iscrowd = iscrowd
        self.id = id

        for key, value in kwargs.items():
            setattr(self, key, value)
        
        self.check_data_validity()

    def from_dict(self, coco_dict: dict):
        for k, v in coco_dict.items():
            setattr(self, k, v)
        self.check_data_validity()
        return self

    def __str__(self):
        return str(self.__dict__)


class CocoCategory(CocoBase):
    def __init__(
        self,
        supercategory: str = "",
        id: int = 0,
        name: str = "",
        **kwargs
    ):
        self.supercategory = supercategory
        self.id = id
        self.name = name
        if "keypoints" in kwargs:
            self.keypoints = kwargs["keypoints"]
        if "skeleton" in kwargs:
            self.skeleton = kwargs["skeleton"]

    def __call__(
        self,
        supercategory: str,
        id: int,
        name: str,
        **kwargs
    ):
        self.supercategory = supercategory
        self.id = id
        self.name = name
        if "keypoints" in kwargs:
            self.keypoints = kwargs["keypoints"]
        if "skeleton" in kwargs:
            self.skeleton = kwargs["skeleton"]

    def __str__(self):
        return str(self.__dict__)

class CocoDataset:

    def __init__(
        self,
        info: CocoInfo = CocoInfo(),
        licenses: List[CocoLicense] = [CocoLicense()],
        images: List[CocoImage] = [CocoImage()],
        annotations: List[CocoAnnotation] = [CocoAnnotation()],
        categories: List[CocoCategory] = [CocoCategory()]
    ):
        self.info = info
        self.licenses = licenses
        self.images = images
        self.annotations = annotations
        self.categories = categories

    def __call__(
        self,
        info: CocoInfo,
        licenses: List[CocoLicense],
        images: List[CocoImage],
        annotations: List[CocoAnnotation],
        categories: List[CocoCategory]
    ):
        self.info = info
        self.licenses = licenses
        self.images = images
        self.annotations = annotations
        self.categories = categories

    def __str__(self):
        return str(self.__dict__)

    @classmethod
    def load_from_dict_path(self, path: str) -> CocoDataset:
        json_dict_list = json.load(open(path, 'r'))
        info = json_dict_list["info"]
        licenses = json_dict_list["licenses"]
        images = json_dict_list["images"]
        annotations = json_dict_list["annotations"]
        categories = json_dict_list["categories"]
        info = CocoInfo().from_dict(coco_dict = info)
        licenses = [CocoLicense().from_dict(coco_dict=license) for license in licenses]
        images = [CocoImage().from_dict(coco_dict=image) for image in images]
        annotations = [CocoAnnotation().from_dict(coco_dict=annotation) for annotation in annotations]
        categories = [CocoCategory().from_dict(coco_dict=category) for category in categories]
        
        return CocoDataset(info=info, licenses=licenses, images=images, annotations=annotations, categories=categories)

    def save_to_path(self, save_path: str = './annot-coco.json'):
        json_string = str(self.__dict__).replace("\'", "\"")
        json_string = str(json_string).replace("None", "null")
        json_dict = json.loads(str(json_string))
        save_path_dir = os.path.dirname(os.path.abspath(save_path))
        Path(save_path_dir).mkdir(parents=True, exist_ok=True)

        with open(save_path, 'w') as fp:
            fp.write(json.dumps(json_dict, indent=4))

    # need to fix this method
    def __add__(self, other):
        if self.info != other.info:
            raise TypeError("Data set doesn't have the same info")
        
        self.licenses += other.licenses
        self.images  += other.images
        self.annotations += other.annotations
        self.categories += other.categories

        return self




if __name__ == "__main__":
    a = CocoDataset.load_from_dict_path(path='/Users/darwinharianto/Desktop/hayashida/Unreal/03_04_2020_15_31_01_coco-data/HSR-coco.json')

    a.save_to_path(save_path="./annot-coco.json")
    # print(coco_dataset + coco_dataset_other)

