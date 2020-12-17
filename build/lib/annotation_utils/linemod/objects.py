from __future__ import annotations
from typing import List, Tuple
import numpy as np
from tqdm import tqdm
from common_utils.file_utils import file_exists, dir_exists, \
    copy_file, create_softlink, make_dir_if_not_exists, \
    delete_all_files_in_dir
from common_utils.path_utils import get_filename, get_extension_from_filename, \
    rel_to_abs_path, get_rootname_from_path, get_dirpath_from_filepath
from common_utils.base.basic import BasicLoadableIdObject, BasicLoadableObject, BasicLoadableIdHandler, BasicHandler
from common_utils.common_types.point import Point2D, Point3D, Point2D_List, Point3D_List
from common_utils.common_types.angle import QuaternionList
from common_utils.common_types.segmentation import Segmentation
from common_utils.common_types.bbox import BBox
from common_utils.common_types.keypoint import Keypoint2D_List, Keypoint3D_List
from common_utils.time_utils import get_ctime
from ..coco.structs.dataset import COCO_Dataset
from ..coco.structs.objects import COCO_Image, COCO_Annotation, COCO_Category, COCO_License
from ..coco.camera import Camera as COCO_Camera

class LinemodCamera(BasicLoadableObject['LinemodCamera']):
    def __init__(self, fx: float, fy: float, cx: float, cy: float):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
    
    def to_matrix(self) -> np.ndarray:
        return np.array(
            [
                [self.fx, 0, self.cx],
                [0, self.fy, self.cy],
                [0, 0, 1]
            ]
        )
    
    @classmethod
    def from_matrix(self, arr: np.ndarray) -> LinemodCamera:
        assert arr.shape == (3, 3)
        assert arr[0, 1] == 0
        assert arr[1, 0] == 0
        assert arr[2, 0] == 0
        assert arr[2, 1] == 0
        assert arr[2, 2] == 1
        return LinemodCamera(
            fx=arr[0,0],
            fy=arr[1, 1],
            cx=arr[0, 2],
            cy=arr[1, 2]
        )
    
    @classmethod
    def from_image_shape(self, image_shape: List[int]) -> LinemodCamera:
        height, width = image_shape[:2]
        return LinemodCamera(
            fx=max([width, height])/2,
            fy=max([width, height])/2,
            cx=width/2,
            cy=height/2
        )
    
    def save_to_txt(self, save_path: str, overwrite: bool=False):
        if file_exists(save_path) and not overwrite:
            raise FileExistsError(
                f"""
                File already exists at {save_path}
                Hint: Use overwrite=True to save anyway.
                """
            )
        np.savetxt(fname=save_path, X=self.to_matrix())

    @classmethod
    def load_from_txt(self, load_path: str) -> LinemodCamera:
        if not file_exists(load_path):
            raise FileNotFoundError(f"Couldn't find file at {load_path}")
        mat = np.loadtxt(load_path)
        return LinemodCamera.from_matrix(mat)

class Linemod_Image(
    BasicLoadableIdObject['Linemod_Image'],
    BasicLoadableObject['Linemod_Image']
):
    def __init__(self, file_name: str, width: int, height: int, id: int):
        super().__init__(id=id)
        self.file_name = file_name # Note: This can also be a path.
        self.width = width
        self.height = height

class Linemod_Image_Handler(
    BasicLoadableIdHandler['Linemod_Image_Handler', 'Linemod_Image'],
    BasicHandler['Linemod_Image_Handler', 'Linemod_Image']
):
    def __init__(self, images: List[Linemod_Image]=None):
        super().__init__(obj_type=Linemod_Image, obj_list=images)
        self.images = self.obj_list
    
    @classmethod
    def from_dict_list(cls, dict_list: List[dict]) -> Linemod_Image_Handler:
        return Linemod_Image_Handler([Linemod_Image.from_dict(item_dict) for item_dict in dict_list])

class Linemod_Annotation(
    BasicLoadableIdObject['Linemod_Annotation'],
    BasicLoadableObject['Linemod_Annotation'],
):
    def __init__(
        self,
        data_root: str,
        mask_path: str,
        type: str, class_name: str,
        corner_2d: Point2D_List, corner_3d: Point3D_List,
        center_2d: Point2D, center_3d: Point3D,
        fps_2d: Point2D_List, fps_3d: Point3D_List,
        K: LinemodCamera,
        pose: QuaternionList, # Is this a list of quaternions? Not sure.
        image_id: int, category_id: int, id: int,
        depth_path: str=None
    ):
        super().__init__(id=id)
        self.data_root = data_root
        self.mask_path = mask_path
        self.depth_path = depth_path
        self.type = type
        self.class_name = class_name
        self.corner_2d = corner_2d
        self.corner_3d = corner_3d
        self.center_2d = center_2d
        self.center_3d = center_3d
        self.fps_2d = fps_2d
        self.fps_3d = fps_3d
        self.K = K
        self.pose = pose
        self.image_id = image_id
        self.category_id = category_id

    def to_dict(self) -> dict:
        result = {
            'data_root': self.data_root,
            'mask_path': self.mask_path,
            'type': self.type,
            'cls': self.class_name,
            'corner_2d': self.corner_2d.to_list(demarcation=True),
            'corner_3d': self.corner_3d.to_list(demarcation=True),
            'center_2d': self.center_2d.to_list(),
            'center_3d': self.center_3d.to_list(),
            'fps_2d': self.fps_2d.to_list(demarcation=True),
            'fps_3d': self.fps_3d.to_list(demarcation=True),
            'K': self.K.to_matrix().tolist(),
            'pose': self.pose.to_list(),
            'image_id': self.image_id,
            'category_id': self.category_id,
            'id': self.id
        }
        if self.depth_path is not None:
            result['depth_path'] = self.depth_path
        return result
    
    @classmethod
    def from_dict(cls, item_dict: dict) -> Linemod_Annotation:
        return Linemod_Annotation(
            data_root=item_dict['data_root'],
            mask_path=item_dict['mask_path'],
            depth_path=item_dict['depth_path'] if 'depth_path' in item_dict else None,
            type=item_dict['type'],
            class_name=item_dict['cls'],
            corner_2d=Point2D_List.from_list(item_dict['corner_2d'], demarcation=True),
            corner_3d=Point3D_List.from_list(item_dict['corner_3d'], demarcation=True),
            center_2d=Point2D.from_list(item_dict['center_2d']),
            center_3d=Point3D.from_list(item_dict['center_3d']),
            fps_2d=Point2D_List.from_list(item_dict['fps_2d'], demarcation=True),
            fps_3d=Point3D_List.from_list(item_dict['fps_3d'], demarcation=True),
            K=LinemodCamera.from_matrix(np.array(item_dict['K'])),
            pose=QuaternionList.from_list(item_dict['pose']),
            image_id=item_dict['image_id'],
            category_id=item_dict['category_id'],
            id=item_dict['id']
        )

class Linemod_Annotation_Handler(
    BasicLoadableIdHandler['Linemod_Annotation_Handler', 'Linemod_Annotation'],
    BasicHandler['Linemod_Annotation_Handler', 'Linemod_Annotation']
):
    def __init__(self, images: List[Linemod_Annotation]=None):
        super().__init__(obj_type=Linemod_Annotation, obj_list=images)
        self.images = self.obj_list

    @classmethod
    def from_dict_list(cls, dict_list: List[dict]) -> Linemod_Annotation_Handler:
        return Linemod_Annotation_Handler([Linemod_Annotation.from_dict(item_dict) for item_dict in dict_list])

    def get(
        self, type: str=None, class_name: str=None,
        image_id: int=None, category_id: int=None, id: int=None
    ) -> Linemod_Annotation_Handler:
        return Linemod_Annotation_Handler(
            [
                ann for ann in self \
                    if (type is None or (type is not None and ann.type == type)) and \
                        (class_name is None or (class_name is not None and ann.class_name == class_name)) and \
                        (image_id is None or (image_id is not None and ann.image_id == image_id)) and \
                        (category_id is None or (category_id is not None and ann.category_id == category_id)) and \
                        (id is None or (id is not None and ann.id == id))
            ]
        )

class Linemod_Category(
    BasicLoadableIdObject['Linemod_Category'],
    BasicLoadableObject['Linemod_Category']
):
    def __init__(self, supercategory: str, name: str, id: int):
        super().__init__(id=id)
        self.supercategory = supercategory
        self.name = name

class Linemod_Category_Handler(
    BasicLoadableIdHandler['Linemod_Category_Handler', 'Linemod_Category'],
    BasicHandler['Linemod_Category_Handler', 'Linemod_Category']
):
    def __init__(self, images: List[Linemod_Category]=None):
        super().__init__(obj_type=Linemod_Category, obj_list=images)
        self.images = self.obj_list
    
    @classmethod
    def from_dict_list(cls, dict_list: List[dict]) -> Linemod_Category_Handler:
        return Linemod_Category_Handler([Linemod_Category.from_dict(item_dict) for item_dict in dict_list])

class Linemod_Dataset(BasicLoadableObject['Linemod_Dataset']):
    def __init__(
        self,
        images: Linemod_Image_Handler=None,
        annotations: Linemod_Annotation_Handler=None,
        categories: Linemod_Category_Handler=None
    ):
        super().__init__()
        self.images = images if images is not None else Linemod_Image_Handler()
        self.annotations = annotations if annotations is not None else Linemod_Annotation_Handler()
        self.categories = categories if categories is not None else Linemod_Category_Handler()
    
    def to_dict(self) -> dict:
        return {
            'images': self.images.to_dict_list(),
            'annotations': self.annotations.to_dict_list(),
            'categories': self.categories.to_dict_list()
        }
    
    @classmethod
    def from_dict(cls, item_dict: dict) -> Linemod_Dataset:
        return Linemod_Dataset(
            images=Linemod_Image_Handler.from_dict_list(item_dict['images']),
            annotations=Linemod_Annotation_Handler.from_dict_list(item_dict['annotations']),
            categories=Linemod_Category_Handler.from_dict_list(item_dict['categories'])
        )
    
    def to_coco(
        self, img_dir: str=None, mask_dir: str=None, coco_license: COCO_License=None, check_paths: bool=True,
        mask_lower_bgr: Tuple[int]=None, mask_upper_bgr: Tuple[int]=(255,255,255),
        show_pbar: bool=True
    ) -> COCO_Dataset:
        dataset = COCO_Dataset.new(description='Dataset converted from Linemod to COCO format.')
        dataset.licenses.append(
            coco_license if coco_license is not None else COCO_License(
                url='https://github.com/cm107/annotation_utils/blob/master/LICENSE',
                name='MIT License',
                id=len(dataset.licenses)
            )
        )
        coco_license0 = dataset.licenses[-1]
        for linemod_image in self.images:
            file_name = get_filename(linemod_image.file_name)
            img_path = linemod_image.file_name if img_dir is None else f'{img_dir}/{file_name}'
            if file_exists(img_path):
                date_captured = get_ctime(img_path)
            else:
                if check_paths:
                    raise FileNotFoundError(f"Couldn't find image at {img_path}")
                date_captured = ''
            coco_image = COCO_Image(
                license_id=coco_license0.id,
                file_name=file_name,
                coco_url=img_path,
                width=linemod_image.width, height=linemod_image.height,
                date_captured=date_captured, flickr_url=None,
                id=linemod_image.id
            )
            dataset.images.append(coco_image)
        
        pbar = tqdm(total=len(self.annotations), unit='annotation(s)') if show_pbar else None
        if pbar is not None:
            pbar.set_description('Converting Linemod to COCO')
        for linemod_ann in self.annotations:
            mask_filename = get_filename(linemod_ann.mask_path)
            if mask_dir is not None:
                mask_path = f'{mask_dir}/{mask_filename}'
                if not file_exists(mask_path):
                    if check_paths:
                        raise FileNotFoundError(f"Couldn't find mask at {mask_path}")
                    else:
                        seg = Segmentation()
                else:
                    seg = Segmentation.from_mask_path(
                        mask_path,
                        lower_bgr=mask_lower_bgr,
                        upper_bgr=mask_upper_bgr
                    )
            elif file_exists(linemod_ann.mask_path):
                seg = Segmentation.from_mask_path(
                    linemod_ann.mask_path,
                    lower_bgr=mask_lower_bgr,
                    upper_bgr=mask_upper_bgr
                )
            elif img_dir is not None and file_exists(f'{img_dir}/{mask_filename}'):
                seg = Segmentation.from_mask_path(
                    f'{img_dir}/{mask_filename}',
                    lower_bgr=mask_lower_bgr,
                    upper_bgr=mask_upper_bgr
                )
            elif not check_paths:
                seg = Segmentation()
            else:
                raise FileNotFoundError(
                    f"""
                    Couldn't resolve mask_path for calculating segmentation.
                    Please either specify mask_dir or correct the mask paths
                    in your linemod dataset.
                    linemod_ann.id: {linemod_ann.id}
                    linemod_ann.mask_path: {linemod_ann.mask_path}
                    """
                )
            if len(seg) > 0:
                bbox = seg.to_bbox()
            else:
                xmin = int(linemod_ann.corner_2d.to_numpy(demarcation=True)[:, 0].min())
                xmax = int(linemod_ann.corner_2d.to_numpy(demarcation=True)[:, 0].max())
                ymin = int(linemod_ann.corner_2d.to_numpy(demarcation=True)[:, 1].min())
                ymax = int(linemod_ann.corner_2d.to_numpy(demarcation=True)[:, 1].max())
                bbox = BBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
            
            keypoints = Keypoint2D_List.from_point_list(linemod_ann.fps_2d, visibility=2)
            keypoints_3d = Keypoint3D_List.from_point_list(linemod_ann.fps_3d, visibility=2)
            num_keypoints = len(keypoints)

            if linemod_ann.category_id not in [cat.id for cat in dataset.categories]:
                linemod_cat = self.categories.get_obj_from_id(linemod_ann.category_id)
                cat_keypoints = list('abcdefghijklmnopqrstuvwxyz'.upper())[:num_keypoints]
                cat_keypoints_idx_list = [idx for idx in range(len(cat_keypoints))]
                cat_keypoints_idx_list_shift_left = cat_keypoints_idx_list[1:]+cat_keypoints_idx_list[:1]
                dataset.categories.append(
                    COCO_Category(
                        id=linemod_ann.category_id,
                        supercategory=linemod_cat.supercategory,
                        name=linemod_cat.name,
                        keypoints=cat_keypoints,
                        skeleton=[[start_idx, end_idx] for start_idx, end_idx in zip(cat_keypoints_idx_list, cat_keypoints_idx_list_shift_left)]
                    )
                )
            
            coco_ann = COCO_Annotation(
                id=linemod_ann.id,
                category_id=linemod_ann.category_id,
                image_id=linemod_ann.image_id,
                segmentation=seg,
                bbox=bbox,
                area=bbox.area(),
                keypoints=keypoints,
                num_keypoints=num_keypoints,
                iscrowd=0,
                keypoints_3d=keypoints_3d,
                camera=COCO_Camera(
                    f=[linemod_ann.K.fx, linemod_ann.K.fy],
                    c=[linemod_ann.K.cx, linemod_ann.K.cy],
                    T=[0, 0]
                )
            )
            dataset.annotations.append(coco_ann)
            if pbar is not None:
                pbar.update()
        if pbar is not None:
            pbar.close()
        return dataset

    @classmethod
    def combine(cls, dataset_list: List[Linemod_Dataset], show_pbar: bool=True) -> Linemod_Dataset:
        combined_dataset = Linemod_Dataset()
        pbar = tqdm(total=len(dataset_list), unit='dataset(s)', leave=True) if show_pbar else None
        if pbar is not None:
            pbar.set_description('Combining Linemod Datasets')
        for dataset in dataset_list:
            assert isinstance(dataset, Linemod_Dataset)
            image_id_map = {}
            category_id_map = {}
            for linemod_image in dataset.images:
                linemod_image0 = linemod_image.copy()
                linemod_image0.id = len(combined_dataset.images)
                image_id_map[linemod_image.id] = linemod_image0.id
                combined_dataset.images.append(linemod_image0)
            for linemod_category in dataset.categories:
                found = False
                for existing_category in combined_dataset.categories:
                    if linemod_category.name == existing_category.name and linemod_category.supercategory == existing_category.supercategory:
                        category_id_map[linemod_category.id] = existing_category.id
                        found = True
                        break
                if not found:
                    linemod_category0 = linemod_category.copy()
                    linemod_category0.id = len(combined_dataset.categories)
                    category_id_map[linemod_category.id] = linemod_category0.id
                    combined_dataset.categories.append(linemod_category0)
            for linemod_ann in dataset.annotations:
                linemod_ann0 = linemod_ann.copy()
                linemod_ann0.id = len(combined_dataset.annotations)
                linemod_ann0.image_id = image_id_map[linemod_ann.image_id]
                linemod_ann0.category_id = category_id_map[linemod_ann.category_id]
                combined_dataset.annotations.append(linemod_ann0)
            if pbar is not None:
                pbar.update()
        if pbar is not None:
            pbar.close()
        return combined_dataset
    
    def set_dataroot(self, dataroot: str, include_mask: bool=True, include_depth: bool=True):
        for ann in self.annotations:
            if include_mask:
                if '/' in ann.mask_path:
                    ann.mask_path = f'{dataroot}/{get_filename(ann.mask_path)}'
                else:
                    ann.mask_path = f'{dataroot}/{ann.mask_path}'
            if include_depth and ann.depth_path is not None:
                if '/' in ann.depth_path:
                    ann.depth_path = f'{dataroot}/{get_filename(ann.depth_path)}'
                else:
                    ann.depth_path = f'{dataroot}/{ann.depth_path}'
            ann.data_root = dataroot

    def set_images_dir(self, img_dir: str):
        for linemod_image in self.images:
            if '/' in linemod_image.file_name:
                linemod_image.file_name = f'{img_dir}/{get_filename(linemod_image.file_name)}'
            else:
                linemod_image.file_name = f'{img_dir}/{linemod_image.file_name}'

    def move(
        self, dst_dataroot: str, include_depth: bool=True, include_RT: bool=False,
        camera_path: str=None, fps_path: str=None,
        preserve_filename: bool=False, use_softlink: bool=False,
        ask_permission_on_delete: bool=True,
        show_pbar: bool=True
    ):
        make_dir_if_not_exists(dst_dataroot)
        delete_all_files_in_dir(
            dst_dataroot,
            ask_permission=ask_permission_on_delete, verbose=False
        )
        processed_image_id_list = []
        pbar = tqdm(total=len(self.annotations), unit='annotation(s)', leave=True) if show_pbar else None
        if pbar is not None:
            pbar.set_description('Moving Linemod Dataset Data')
        for linemod_ann in self.annotations:
            if not dir_exists(linemod_ann.data_root):
                raise FileNotFoundError(f"Couldn't find data_root at {linemod_ann.data_root}")
            
            # Images
            linemod_image = self.images.get(id=linemod_ann.image_id)[0]
            if linemod_image.id not in processed_image_id_list:
                img_path = f'{linemod_ann.data_root}/{get_filename(linemod_image.file_name)}'
                if not file_exists(img_path):
                    raise FileNotFoundError(f"Couldn't find image at {img_path}")
                if preserve_filename:
                    dst_img_path = f'{dst_dataroot}/{get_filename(linemod_image.file_name)}'
                    if file_exists(dst_img_path):
                        raise FileExistsError(
                            f"""
                            Image already exists at {dst_img_path}
                            Hint: Use preserve_filename=False to bypass this error.
                            """
                        )
                else:
                    dst_filename = f'{linemod_image.id}.{get_extension_from_filename(linemod_image.file_name)}'
                    linemod_image.file_name = dst_filename
                    dst_img_path = f'{dst_dataroot}/{dst_filename}'
                if not use_softlink:
                    copy_file(src_path=img_path, dest_path=dst_img_path, silent=True)
                else:
                    create_softlink(src_path=rel_to_abs_path(img_path), dst_path=rel_to_abs_path(dst_img_path))
                processed_image_id_list.append(linemod_image.id)
            
            # Masks
            if not file_exists(linemod_ann.mask_path):
                raise FileNotFoundError(f"Couldn't find mask at {linemod_ann.mask_path}")
            mask_path = linemod_ann.mask_path
            if preserve_filename:
                dst_mask_path = f'{dst_dataroot}/{get_filename(linemod_ann.mask_path)}'
                if file_exists(dst_mask_path):
                    raise FileExistsError(
                        f"""
                        Mask already exists at {dst_mask_path}
                        Hint: Use preserve_filename=False to bypass this error.
                        """
                    )
            else:
                mask_filename = get_filename(linemod_ann.mask_path)
                dst_filename = f'{linemod_ann.id}_mask.{get_extension_from_filename(mask_filename)}'
                dst_mask_path = f'{dst_dataroot}/{dst_filename}'
                linemod_ann.mask_path = dst_mask_path
            if not use_softlink:
                copy_file(src_path=mask_path, dest_path=dst_mask_path, silent=True)
            else:
                create_softlink(src_path=rel_to_abs_path(mask_path), dst_path=rel_to_abs_path(dst_mask_path))
            
            # Depth
            if include_depth and linemod_ann.depth_path is not None:
                if not file_exists(linemod_ann.depth_path):
                    raise FileNotFoundError(f"Couldn't find depth at {linemod_ann.depth_path}")
                depth_path = linemod_ann.depth_path
                if preserve_filename:
                    dst_depth_path = f'{dst_dataroot}/{get_filename(linemod_ann.depth_path)}'
                    if file_exists(dst_depth_path):
                        raise FileExistsError(
                            f"""
                            Depth already exists at {dst_depth_path}
                            Hint: Use preserve_filename=False to bypass this error.
                            """
                        )
                else:
                    depth_filename = get_filename(linemod_ann.depth_path)
                    dst_filename = f'{linemod_ann.id}_depth.{get_extension_from_filename(depth_filename)}'
                    dst_depth_path = f'{dst_dataroot}/{dst_filename}'
                    linemod_ann.depth_path = dst_depth_path
                if not use_softlink:
                    copy_file(src_path=depth_path, dest_path=dst_depth_path, silent=True)
                else:
                    create_softlink(src_path=rel_to_abs_path(depth_path), dst_path=rel_to_abs_path(dst_depth_path))
            
            # RT pickle files
            if include_RT:
                rootname = get_rootname_from_path(mask_path)
                if rootname.endswith('_mask'):
                    rootname = rootname.replace('_mask', '')
                rt_filename = f'{rootname}_RT.pkl'
                rt_path = f'{linemod_ann.data_root}/{rt_filename}'
                if not file_exists(rt_path):
                    raise FileNotFoundError(f"Couldn't find RT pickle file at {rt_path}")
                if preserve_filename:
                    dst_rt_path = f'{dst_dataroot}/{rt_filename}'
                    if file_exists(dst_depth_path):
                        raise FileExistsError(
                            f"""
                            RT pickle file already exists at {dst_rt_path}
                            Hint: Use preserve_filename=False to bypass this error.
                            """
                        )
                else:
                    dst_rt_filename = f'{linemod_ann.id}_RT.pkl'
                    dst_rt_path = f'{dst_dataroot}/{dst_rt_filename}'
                if not use_softlink:
                    copy_file(src_path=rt_path, dest_path=dst_rt_path, silent=True)
                else:
                    create_softlink(src_path=rel_to_abs_path(rt_path), dst_path=rel_to_abs_path(dst_rt_path))
            if pbar is not None:
                pbar.update()
        # Camera setting
        if camera_path is not None:
            if not file_exists(camera_path):
                raise FileNotFoundError(f"Couldn't find camera settings at {camera_path}")
            dst_camera_path = f'{dst_dataroot}/{get_filename(camera_path)}'
            if file_exists(dst_camera_path):
                raise FileExistsError(f'Camera settings already saved at {dst_camera_path}')
            if not use_softlink:
                copy_file(src_path=camera_path, dest_path=dst_camera_path, silent=True)
            else:
                create_softlink(src_path=rel_to_abs_path(camera_path), dst_path=rel_to_abs_path(dst_camera_path))
        
        # FPS setting
        if fps_path is not None:
            if not file_exists(fps_path):
                raise FileNotFoundError(f"Couldn't find FPS settings at {fps_path}")
            dst_fps_path = f'{dst_dataroot}/{get_filename(fps_path)}'
            if file_exists(dst_fps_path):
                raise FileExistsError(f'FPS settings already saved at {dst_fps_path}')
            if not use_softlink:
                copy_file(src_path=fps_path, dest_path=dst_fps_path, silent=True)
            else:
                create_softlink(src_path=rel_to_abs_path(fps_path), dst_path=rel_to_abs_path(dst_fps_path))
        if pbar is not None:
            pbar.close()
    
    def sample_3d_hyperparams(self, idx: int=0, include_center: bool=True) -> (Point3D_List, Point3D_List, LinemodCamera):
        linemod_ann_sample = self.annotations[idx]
        kpt_3d = linemod_ann_sample.fps_3d.copy()
        if include_center:
            kpt_3d.append(linemod_ann_sample.center_3d)
        corner_3d = linemod_ann_sample.corner_3d
        K = linemod_ann_sample.K
        return kpt_3d, corner_3d, K

    def sample_dsize(self, idx: int=0) -> (int, int):
        linemod_image_sample = self.images[idx]
        return (linemod_image_sample.width, linemod_image_sample.height)