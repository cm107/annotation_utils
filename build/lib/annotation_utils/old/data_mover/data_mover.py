from logger import logger
from common_utils.file_utils import dir_exists, file_exists, move_file
from common_utils.path_utils import get_all_files_of_extension, get_rootname_from_path, \
    get_next_dump_path

class DataMover:
    def __init__(
        self, img_src: str, ann_src: str, img_dst: str, ann_dst: str,
        ann_extension: str, auto_renaming: bool=False, assume_labelme: bool=False
    ):
        self.img_src = img_src
        self.ann_src = ann_src
        self.img_dst = img_dst
        self.ann_dst = ann_dst
        self.ann_extension = ann_extension
        self.auto_renaming = auto_renaming
        self.assume_labelme = assume_labelme
        if not dir_exists(self.img_src):
            logger.error(f"Directory doesn't exist: {self.img_src}")
            raise Exception
        if not dir_exists(self.ann_src):
            logger.error(f"Directory doesn't exist: {self.ann_src}")
            raise Exception
        if not dir_exists(self.img_dst):
            logger.error(f"Directory doesn't exist: {self.img_dst}")
            raise Exception
        if not dir_exists(self.ann_dst):
            logger.error(f"Directory doesn't exist: {self.ann_dst}")
            raise Exception

    def get_pathlists(self) -> (list, list, list, list):
        img_src_pathlist = get_all_files_of_extension(dir_path=self.img_src, extension='png')
        ann_src_pathlist = get_all_files_of_extension(dir_path=self.ann_src, extension=self.ann_extension)
        img_dst_pathlist = get_all_files_of_extension(dir_path=self.img_dst, extension='png')
        ann_dst_pathlist = get_all_files_of_extension(dir_path=self.ann_dst, extension=self.ann_extension)
        return img_src_pathlist, ann_src_pathlist, img_dst_pathlist, ann_dst_pathlist

    def move(self):
        img_src_pathlist, ann_src_pathlist, img_dst_pathlist, ann_dst_pathlist = self.get_pathlists()
        nonempty = self.is_nonempty(self.img_src, img_src_pathlist, self.ann_src, ann_src_pathlist)
        if nonempty:
            self.check_dst_size(img_dst_pathlist, ann_dst_pathlist)
            self.check_all_ann_exists_in_img(img_src_pathlist, ann_src_pathlist, img_dst_pathlist, ann_dst_pathlist)
            if not self.assume_labelme:
                self.move_all_src_to_dst(img_src_pathlist, ann_src_pathlist)
            else:
                self.labelme_move_all_src_to_dst(img_dst_pathlist, ann_src_pathlist)
            self.post_move_check()
        else:
            logger.warning(f"Since the src directory is empty, there is nothing to be done.")

    def is_nonempty(self, img_dir: str, img_pathlist: list, ann_dir: str, ann_pathlist: list):
        if len(img_pathlist) == 0:
            logger.warning(f"{img_dir} is empty.")
            return False
        if len(ann_pathlist) == 0:
            logger.warning(f"{ann_dir} is empty.")
            return False
        return True

    def check_ann_empty(self, ann_dir: str, ann_pathlist):
        if len(ann_pathlist) > 0:
            logger.error(f"{ann_dir} is not empty.")
            raise Exception

    def check_dst_size(self, img_dst_pathlist: list, ann_dst_pathlist: list):
        if len(img_dst_pathlist) != len(ann_dst_pathlist):
            logger.error(f"Directory size mismatch.")
            logger.error(f"len(img_dst_pathlist) = {len(img_dst_pathlist)} != {len(ann_dst_pathlist)} = len(ann_dst_pathlist)")
            raise Exception

    def check_ann_exists_in_img(self, img_pathlist: list, ann_pathlist: list, img_dir_path: str):
        paths_not_found = []
        for ann_path in ann_pathlist:
            rootname = get_rootname_from_path(ann_path)
            corresponding_img_path = f"{img_dir_path}/{rootname}.png"
            if not file_exists(corresponding_img_path):
                paths_not_found.append(corresponding_img_path)
        if len(paths_not_found) > 0:
            for path_not_found in paths_not_found:
                logger.error(f"File not found: {path_not_found}")
            raise Exception

    def check_all_ann_exists_in_img(self, img_src_pathlist: list, ann_src_pathlist: list, img_dst_pathlist: list, ann_dst_pathlist: list):
        self.check_ann_exists_in_img(img_src_pathlist, ann_src_pathlist, self.img_src)
        self.check_ann_exists_in_img(img_dst_pathlist, ann_dst_pathlist, self.img_dst)

    def move_all_src_to_dst(self, img_src_pathlist: list, ann_src_pathlist: list):
        for ann_src_path in ann_src_pathlist:
            rootname = get_rootname_from_path(ann_src_path)
            img_src_path = f"{self.img_src}/{rootname}.png"
            if not self.auto_renaming:
                ann_dst_path = f"{self.ann_dst}/{rootname}.{self.ann_extension}"
                img_dst_path = f"{self.img_dst}/{rootname}.png"
                
            else:
                ann_dst_path = get_next_dump_path(
                    dump_dir=self.ann_dst,
                    file_extension=self.ann_extension
                )
                rootname = get_rootname_from_path(ann_dst_path)
                img_dst_path = f"{self.img_dst}/{rootname}.png"

            self.premove_check(img_src_path, img_dst_path, ann_src_path, ann_dst_path)
            move_file(ann_src_path, ann_dst_path, silent=False)
            move_file(img_src_path, img_dst_path, silent=False)

    def labelme_move_all_src_to_dst(self, img_src_pathlist: list, ann_src_pathlist: list):
        from .util.labelme_utils import move_annotation

        for ann_src_path in ann_src_pathlist:
            rootname = get_rootname_from_path(ann_src_path)
            img_src_path = f"{self.img_src}/{rootname}.png"
            if not self.auto_renaming:
                ann_dst_path = f"{self.ann_dst}/{rootname}.{self.ann_extension}"
                img_dst_path = f"{self.img_dst}/{rootname}.png"
                
            else:
                ann_dst_path = get_next_dump_path(
                    dump_dir=self.ann_dst,
                    file_extension=self.ann_extension
                )
                rootname = get_rootname_from_path(ann_dst_path)
                img_dst_path = f"{self.img_dst}/{rootname}.png"

            self.premove_check(img_src_path, img_dst_path, ann_src_path, ann_dst_path)
            move_file(img_src_path, img_dst_path, silent=False)
            move_annotation(
                src_img_path=img_src_path,
                src_json_path=ann_src_path,
                dst_img_path=img_dst_path,
                dst_json_path=ann_dst_path,
                bound_type='rect'
            )
            src_preview = '/'.join(ann_src_path.split('/')[-3:])
            dest_preview = '/'.join(ann_dst_path.split('/')[-3:])
            print('Moved {} to {}'.format(src_preview, dest_preview))

    def premove_check(self, img_src_path: str, img_dst_path: str, ann_src_path: str, ann_dst_path: str):
        paths_not_found = []
        paths_already_exists = []
        if not file_exists(img_src_path):
            paths_not_found.append(img_src_path)
        if file_exists(img_dst_path):
            paths_already_exists.append(img_dst_path)
        if not file_exists(ann_src_path):
            paths_not_found.append(ann_src_path)
        if file_exists(ann_dst_path):
            paths_already_exists.append(ann_dst_path)
        if len(paths_not_found) > 0:
            for path_not_found in paths_not_found:
                logger.error(f"File not found: {path_not_found}")
            raise Exception
        if len(paths_already_exists) > 0:
            for path_already_exists in paths_already_exists:
                logger.error(f"File already exists: {path_already_exists}")
            raise Exception

    def post_move_check(self):
        img_src_pathlist, ann_src_pathlist, img_dst_pathlist, ann_dst_pathlist = self.get_pathlists()
        nonempty = self.is_nonempty(self.img_dst, img_dst_pathlist, self.ann_dst, ann_dst_pathlist)
        if nonempty:
            self.check_ann_empty(self.ann_src, ann_src_pathlist)
            logger.info(f"There are currently {len(img_src_pathlist)} unlabeled images in {self.img_src}")
        else:
            logger.error(f"The destination directories is empty after post-move. Something has gone wrong.")
            raise Exception
