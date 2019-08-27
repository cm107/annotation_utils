from ..submodules.logger.logger_handler import logger
from ..submodules.common_utils.file_utils import dir_exists, file_exists, move_file
from ..submodules.common_utils.path_utils import get_all_files_of_extension, get_rootname_from_path, \
    get_dirpath_from_filepath, get_filename

class DataMover:
    def __init__(self, img_src: str, xml_src: str, img_dst: str, xml_dst: str):
        self.img_src = img_src
        self.xml_src = xml_src
        self.img_dst = img_dst
        self.xml_dst = xml_dst
        if not dir_exists(self.img_src):
            logger.error(f"Directory doesn't exist: {self.img_src}")
            raise Exception
        if not dir_exists(self.xml_src):
            logger.error(f"Directory doesn't exist: {self.xml_src}")
            raise Exception
        if not dir_exists(self.img_dst):
            logger.error(f"Directory doesn't exist: {self.img_dst}")
            raise Exception
        if not dir_exists(self.xml_dst):
            logger.error(f"Directory doesn't exist: {self.xml_dst}")
            raise Exception

    def get_pathlists(self) -> (list, list, list, list):
        img_src_pathlist = get_all_files_of_extension(dir_path=self.img_src, extension='png')
        xml_src_pathlist = get_all_files_of_extension(dir_path=self.xml_src, extension='xml')
        img_dst_pathlist = get_all_files_of_extension(dir_path=self.img_dst, extension='png')
        xml_dst_pathlist = get_all_files_of_extension(dir_path=self.xml_dst, extension='xml')
        return img_src_pathlist, xml_src_pathlist, img_dst_pathlist, xml_dst_pathlist

    def move(self):
        img_src_pathlist, xml_src_pathlist, img_dst_pathlist, xml_dst_pathlist = self.get_pathlists()
        nonempty = self.is_nonempty(self.img_src, img_src_pathlist, self.xml_src, xml_src_pathlist)
        if nonempty:
            self.check_dst_size(img_dst_pathlist, xml_dst_pathlist)
            self.check_all_xml_exists_in_img(img_src_pathlist, xml_src_pathlist, img_dst_pathlist, xml_dst_pathlist)
            self.move_all_src_to_dst(img_src_pathlist, xml_src_pathlist)
            self.post_move_check()
        else:
            logger.warning(f"Since the src directory is empty, there is nothing to be done.")

    def is_nonempty(self, img_dir: str, img_pathlist: list, xml_dir: str, xml_pathlist: list):
        if len(img_pathlist) == 0:
            logger.warning(f"{img_dir} is empty.")
            return False
        if len(xml_pathlist) == 0:
            logger.warning(f"{xml_dir} is empty.")
            return False
        return True

    def check_xml_empty(self, xml_dir: str, xml_pathlist):
        if len(xml_pathlist) > 0:
            logger.error(f"{xml_dir} is not empty.")
            raise Exception

    def check_dst_size(self, img_dst_pathlist: list, xml_dst_pathlist: list):
        if len(img_dst_pathlist) != len(xml_dst_pathlist):
            logger.error(f"Directory size mismatch.")
            logger.error(f"len(img_dst_pathlist) = {len(img_dst_pathlist)} != {len(xml_dst_pathlist)} = len(xml_dst_pathlist)")
            raise Exception

    def check_xml_exists_in_img(self, img_pathlist: list, xml_pathlist: list, img_dir_path: str):
        paths_not_found = []
        for xml_path in xml_pathlist:
            rootname = get_rootname_from_path(xml_path)
            corresponding_img_path = f"{img_dir_path}/{rootname}.png"
            if not file_exists(corresponding_img_path):
                paths_not_found.append(corresponding_img_path)
        if len(paths_not_found) > 0:
            for path_not_found in paths_not_found:
                logger.error(f"File not found: {path_not_found}")
            raise Exception

    def check_all_xml_exists_in_img(self, img_src_pathlist: list, xml_src_pathlist: list, img_dst_pathlist: list, xml_dst_pathlist: list):
        self.check_xml_exists_in_img(img_src_pathlist, xml_src_pathlist, self.img_src)
        self.check_xml_exists_in_img(img_dst_pathlist, xml_dst_pathlist, self.img_dst)

    def move_all_src_to_dst(self, img_src_pathlist: list, xml_src_pathlist: list):
        for xml_src_path in xml_src_pathlist:
            rootname = get_rootname_from_path(xml_src_path)
            xml_dst_path = f"{self.xml_dst}/{rootname}.xml"
            img_src_path = f"{self.img_src}/{rootname}.png"
            img_dst_path = f"{self.img_dst}/{rootname}.png"
            self.premove_check(img_src_path, img_dst_path, xml_src_path, xml_dst_path)
            move_file(xml_src_path, xml_dst_path, silent=False)
            move_file(img_src_path, img_dst_path, silent=False)

    def premove_check(self, img_src_path: str, img_dst_path: str, xml_src_path: str, xml_dst_path: str):
        paths_not_found = []
        paths_already_exists = []
        if not file_exists(img_src_path):
            paths_not_found.append(img_src_path)
        if file_exists(img_dst_path):
            paths_already_exists.append(img_dst_path)
        if not file_exists(xml_src_path):
            paths_not_found.append(xml_src_path)
        if file_exists(xml_dst_path):
            paths_already_exists.append(xml_dst_path)
        if len(paths_not_found) > 0:
            for path_not_found in paths_not_found:
                logger.error(f"File not found: {path_not_found}")
            raise Exception
        if len(paths_already_exists) > 0:
            for path_already_exists in paths_already_exists:
                logger.error(f"File already exists: {path_already_exists}")
            raise Exception

    def post_move_check(self):
        img_src_pathlist, xml_src_pathlist, img_dst_pathlist, xml_dst_pathlist = self.get_pathlists()
        nonempty = self.is_nonempty(self.img_dst, img_dst_pathlist, self.xml_dst, xml_dst_pathlist)
        if nonempty:
            self.check_xml_empty(self.xml_src, xml_src_pathlist)
            logger.info(f"There are currently {len(img_src_pathlist)} unlabeled images in {self.img_src}")
        else:
            logger.error(f"The destination directories is empty after post-move. Something has gone wrong.")
            raise Exception
