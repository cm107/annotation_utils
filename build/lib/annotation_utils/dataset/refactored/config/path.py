from __future__ import annotations
from typing import List
from logger import logger
from common_utils.path_utils import rel_to_abs_path
from common_utils.check_utils import check_type, check_type_from_list

class Path:
    def __init__(self, path_str: str):
        self.path_str = path_str

    def __str__(self) -> str:
        return self.path_str

    def __repr__(self) -> str:
        return self.__str__()

    def __key(self) -> tuple:
        return tuple([self.__class__] + list(self.__dict__.values()))

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other) -> bool:
        if isinstance(other, self.__class__):
            return self.__key() == other.__key()
        return NotImplemented

    def __len__(self) -> int:
        return len(self.split())

    def __getitem__(self, idx: int) -> Path:
        if type(idx) is int:
            if len(self) == 0:
                logger.error(f"{type(self).__name__} is empty.")
                raise IndexError
            elif idx >= len(self) or idx < -len(self):
                logger.error(f"Index out of range: {idx}")
                raise IndexError
            else:
                return Path(self.split()[idx])
        elif type(idx) is slice:
            return Path.from_split(self.split()[idx.start:idx.stop:idx.step])
        else:
            logger.error(f'Expected int or slice. Got type(idx)={type(idx)}')
            raise TypeError

    def __setitem__(self, idx: int, value: Path):
        check_type(value, valid_type_list=[Path, str])
        if type(value) is str:
            value0 = Path(value)
        else:
            value0 = value
        path_parts = self.split()
        if type(idx) is int:
            path_parts[idx] = value.path_str
        elif type(idx) is slice:
            path_parts[idx.start:idx.stop:idx.step] = value.split()
        else:
            logger.error(f'Expected int or slice. Got type(idx)={type(idx)}')
            raise TypeError

    def __delitem__(self, idx):
        if type(idx) is int:
            if len(self) == 0:
                logger.error(f"{type(self).__name__} is empty.")
                raise IndexError
            elif idx >= len(self) or idx < -len(self):
                logger.error(f"Index out of range: {idx}")
                raise IndexError
            else:
                path_parts = self.split()
                del path_parts[idx]
                result = Path.from_split(path_parts)
                self.path_str = result.to_str()
        elif type(idx) is slice:
            path_parts = self.split()
            del path_parts[idx.start:idx.stop:idx.step]
            result = Path.from_split(path_parts)
            self.path_str = result.to_str()
        else:
            logger.error(f'Expected int or slice. Got type(idx)={type(idx)}')
            raise TypeError

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self) -> Path:
        if self.n < len(self):
            result = Path(self.split()[self.n])
            self.n += 1
            return result
        else:
            raise StopIteration

    def __add__(self, other: Path) -> Path:
        return Path.from_split(self.split() + other.split()).prune_slashes()

    @classmethod
    def buffer(cls, path: Path) -> Path:
        return path

    def copy(self) -> Path:
        return Path(self.path_str)

    def split(self) -> List[str]:
        result = self.path_str.split('/')
        result = [result_part for result_part in result]
        return result

    @classmethod
    def from_split(self, str_list: List[str]) -> Path:
        return Path('/'.join(str_list))

    def head(self) -> Path:
        return self[0]

    def no_head(self) -> Path:
        return self[1:]

    def tail(self) -> Path:
        return self[-1]

    def no_tail(self) -> Path:
        return self[:-1]

    def to_str(self) -> str:
        return self.path_str

    def pop_head(self) -> Path:
        result = self.head()
        self.path_str = self.no_head().to_str()
        return result

    def pop_tail(self) -> Path:
        result = self.tail()
        self.path_str = self.no_tail().to_str()
        return result

    def push_head(self, path_part: Path):
        check_type(path_part, valid_type_list=[Path, str])
        if type(path_part) is str:
            path_part0 = Path(path_part)
        else:
            path_part0 = path_part
        return Path.from_split([path_part0.path_str] + self.split())

    def push_tail(self, path_part: Path):
        check_type(path_part, valid_type_list=[Path, str])
        if type(path_part) is str:
            path_part0 = Path(path_part)
        else:
            path_part0 = path_part
        return Path.from_split(self.split() + [path_part0.path_str])

    def get_extension(self) -> str:
        if '.' in self.tail().path_str:
            return self.tail().path_str.split('.')[-1]
        else:
            return ''

    def has_extension(self) -> bool:
        return self.get_extension() != ''

    def abs(self) -> Path:
        return Path(rel_to_abs_path(self.to_str()))

    def prune_slashes(self) -> Path:
        path_str = self.to_str()
        while '//' in path_str:
            path_str = path_str.replace('//', '/')
        return Path(path_str)

    @classmethod
    def get_unique_paths(cls, paths: List[Path]) -> List[Path]:
        path_str_list = [path.path_str for path in paths]
        unique_path_str_list = list(dict.fromkeys(path_str_list))
        unique_paths = [Path(unique_path_str) for unique_path_str in unique_path_str_list]
        return unique_paths

    @classmethod
    def has_common_head(cls, paths: List[Path]) -> bool:
        if len(paths) > 1:
            path_len_list = [len(path) for path in paths]
            if 0 in path_len_list:
                return False
            path_heads = [path.head() for path in paths]
            unique_path_heads = cls.get_unique_paths(path_heads)
            if len(unique_path_heads) == 1:
                return True
            else:
                return False
        elif len(paths) == 1 and len(paths[0]) > 0 and paths[0] != Path(''):
            return True
        else:
            return False

    @classmethod
    def has_common_tail(cls, paths: List[Path]) -> bool:
        if len(paths) > 1:
            path_len_list = [len(path) for path in paths]
            if 0 in path_len_list:
                return False
            path_tails = [path.tail() for path in paths]
            unique_path_tails = cls.get_unique_paths(path_tails)
            if len(unique_path_tails) == 1:
                return True
            else:
                return False
        elif len(paths) == 1 and len(paths[0]) > 0 and paths[0] != Path(''):
            return True
        else:
            return False

    @classmethod
    def get_common_head(cls, paths: List[Path]) -> (bool, Path):
        if len(paths) > 1:
            path_len_list = [len(path) for path in paths]
            if 0 in path_len_list:
                return False, None
            path_heads = [path.head() for path in paths]
            unique_path_heads = cls.get_unique_paths(path_heads)
            if len(unique_path_heads) == 1 and unique_path_heads[0] != Path(''):
                return True, unique_path_heads[0]
            else:
                return False, None
        elif len(paths) == 1 and len(paths[0]) > 0 and paths[0].head() != Path(''):
            return True, paths[0].head()
        else:
            return False, None

    @classmethod
    def get_common_tail(cls, paths: List[Path]) -> (bool, Path):
        if len(paths) > 1:
            path_len_list = [len(path) for path in paths]
            if 0 in path_len_list:
                return False, None
            path_tails = [path.tail() for path in paths]
            unique_path_tails = cls.get_unique_paths(path_tails)
            if len(unique_path_tails) == 1 and unique_path_tails[0] != Path(''):
                return True, unique_path_tails[0]
            else:
                return False, None
        elif len(paths) == 1 and len(paths[0]) > 0 and paths[0].tail() != Path(''):
            return True, paths[0].tail()
        else:
            return False, None

    def replace(self, old: Path, new: Path) -> Path:
        check_type_from_list([old, new], valid_type_list=[Path, str])
        if type(old) is str:
            old_path = Path(old)
        else:
            old_path = old
        if type(new) is str:
            new_path = Path(new)
        else:
            new_path = new
        result = self.to_str()
        result = result.replace(old_path.to_str(), new_path.to_str())
        while '//' in result:
            result = result.replace('//', '/')
        return Path(result)

    def possible_rel_paths(self) -> List[Path]:
        return [Path.from_split(self.split()[i:]) for i in range(len(self))]

    def possible_container_dirs(self) -> List[Path]:
        result = [
            Path.from_split(self.split()[:i]) if self.has_extension() else Path.from_split(self.split()[:i+1]) \
                for i in range(len(self))
        ]
        result = [path for path in result if path.path_str != '']
        return result

    @classmethod
    def del_duplicates(cls, path_list: List[Path]) -> List[Path]:
        path_str_list = list(dict.fromkeys([path.path_str for path in path_list]))
        return [Path(path_str) for path_str in path_str_list]

    @classmethod
    def get_common_container_dirs(cls, path_list: List[Path]) -> List[Path]:
        if len(path_list) == 0:
            logger.error(f"Encountered len(path_list) == 0")
            raise Exception
        possible_container_dirs_sets = []
        for path in path_list:
            possible_container_dirs_sets.append(set([path0.to_str() for path0 in path.possible_container_dirs()]))
        common_container_dirs = list(set.intersection(*possible_container_dirs_sets))
        return [Path(common_container_dir) for common_container_dir in common_container_dirs]

    @classmethod
    def get_longest_container_dir(cls, path_list: List[Path]) -> Path:
        if len(path_list) == 0:
            logger.error(f"Encountered len(path_list) == 0")
            raise Exception
        common_container_dirs = Path.get_common_container_dirs(path_list)
        longest_container_dir = None
        for common_container_dir in common_container_dirs:
            if longest_container_dir is None or len(common_container_dir) > len(longest_container_dir):
                longest_container_dir = common_container_dir
        return longest_container_dir

    @staticmethod
    def _root_src_tail2dst_head(src_path: Path, dst_path: Path) -> bool:
        check_type(src_path, valid_type_list=[Path])
        check_type(dst_path, valid_type_list=[Path])
        if len(src_path) > 0:
            tail = src_path.pop_tail()
            if tail != Path(''):
                dst_path.path_str = (tail + dst_path).to_str()
                success = True
            else:
                success = False
        else:
            success = False
        return success

    @classmethod
    def _root_src_head2dst_tail(cls, src_path: Path, dst_path: Path) -> bool:
        check_type(src_path, valid_type_list=[Path])
        check_type(dst_path, valid_type_list=[Path])
        if len(src_path) > 0:
            head = src_path.pop_head()
            if head != Path(''):
                dst_path.path_str = (dst_path + head).to_str()
                success = True
            else:
                success = False
        else:
            success = False
        return success

    @classmethod
    def tail2head(cls, src_obj: List[Path], dst_obj: List[Path]) -> bool:
        # TODO: Assertion test
        check_type_from_list([src_obj, dst_obj], valid_type_list=[Path, list])
        if type(src_obj) is list:
            check_type_from_list(src_obj, valid_type_list=[Path])
        if type(dst_obj) is list:
            check_type_from_list(dst_obj, valid_type_list=[Path])
        
        if type(src_obj) is Path:
            if type(dst_obj) is Path:
                success = cls._root_src_tail2dst_head(src_obj, dst_obj)
            elif type(dst_obj) is list:
                if len(src_obj) > 0:
                    tail = src_obj.pop_tail()
                    for dst_path in dst_obj:
                        dst_path.path_str = (tail + dst_path).to_str()
                    success = True
                else:
                    success = False
            else:
                raise Exception
        elif type(src_obj) is list:
            has_common_tail, common_tail = cls.get_common_tail(src_obj)
            if has_common_tail:
                if type(dst_obj) is Path:
                    dst_obj.path_str = (common_tail + dst_obj).to_str()
                    for src_path in src_obj:
                        src_path.path_str = src_path.no_tail().to_str()
                    success = True
                elif type(dst_obj) is list:
                    if len(src_obj) != len(dst_obj):
                        logger.error(f'len(src_obj) == {len(src_obj)} != {len(dst_obj)} == len(dst_obj)')
                        raise Exception
                    for dst_path in dst_obj:
                        dst_path.path_str = (common_tail + dst_path).to_str()
                    for src_path in src_obj:
                        src_path.path_str = src_path.no_tail().to_str()
                    success = True
                else:
                    raise Exception
            else:
                success = False
        else:
            raise Exception
        return success

    @classmethod
    def head2tail(cls, src_obj: List[Path], dst_obj: List[Path]) -> bool:
        check_type_from_list([src_obj, dst_obj], valid_type_list=[Path, list])
        if type(src_obj) is list:
            check_type_from_list(src_obj, valid_type_list=[Path])
        if type(dst_obj) is list:
            check_type_from_list(dst_obj, valid_type_list=[Path])
        
        if type(src_obj) is Path:
            if type(dst_obj) is Path:
                success = cls._root_src_head2dst_tail(src_obj, dst_obj)
            elif type(dst_obj) is list:
                if len(src_obj) > 0:
                    head = src_obj.pop_head()
                    for dst_path in dst_obj:
                        dst_path.path_str = (dst_path + head).to_str()
                    success = True
                else:
                    success = False
            else:
                raise Exception
        elif type(src_obj) is list:
            has_common_head, common_head = cls.get_common_head(src_obj)
            if has_common_head:
                if type(dst_obj) is Path:
                    dst_obj.path_str = (dst_obj + common_head).to_str()
                    for src_path in src_obj:
                        src_path.path_str = src_path.no_head().to_str()
                    success = True
                elif type(dst_obj) is list:
                    if len(src_obj) != len(dst_obj):
                        logger.error(f'len(src_obj) == {len(src_obj)} != {len(dst_obj)} == len(dst_obj)')
                        raise Exception
                    for dst_path in dst_obj:
                        dst_path.path_str = (dst_path + common_head).to_str()
                    for src_path in src_obj:
                        src_path.path_str = src_path.no_head().to_str()
                    success = True
                else:
                    raise Exception
            else:
                success = False
        else:
            raise Exception
        return success