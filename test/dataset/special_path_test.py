from annotation_utils.dataset.config.dataset_config import Path

assert Path.get_longest_container_dir(
    [
        Path('/a/b/c/d/e.jpg'),
        Path('/a/b/c/d.jpg'),
        Path('/a/b/c/d/e/f.jpg')
    ]
).path_str == '/a/b/c'

assert Path.get_longest_container_dir(
    [
        Path('/a/b/c/d/e.jpg'),
        Path('/a/d.jpg'),
        Path('/a/f.jpg')
    ]
).path_str == '/a'

assert Path.get_longest_container_dir(
    [
        Path('/a/b/c/d/e'),
        Path('/a/b/c/d'),
        Path('/a/b/c/d/e/f'),
    ]
).path_str == '/a/b/c/d'

assert Path.get_longest_container_dir(
    [
        Path('/a'),
        Path('/a/b/c/d'),
        Path('/a/b/c/d/e/f'),
    ]
).path_str == '/a'

assert Path('a/b/c/d')[1].path_str == 'b'
assert Path('a/b/c/d')[1:3].path_str == 'b/c'
assert Path('/a/b/c/d')[0].path_str == ''
assert Path('/a/b/c/d').replace('b/c', 'x/y').path_str == '/a/x/y/d'
assert Path('/a/b/c/d').replace('b/c', '').path_str == '/a/d'
assert Path('a/b') + Path('c/d') == Path('a/b/c/d')

test_path = Path('/a/b/c/d/e/f')
del test_path[2]
assert test_path == Path('/a/c/d/e/f')
del test_path[-3:]
assert test_path == Path('/a/c')

src_path_list0 = [
    Path('/a/b/c/d/e/f'),
    Path('/A/B/C/D/e/f'),
    Path('/X/Y/Z/d/e/f')
]
dst_path = Path('collection_dir')
dst_path_list = [
    Path('x/y/z'),
    Path('x/y/zz'),
    Path('X/X/Z')
]
src_path_list0_orig = src_path_list0.copy()
dst_path_orig = dst_path.copy()
dst_path_list_orig = dst_path_list.copy()

success0 = Path.tail2head(src_path_list0, dst_path)
assert success0
success1 = Path.tail2head(src_path_list0, dst_path_list)
assert success1
success2 = Path.head2tail(dst_path_list, src_path_list0)
assert success2
success3 = Path.head2tail(dst_path, src_path_list0)
assert success3
assert src_path_list0 == src_path_list0_orig
assert dst_path == dst_path_orig
assert dst_path_list == dst_path_list_orig
