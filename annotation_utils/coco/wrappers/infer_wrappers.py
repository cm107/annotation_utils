from typing import cast
from tqdm import tqdm
from common_utils.file_utils import make_dir_if_not_exists, delete_all_files_in_dir
from common_utils.base.basic import BasicLoadableHandler
from streamer.recorder.stream_writer import StreamWriter
from ..structs.dataset import COCO_Dataset

def infer_tests_wrapper(
    weight_path: str, model_name: str, dataset: COCO_Dataset, test_name: str,
    handler_constructor: type,
    data_dump_dir: str=None, video_dump_dir: str=None, img_dump_dir: str=None,
    show_preview: bool=False,
    show_pbar: bool=True
):
    """
    Usage:
        infer_tests_wrapper(*args, **kwargs)(infer_func)(*infer_func_args, **infer_func_kwargs)
        The following are reserved parameters:
            'weight_path', 'model_name', 'dataset', 'test_name',
            'accumulate_pred_dump', 'stream_writer', 'leave_stream_writer_open'
        These reserved parameters should be keyword parameters in infer_func, but not specified
        in infer_func_kwargs.
        Ideally these reserved parameters should be last in the list of parameters in infer_func without a default value.
        This is to allow freedom for remaining parameters to be specified as non-keyward arguments.
    
    TODO: Add functionality for calculating error during inference. (Will need to pass filtered GT handler.)
    """
    def _wrapper(infer_func):
        def _wrapper_inner(*args, **kwargs):
            # Check/Adjust Parameters
            if isinstance(weight_path, str):
                weight_paths = [weight_path]
            elif isinstance(weight_path, (tuple, list)):
                assert all([type(part) is str for part in weight_path])
                weight_paths = weight_path
            else:
                raise TypeError
            if isinstance(model_name, str):
                model_names = [model_name]
            elif isinstance(model_name, (tuple, list)):
                assert all([type(part) is str for part in model_name])
                model_names = model_name
            else:
                raise TypeError
            assert len(weight_paths) == len(model_names)
            if isinstance(dataset, COCO_Dataset):
                datasets = [dataset]
            elif isinstance(dataset, (tuple, list)):
                assert all([isinstance(part, COCO_Dataset) for part in dataset])
                datasets = dataset
            else:
                raise TypeError
            if isinstance(test_name, str):
                test_names = [test_name]
            elif isinstance(test_name, (tuple, list)):
                assert all([type(part) is str for part in test_name])
                test_names = test_name
            else:
                raise TypeError
            assert len(datasets) == len(test_names)
            
            # Prepare Dump Directory
            if data_dump_dir is not None:
                make_dir_if_not_exists(data_dump_dir)
                delete_all_files_in_dir(data_dump_dir, ask_permission=True)
            if video_dump_dir is not None:
                make_dir_if_not_exists(video_dump_dir)
                delete_all_files_in_dir(video_dump_dir, ask_permission=True)
            if img_dump_dir is not None:
                make_dir_if_not_exists(img_dump_dir)
                delete_all_files_in_dir(img_dump_dir, ask_permission=True)
            stream_writer = cast(StreamWriter, None)

            # Accumulate/Save Inference Data On Tests
            total_images = sum([len(dataset.images) for dataset in datasets])
            test_pbar = tqdm(total=total_images*len(model_names), unit='image(s)', leave=True) if show_pbar else None
            reserved_params = [
                'weight_path', 'model_name', 'dataset', 'test_name',
                'accumulate_pred_dump', 'stream_writer', 'leave_stream_writer_open'
            ]
            for param in reserved_params:
                assert param not in kwargs, f'{param} already exists in kwargs'
                assert param in infer_func.__annotations__, f"{infer_func.__name__} needs to accept a {param} keyword argument to be wrapped by infer_tests_wrapper"
            for weight_path0, model_name0 in zip(weight_paths, model_names):
                video_save_path = f'{video_dump_dir}/{model_name0}.avi' if video_dump_dir is not None else None
                if stream_writer is None:
                    stream_writer = StreamWriter(
                        show_preview=show_preview,
                        video_save_path=video_save_path,
                        dump_dir=img_dump_dir
                    )
                elif video_save_path is not None:
                    stream_writer.video_writer._save_path = video_save_path
                if img_dump_dir is not None:
                    model_img_dump_dir = f'{img_dump_dir}/{model_name0}'
                    make_dir_if_not_exists(model_img_dump_dir)
                else:
                    model_img_dump_dir = None
                data = handler_constructor()
                assert isinstance(data, BasicLoadableHandler)
                assert hasattr(data, '__add__')
                if video_dump_dir is not None:
                    video_save_path = f'{video_dump_dir}/{model_name0}.avi'
                else:
                    video_save_path = None
                for dataset0, test_name0 in zip(datasets, test_names):
                    if test_pbar is not None:
                        test_pbar.set_description(f'{model_name0} {test_name0}')
                    if img_dump_dir is not None:
                        test_img_dump_dir = f'{model_img_dump_dir}/{test_name0}'
                        make_dir_if_not_exists(test_img_dump_dir)
                        stream_writer.dump_writer._save_dir = test_img_dump_dir
                    kwargs['weight_path'] = weight_path0
                    kwargs['model_name'] = model_name0
                    kwargs['dataset'] = dataset0
                    kwargs['test_name'] = test_name0
                    kwargs['accumulate_pred_dump'] = data_dump_dir is not None
                    kwargs['stream_writer'] = stream_writer
                    kwargs['leave_stream_writer_open'] = True
                    if data_dump_dir is not None:
                        data0 = infer_func(*args, **kwargs)
                        assert isinstance(data0, handler_constructor)
                        data += data0
                    else:
                        infer_func(*args, **kwargs)
                    if test_pbar is not None:
                        test_pbar.update(len(dataset0.images))
                if data_dump_dir is not None:
                    data.save_to_path(f'{data_dump_dir}/{model_name0}.json', overwrite=True)
                if stream_writer is not None and stream_writer.video_writer is not None and stream_writer.video_writer.recorder is not None:
                    stream_writer.video_writer.recorder.close()
                    stream_writer.video_writer.recorder = None
            if test_pbar is not None:
                test_pbar.close()
            if stream_writer is not None:
                del stream_writer
        return _wrapper_inner
    return _wrapper