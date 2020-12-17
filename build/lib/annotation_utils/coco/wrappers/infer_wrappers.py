from typing import cast, List, Dict
from tqdm import tqdm
import cv2
import numpy as np
from common_utils.file_utils import make_dir_if_not_exists, delete_all_files_in_dir, \
    dir_exists, file_exists
from common_utils.base.basic import BasicLoadableHandler
from common_utils.cv_drawing_utils import draw_text_rows_at_point
from common_utils.image_utils import collage_from_img_buffer
from streamer.recorder.stream_writer import StreamWriter
from ..structs.dataset import COCO_Dataset

def infer_tests_wrapper(
    weight_path: str, model_name: str, dataset: COCO_Dataset, test_name: str,
    handler_constructor: type,
    data_dump_dir: str=None, video_dump_dir: str=None, img_dump_dir: str=None,
    skip_if_data_dump_exists: bool=False,
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
            if isinstance(weight_path, (str, dict)):
                weight_paths = [weight_path]
            elif isinstance(weight_path, (tuple, list)):
                assert all([type(part) in [str, dict] for part in weight_path])
                for part in weight_path:
                    if isinstance(part, dict):
                        for key, val in part.items():
                            assert isinstance(val, str)
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
                # delete_all_files_in_dir(data_dump_dir, ask_permission=True)
            if video_dump_dir is not None:
                make_dir_if_not_exists(video_dump_dir)
                # delete_all_files_in_dir(video_dump_dir, ask_permission=True)
            if img_dump_dir is not None:
                make_dir_if_not_exists(img_dump_dir)
                # delete_all_files_in_dir(img_dump_dir, ask_permission=True)
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
                data_dump_save = f'{data_dump_dir}/{model_name0}.json' if data_dump_dir is not None else None
                if data_dump_save is not None and file_exists(data_dump_save) and skip_if_data_dump_exists:
                    if test_pbar is not None:
                        for dataset0, test_name0 in zip(datasets, test_names):
                            test_pbar.update(len(dataset0.images))
                    continue
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
                # if video_dump_dir is not None:
                #     video_save_path = f'{video_dump_dir}/{model_name0}.avi'
                # else:
                #     video_save_path = None
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
                        assert isinstance(data0, handler_constructor), f"Encountered dump data of type {type(data0).__name__}. Expected {handler_constructor.__name__}."
                        data += data0
                    else:
                        infer_func(*args, **kwargs)
                    if test_pbar is not None:
                        test_pbar.update(len(dataset0.images))
                if data_dump_dir is not None:
                    data.save_to_path(data_dump_save, overwrite=True)
                if stream_writer is not None and stream_writer.video_writer is not None and stream_writer.video_writer.recorder is not None:
                    stream_writer.video_writer.recorder.close()
                    stream_writer.video_writer.recorder = None
            if test_pbar is not None:
                test_pbar.close()
            if stream_writer is not None:
                del stream_writer
        return _wrapper_inner
    return _wrapper

def gen_infer_comparison(
    gt: BasicLoadableHandler, dt: BasicLoadableHandler, error: BasicLoadableHandler,
    model_names: List[str], test_names: List[str],
    collage_shape: (int, int),
    test_img_dir_map: Dict[str, str]=None,
    model_aliases: Dict[str, str]=None, test_aliases: Dict[str, str]=None,
    video_save: str=None, img_dump_dir: str=None, show_preview: bool=False,
    show_pbar: bool=True,
    draw_settings=None, draw_inference: bool=False,
    details_func=None, debug_verbose: bool=False
):
    for handler in [gt, dt, error]:
        assert isinstance(handler, BasicLoadableHandler)
        for attr_key in ['frame', 'test_name']:
            assert hasattr(handler[0], attr_key)
    for handler in [dt, error]:
        assert hasattr(handler[0], 'model_name')
    model_names0 = list(set([datum.model_name for datum in dt])) if model_names == 'all' else model_names
    test_names0 = list(set([datum.test_name for datum in gt])) if test_names == 'all' else test_names
    for val_list in [model_names0, test_names0]:
        if val_list != 'all':
            assert isinstance(val_list, (tuple, list))
            for val in val_list:
                assert isinstance(val, str)
    assert isinstance(collage_shape, (tuple, list))
    for val in collage_shape:
        assert isinstance(val, int)
    assert len(collage_shape) == 2
    assert len(model_names0) <= collage_shape[0] * collage_shape[1]
    if img_dump_dir is not None:
        make_dir_if_not_exists(img_dump_dir)
        delete_all_files_in_dir(img_dump_dir, ask_permission=False)
    if test_img_dir_map is None:
        test_img_dir_map0 = {test_name: test_name for test_name in test_names0}
    else:
        assert isinstance(test_img_dir_map, dict)
        for key, val in test_img_dir_map.items():
            assert key in test_names0
            assert isinstance(key, str)
            assert isinstance(val, str)
        test_img_dir_map0 = {
            test_name: (test_img_dir_map[test_name] if test_name in test_img_dir_map else test_name) \
            for test_name in test_names0
        }
    for test_name, img_dir in test_img_dir_map0.items():
        if not dir_exists(img_dir):
            raise FileNotFoundError(
                f"""
                Couldn't find image directory {img_dir} for {test_name}.
                Please modify test_img_dir_map to match the image directory path for {test_name}.
                test_img_dir_map: {test_img_dir_map0}
                """
            )
    stream_writer = StreamWriter(show_preview=show_preview, video_save_path=video_save, dump_dir=img_dump_dir)

    total_images = len(gt.get(test_name=test_names0))
    pbar = tqdm(total=total_images, unit='image(s)', leave=True) if show_pbar else None
    if pbar is not None:
        pbar.set_description('Generating Comparison')
    for test_name in test_names0:
        if img_dump_dir is not None:
            test_img_dump_dir = f'{img_dump_dir}/{test_name}'
            make_dir_if_not_exists(test_img_dump_dir)
            stream_writer.dump_writer._save_dir = test_img_dump_dir

        img_dir = test_img_dir_map0[test_name]
        gt_test_data = gt.get(test_name=test_name)
        gt_test_data.sort(attr_name='frame')
        dt_test_data = dt.get(test_name=test_name)
        dt_test_data.sort(attr_name='frame')
        error_test_data = error.get(test_name=test_name)
        error_test_data.sort(attr_name='frame')
        
        for gt_datum in gt_test_data:
            file_name = gt_datum.frame
            img_path = f'{img_dir}/{file_name}'
            if not file_exists(img_path):
                if debug_verbose:
                    print(
                        f"""
                        Couldn't find image. Skipping.
                            test_name: {test_name}
                            img_path: {img_path}
                        """
                    )
                pbar.update()
                continue
            img = cv2.imread(img_path)
            dt_frame_data = dt_test_data.get(frame=gt_datum.frame)
            error_frame_data = error_test_data.get(frame=gt_datum.frame)

            img_buffer = cast(List[np.ndarray], [])
            for model_name in model_names0:
                dt_model_data = dt_frame_data.get(model_name=model_name)
                dt_model_datum = dt_model_data[0] if len(dt_model_data) > 0 else None
                error_model_data = error_frame_data.get(model_name=model_name)
                error_datum = error_model_data[0] if len(error_model_data) > 0 else None

                result = img.copy()
                if draw_inference or draw_settings is not None:
                    if draw_settings is not None:
                        if dt_model_datum is not None:
                            result = dt_model_datum.draw(result, settings=draw_settings)
                    else:
                        if dt_model_datum is not None:
                            result = dt_model_datum.draw(result)
                
                if test_aliases is not None and gt_datum.test_name in test_aliases:
                    test_text = test_aliases[gt_datum.test_name]
                else:
                    test_text = gt_datum.test_name if gt_datum is not None else None
                if model_aliases is not None and dt_model_datum.model_name in model_aliases:
                    model_text = model_aliases[dt_model_datum.model_name]
                else:
                    model_text = dt_model_datum.model_name if dt_model_datum is not None else None

                if details_func is not None:
                    for key in ['gt', 'dt', 'error']:
                        assert key in details_func.__annotations__, f'{details_func.__name__} must have a {key} parameter.'
                    details_func_params = {'img': result, 'gt': gt_datum, 'dt': dt_model_datum, 'error': error_datum}
                    suggested_params = {'test_text': test_text, 'model_text': model_text, 'frame_text': gt_datum.frame}
                    for key, val in suggested_params.items():
                        if key in details_func.__annotations__:
                            details_func_params[key] = val
                    result = details_func(**details_func_params)
                else:
                    row_text_list = [
                        f'Test: {test_text}',
                        f'Model: {model_text}',
                        f'Frame: {gt_datum.frame}'
                    ]
                    result_h, result_w = result.shape[:2]
                    combined_row_height = len(row_text_list)*0.04*result_h
                    result = draw_text_rows_at_point(
                        img=result,
                        row_text_list=row_text_list,
                        x=result_w*0.01, y=result_h*0.01,
                        combined_row_height=combined_row_height
                    )
                img_buffer.append(result)
            collage_img = collage_from_img_buffer(
                img_buffer=img_buffer, collage_shape=collage_shape
            )
            stream_writer.step(img=collage_img, file_name=file_name)
            if pbar is not None:
                pbar.update()
    if pbar is not None:
        pbar.close()