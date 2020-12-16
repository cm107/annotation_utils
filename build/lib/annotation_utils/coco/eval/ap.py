from __future__ import annotations
from typing import cast, List
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from common_utils.base.basic import BasicLoadableObject, BasicLoadableHandler, BasicHandler
from ..structs.dataset import COCO_Dataset

class AP_Result(BasicLoadableObject['AP_Result']):
    def __init__(
        self, model_name: str, test_name: str, ann_type: str,
        ap: float, ap_50: float, ap_75: float,
        ap_s: float, ap_m: float, ap_l: float,
        ar: float, ar_50: float, ar_75: float,
        ar_s: float, ar_m: float, ar_l: float,
        prec50: List[float]=None, prec75: List[float]=None, rec: List[float]=None
    ):
        super().__init__()
        self.model_name = model_name
        self.test_name = test_name
        self.ann_type = ann_type
        self.ap = ap
        self.ap_50 = ap_50
        self.ap_75 = ap_75
        self.ap_s = ap_s
        self.ap_m = ap_m
        self.ap_l = ap_l
        self.ar = ar
        self.ar_50 = ar_50
        self.ar_75 = ar_75
        self.ar_s = ar_s
        self.ar_m = ar_m
        self.ar_l = ar_l
        self.prec50 = prec50
        self.prec75 = prec75
        self.rec = rec

    @classmethod
    def _get_precision_and_recall(cls, cocoEval: COCOeval, areaRng: str='all', maxDets: int=100, iouThr: float=None) -> List[float]:
        aind = [i for i, aRng in enumerate(cocoEval.params.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(cocoEval.params.maxDets) if mDet == maxDets]
        if iouThr is not None:
            t = np.where(iouThr == cocoEval.params.iouThrs)[0]
        precision = cocoEval.eval['precision']
        if iouThr is not None:
            precision = precision[t]
        precision = precision[:,:,:,aind,mind]
        precision = precision.reshape(-1).tolist()
        recall = cocoEval.eval['recall']
        if iouThr is not None:
            recall = recall[t]
        recall = recall[:,:,aind,mind]
        recall = recall.reshape(-1).tolist()
        return precision, recall

    @classmethod
    def from_cocoEval(cls, cocoEval: COCOeval, model_name: str, test_name: str, ann_type: str) -> AP_Result:
        assert hasattr(cocoEval, 'stats')
        precision50, recall50 = cls._get_precision_and_recall(
            cocoEval=cocoEval,
            areaRng='all',
            maxDets=100 if ann_type == 'bbox' else 20, # 100 for bbox, 20 for keypoints
            iouThr=0.5
        )
        precision75, recall75 = cls._get_precision_and_recall(
            cocoEval=cocoEval,
            areaRng='all',
            maxDets=100 if ann_type == 'bbox' else 20, # 100 for bbox, 20 for keypoints
            iouThr=0.75
        )
        if ann_type == 'bbox':
            return AP_Result(
                model_name=model_name, test_name=test_name, ann_type=ann_type,
                ap=cocoEval.stats[0],
                ap_50=cocoEval.stats[1],
                ap_75=cocoEval.stats[2],
                ap_s=cocoEval.stats[3],
                ap_m=cocoEval.stats[4],
                ap_l=cocoEval.stats[5],
                ar=cocoEval.stats[6],
                ar_50=cocoEval.stats[7],
                ar_75=cocoEval.stats[8],
                ar_s=cocoEval.stats[9],
                ar_m=cocoEval.stats[10],
                ar_l=cocoEval.stats[11],
                prec50=precision50,
                prec75=precision75,
                rec=[val for val in np.arange(0,1.01,0.01).tolist()]
            )
        elif ann_type == 'keypoints':
            return AP_Result(
                model_name=model_name, test_name=test_name, ann_type=ann_type,
                ap=cocoEval.stats[0],
                ap_50=cocoEval.stats[1],
                ap_75=cocoEval.stats[2],
                ap_s=None,
                ap_m=cocoEval.stats[3],
                ap_l=cocoEval.stats[4],
                ar=cocoEval.stats[5],
                ar_50=cocoEval.stats[6],
                ar_75=cocoEval.stats[7],
                ar_s=None,
                ar_m=cocoEval.stats[8],
                ar_l=cocoEval.stats[9],
                prec50=precision50,
                prec75=precision75,
                rec=[val for val in np.arange(0,1.01,0.01).tolist()]
            )
        else:
            raise Exception(f'Invalid ann_type: {ann_type}')

    def save_pr_curve(self, save_path: str, target: str='pr50'):
        if target == 'pr50':
            p_vals = self.prec50
        elif target == 'pr75':
            p_vals = self.prec75
        else:
            raise ValueError
        assert p_vals is not None
        assert len(p_vals) == len(self.rec), f'len(p_vals)=={len(p_vals)}!={len(self.rec)}==len(self.rec)'
        data = {
            'Precision': p_vals,
            'Recall': self.rec
        }
        ax = sns.lineplot(x='Recall', y='Precision', data=data)
        plt.savefig(save_path)
        plt.clf()
        plt.close('all')

class AP_Result_List(
    BasicLoadableHandler['AP_Result_List', 'AP_Result'],
    BasicHandler['AP_Result_List', 'AP_Result']
):
    def __init__(self, result_list: List[AP_Result]=None):
        super().__init__(obj_type=AP_Result, obj_list=result_list)
        self.result_list = self.obj_list
    
    @classmethod
    def from_dict_list(cls, dict_list: List[dict]) -> AP_Result_List:
        return AP_Result_List([AP_Result.from_dict(item_dict) for item_dict in dict_list])

    @property
    def bbox_results(self) -> AP_Result_List:
        return AP_Result_List([result for result in self if result.ann_type == 'bbox'])

    @property
    def keypoint_results(self) -> AP_Result_List:
        return AP_Result_List([result for result in self if result.ann_type == 'keypoints'])

    @property
    def model_names(self) -> List[str]:
        result = [datum.model_name for datum in self]
        result.sort()
        return result

    @property
    def test_names(self) -> List[str]:
        result = [datum.test_name for datum in self]
        result.sort()
        return result

    def get_model(self, model_name: str) -> AP_Result_List:
        if isinstance(model_name, str):
            return AP_Result_List([result for result in self if result.model_name == model_name])
        elif isinstance(model_name, (tuple, list)):
            return AP_Result_List([result for result in self if result.model_name in model_name])
        else:
            raise TypeError

    def get_test(self, test_name: str) -> AP_Result_List:
        if isinstance(test_name, str):
            return AP_Result_List([result for result in self if result.test_name == test_name])
        elif isinstance(test_name, (tuple, list)):
            return AP_Result_List([result for result in self if result.test_name in test_name])
        else:
            raise TypeError

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(self.to_dict_list())

    @classmethod
    def from_paths(
        cls, dt: BasicLoadableHandler,
        datasets: List[COCO_Dataset],
        test_names: List[str],
        model_names: List[str]=None,
        ann_types: List[str]=['bbox', 'keypoints']
    ) -> AP_Result_List:
        assert len(datasets) == len(test_names)
        for dataset in datasets:
            if isinstance(dataset, (COCO_Dataset, str)):
                pass
            else:
                raise TypeError
        ap_result_list = AP_Result_List()
        if model_names is None:
            model_names0 = list(set([datum.model_name for datum in dt]))
        else:
            model_names0 = model_names.copy()
        for model_name in model_names0:
            for dataset, test_name in zip(datasets, test_names):
                print(f'{model_name} {test_name}')
                if isinstance(dataset, str):
                    cocoGt = COCO(dataset)
                    dataset0 = COCO_Dataset.load_from_path(dataset, check_paths=False, strict=False)
                elif isinstance(dataset, COCO_Dataset):
                    dataset.save_to_path('/tmp/temp_dataset.json', overwrite=True)
                    cocoGt = COCO('/tmp/temp_dataset.json')
                    dataset0 = dataset
                else:
                    raise Exception
                results = dt.to_coco_result(gt_dataset=dataset0, model_name=model_name, test_name=test_name)
                if len(results) == 0:
                    print(f'\tNo results found for model_name={model_name}, test_name={test_name}. Skipping.')
                    continue
                cocoDt = cocoGt.loadRes(results.to_dict_list())
                imgIds = sorted(cocoGt.getImgIds())

                for ann_type in ann_types:
                    cocoEval = COCOeval(cocoGt, cocoDt, ann_type)
                    cocoEval.params.imgIds = imgIds
                    if ann_type == 'keypoints':
                        num_keypoints = len(gt_dataset.categories[0].keypoints)
                        cocoEval.params.kpt_oks_sigmas = np.array([0.79]*num_keypoints)/10.0
                    cocoEval.evaluate()
                    cocoEval.accumulate()
                    cocoEval.summarize()
                    ap_result = AP_Result.from_cocoEval(
                        cocoEval=cocoEval, model_name=model_name,
                        test_name=test_name, ann_type=ann_type
                    )
                    ap_result_list.append(ap_result)
        return ap_result_list

    @property
    def ann_type_name_map(self) -> dict:
        return {
            'bbox': 'BBox',
            'keypoints': 'Keypoints'
        }

    @property
    def ap_target_name_map(self) -> dict:
        return {
            'ap': 'AP',
            'ap_50': 'AP50',
            'ap_75': 'AP75',
            'ap_s': 'AP_Small',
            'ap_m': 'AP_Medium',
            'ap_l': 'AP_Large',
            'ar': 'AR',
            'ar_50': 'AR50',
            'ar_75': 'AR75',
            'ar_s': 'AR_Small',
            'ar_m': 'AR_Medium',
            'ar_l': 'AR_Large'
        }

    def plotly_show(self, test_targets: List[str], model_targets: List[str], ann_type: str):
        fig = make_subplots(rows=1, cols=1)

        if ann_type == 'bbox':
            ap_res_list = self.bbox_results
        elif ann_type == 'keypoints':
            ap_res_list = self.keypoint_results
        else:
            raise ValueError

        for test_target in test_targets:
            y = []
            for model_target in model_targets:
                val = ap_res_list.get_test(test_target).get_model(model_target)[0].ap
                y.append(val)
            fig.add_trace(
                go.Bar(x=model_targets, y=y, name=test_target),
                row=1, col=1
            )
        fig.update_layout(title_text=f"COCO {ann_type} AP", template="seaborn")
        fig.show()

    def save_plot(
        self, test_targets: List[str], model_targets: List[str], ann_type: str, save_path: str,
        model_target_aliases: List[str]=None,
        ap_target: str='ap',
        legend_prop: float=0.7,
        xlabel: str=None, ylabel: str=None, title: str=None,
        xticklabel_fontsize: int=6, xticklabel_rotation: int=13,
        show_bar_values: bool=False, bar_values_fontsize: int=6,
        combine_test_targets: bool=False
    ):
        if ann_type == 'bbox':
            ap_res_list = self.bbox_results
        elif ann_type == 'keypoints':
            ap_res_list = self.keypoint_results
        else:
            raise ValueError

        data = ap_res_list.get_model(model_targets).get_test(test_targets)
        if not combine_test_targets:
            data = data.to_df()
        else:
            data0 = AP_Result_List()
            for model_name in data.model_names:                
                ap_res = AP_Result(
                    model_name=model_name,
                    test_name=None,
                    ann_type=ann_type,
                    ap=float(np.mean([datum.ap for datum in data if datum.model_name == model_name])),
                    ap_50=float(np.mean([datum.ap_50 for datum in data if datum.model_name == model_name])),
                    ap_75=float(np.mean([datum.ap_75 for datum in data if datum.model_name == model_name])),
                    ap_s=float(np.mean([datum.ap_s for datum in data if datum.model_name == model_name])),
                    ap_m=float(np.mean([datum.ap_m for datum in data if datum.model_name == model_name])),
                    ap_l=float(np.mean([datum.ap_l for datum in data if datum.model_name == model_name])),
                    ar=float(np.mean([datum.ar for datum in data if datum.model_name == model_name])),
                    ar_50=float(np.mean([datum.ar_50 for datum in data if datum.model_name == model_name])),
                    ar_75=float(np.mean([datum.ar_75 for datum in data if datum.model_name == model_name])),
                    ar_s=float(np.mean([datum.ar_s for datum in data if datum.model_name == model_name])),
                    ar_m=float(np.mean([datum.ar_m for datum in data if datum.model_name == model_name])),
                    ar_l=float(np.mean([datum.ar_l for datum in data if datum.model_name == model_name])),
                    prec50=None,
                    prec75=None,
                    rec=None
                )
                data0.append(ap_res)
            data = data0.to_df()
        if data.empty:
            return

        fig = make_subplots(rows=1, cols=1)
        fig = plt.figure()

        ax = sns.barplot(x='model_name', y=ap_target, hue='test_name' if not combine_test_targets else None, data=data, order=model_targets)
        if show_bar_values:
            for p in ax.patches:
                text = f'{int(p.get_height()*100)}%' if not np.isnan(p.get_height()) else f'0%'
                ax.annotate(
                    text,
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 5),
                    textcoords='offset points',
                    fontsize=bar_values_fontsize
                )
        
        if xlabel is not None:
            plt.xlabel(xlabel)
        else:
            plt.xlabel('Model')
        if ylabel is not None:
            plt.ylabel(ylabel)
        else:
            plt.ylabel(self.ap_target_name_map[ap_target])
        if title is not None:
            plt.title(title)
        else:
            plt.title(f'{self.ann_type_name_map[ann_type]} {self.ap_target_name_map[ap_target]}')
        
        plt.ylim(0,1)
        for label in ax.get_xticklabels():
            label.set_fontsize(xticklabel_fontsize)
            label.set_rotation(xticklabel_rotation)

        if not combine_test_targets:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * legend_prop, box.height])
            legend = ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        
        if model_target_aliases is not None:
            assert len(model_target_aliases) == len(ax.get_xticklabels()), f'model_target_aliases:\n{model_target_aliases}\nax.get_xticklabels():\n{ax.get_xticklabels()}'
            ax.set_xticklabels(model_target_aliases)

        plt.savefig(save_path)
        plt.clf()
        plt.close('all')
    
    def save_pr_plot(
        self, test_targets: List[str], model_targets: List[str], ann_type: str, save_path: str,
        pr_target: str='pr50', legend_prop: float=0.8
    ):
        test_targets = test_targets if isinstance(test_targets, (list, tuple)) else [test_targets]
        model_targets = model_targets if isinstance(model_targets, (list, tuple)) else [model_targets]
        assert len(test_targets) == 1 or len(model_targets) == 1, f'Either the test_target or model_target must be fixed.'

        if ann_type == 'bbox':
            ap_res_list = self.bbox_results
        elif ann_type == 'keypoints':
            ap_res_list = self.keypoint_results
        else:
            raise ValueError
        ap_res_list = ap_res_list.get_model(model_targets).get_test(test_targets)
        hue_target = 'test_name' if len(test_targets) > 1 else 'model_name'
        data = {
            'Precision': [],
            'Recall': [],
            'Target': []
        }
        for ap_res in ap_res_list:
            if pr_target == 'pr50':
                data['Precision'].extend(ap_res.prec50)
                data['Recall'].extend(ap_res.rec)
            elif pr_target == 'pr75':
                data['Precision'].extend(ap_res.prec75)
                data['Recall'].extend(ap_res.rec)
            else:
                raise Exception
            if hue_target == 'test_name':
                data['Target'].extend([ap_res.test_name]*len(ap_res.rec))
            elif hue_target == 'model_name':
                data['Target'].extend([ap_res.model_name]*len(ap_res.rec))
            else:
                raise Exception
        ax = sns.lineplot(x='Recall', y='Precision', hue='Target', data=data)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        if hue_target == 'test_name':
            plt.title(f'{model_targets[0]} {pr_target.upper()}')
        elif hue_target == 'model_name':
            plt.title(f'{test_targets[0]} {pr_target.upper()}')
        else:
            raise Exception
        plt.ylim(0,1)

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * legend_prop, box.height])
        legend = ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig(save_path)
        plt.clf()
        plt.close('all')