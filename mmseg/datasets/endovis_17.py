import os
import os.path as osp
from .builder import DATASETS
from .custom import CustomDataset
import mmcv
import numpy as np

@DATASETS.register_module()
class EndoVis17Dataset(CustomDataset):

    CLASSES = (
        'bipolar_forceps',
        'prograsp_forceps',
        'large_needle_driver',
        'vessel_sealer',
        'grasping_retractor',
        'monopolar_curved_scissors',
        'others'
    )

    PALETTE = [
        [128, 0, 0],      # bipolar
        [0, 128, 0],      # prograsp
        [128, 128, 0],    # LND
        [0, 0, 128],      # vessel sealer
        [128, 0, 128],    # grasping retractor
        [0, 128, 128],    # MCS
        [128, 128, 128]   # others
    ]

    def __init__(self, convert2source=False, **kwargs):
        self.convert2source = convert2source
        super(EndoVis17Dataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=True,
            **kwargs)

    def get_gt_seg_maps(self, efficient_test=False, convert2source=False):
        """Get ground truth segmentation maps for evaluation."""
        gt_seg_maps = []
        for img_info in self.img_infos:
            seg_map = osp.join(self.ann_dir, img_info['ann']['seg_map'])
            if efficient_test:
                gt_seg_map = seg_map
            else:
                gt_seg_map = mmcv.imread(seg_map, flag='unchanged', backend='pillow')

                if convert2source:
                    gt_seg_map = gt_seg_map.astype('int32')

                    # Keep only 1,2,3,4 → others become 0
                    valid_mask = np.isin(gt_seg_map, [1, 2, 3, 4])
                    gt_seg_map = np.where(valid_mask, gt_seg_map, 0)

                    # Then remap 4 → 6
                    gt_seg_map[gt_seg_map == 4] = 6

            gt_seg_maps.append(gt_seg_map)
        return gt_seg_maps


    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 efficient_test=False,
                 **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU' and
                'mDice' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: Default metrics.
        """
        from functools import reduce
        from mmseg.core import eval_metrics
        from mmcv.utils import print_log
        from terminaltables import AsciiTable

        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))
        eval_results = {}
        print(f"convert2source={self.convert2source}")
        gt_seg_maps = self.get_gt_seg_maps(
            efficient_test, convert2source=self.convert2source)

        if self.CLASSES is None:
            num_classes = len(
                reduce(np.union1d, [np.unique(_) for _ in gt_seg_maps]))
        else:
            num_classes = len(self.CLASSES)
        ret_metrics = eval_metrics(
            results,
            gt_seg_maps,
            num_classes,
            self.ignore_index,
            metric,
            label_map=self.label_map,
            reduce_zero_label=self.reduce_zero_label)
        class_table_data = [['Class'] + [m[1:] for m in metric] + ['Acc']]
        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES
        ret_metrics_round = [
            np.round(ret_metric * 100, 2) for ret_metric in ret_metrics
        ]
        for i in range(num_classes):
            class_table_data.append([class_names[i]] +
                                    [m[i] for m in ret_metrics_round[2:]] +
                                    [ret_metrics_round[1][i]])
        summary_table_data = [['Scope'] +
                              ['m' + head
                               for head in class_table_data[0][1:]] + ['aAcc']]
        ret_metrics_mean = [
            np.round(np.nanmean(ret_metric) * 100, 2)
            for ret_metric in ret_metrics
        ]
        summary_table_data.append(['global'] + ret_metrics_mean[2:] +
                                  [ret_metrics_mean[1]] +
                                  [ret_metrics_mean[0]])
        print_log('per class results:', logger)
        table = AsciiTable(class_table_data)
        print_log('\n' + table.table, logger=logger)
        print_log('Summary:', logger)
        table = AsciiTable(summary_table_data)
        print_log('\n' + table.table, logger=logger)

        for i in range(1, len(summary_table_data[0])):
            eval_results[summary_table_data[0]
                         [i]] = summary_table_data[1][i] / 100.0
        if mmcv.is_list_of(results, str):
            for file_name in results:
                os.remove(file_name)
        return eval_results

