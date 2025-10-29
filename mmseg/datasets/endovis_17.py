from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class EndoVis17Dataset(CustomDataset):

    CLASSES = (
        'background',
        'bipolar_forceps',
        'prograsp_forceps',
        'large_needle_driver',
        'vessel_sealer',
        'grasping_retractor',
        'monopolar_curved_scissors',
        'others'
    )

    PALETTE = [
        [0, 0, 0],        # background
        [128, 0, 0],      # bipolar
        [0, 128, 0],      # prograsp
        [128, 128, 0],    # LND
        [0, 0, 128],      # vessel sealer
        [128, 0, 128],    # grasping retractor
        [0, 128, 128],    # MCS
        [128, 128, 128]   # others
    ]

    def __init__(self, **kwargs):
        super(EndoVis17Dataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=True,
            **kwargs)
