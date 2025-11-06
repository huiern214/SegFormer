import random
import warnings
import numpy as np

# 🩹 Fix for deprecated np.float / np.int / np.bool in newer NumPy
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'bool'):
    np.bool = bool

import torch
import torch.nn.parallel._functions as torch_parallel_functions

# ⚙️ Patch PyTorch internal function
if hasattr(torch_parallel_functions, "_get_stream"):
    _orig_get_stream = torch_parallel_functions._get_stream

    def _patched_get_stream(device):
        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        return _orig_get_stream(device)

    torch_parallel_functions._get_stream = _patched_get_stream
    print("[Patch] Patched torch.nn.parallel._functions._get_stream")

# ⚙️ Patch MMCV's copy as well
try:
    import mmcv.parallel._functions as mmcv_parallel_functions
    if hasattr(mmcv_parallel_functions, "_get_stream"):
        _orig_mmcv_get_stream = mmcv_parallel_functions._get_stream

        def _patched_mmcv_get_stream(device):
            if isinstance(device, int):
                device = torch.device(f"cuda:{device}")
            return _orig_mmcv_get_stream(device)

        mmcv_parallel_functions._get_stream = _patched_mmcv_get_stream
        print("[Patch] Patched mmcv.parallel._functions._get_stream")
except ImportError:
    print("[Patch] mmcv.parallel._functions not found yet; safe to ignore.")

from mmcv.runner import hooks

# 🩹 Patch TextLoggerHook to safely handle missing keys like data_time / time
if hasattr(hooks, "TextLoggerHook"):
    _orig_log_info = hooks.TextLoggerHook._log_info

    def _safe_log_info(self, log_dict, runner):
        # ensure required keys exist
        log_dict.setdefault("data_time", 0.0)
        log_dict.setdefault("time", 0.0)
        return _orig_log_info(self, log_dict, runner)

    hooks.TextLoggerHook._log_info = _safe_log_info
    print("[Patch] Safe TextLoggerHook._log_info applied.")


from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import build_optimizer, build_runner

from mmseg.core import DistEvalHook, EvalHook
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.utils import get_root_logger


def set_random_seed(seed, deterministic=False):
    """Set random seed.
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_segmentor(model,
                    dataset,
                    cfg,
                    distributed=False,
                    validate=False,
                    timestamp=None,
                    meta=None):
    """Launch segmentor training."""
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            drop_last=True) for ds in dataset
    ]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        # device = torch.device('cuda:0')
        # model = model.to(device)


        # model = MMDataParallel(model.cuda(), device_ids=[0])
        # model = MMDataParallel(model.cuda(torch.device('cuda:0')), device_ids=[torch.device('cuda:0')])

        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=[0])

        # model = MMDataParallel(
        #     model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    if cfg.get('runner') is None:
        cfg.runner = {'type': 'IterBasedRunner', 'max_iters': cfg.total_iters}
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # register hooks
    runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))

    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    # register eval hooks
    if validate:
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)