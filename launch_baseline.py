import argparse
import contextlib
import logging
import os
import sys

from tqdm import tqdm
import torch
from threestudio.utils.base import (
    Updateable,
    update_end_if_possible,
    update_if_possible,
    get_device
)
from threestudio.utils.config import dump_config
from datetime import datetime, timedelta
class ColoredFilter(logging.Filter):
    """
    A logging filter to add color to certain log levels.
    """

    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"

    COLORS = {
        "WARNING": YELLOW,
        "INFO": GREEN,
        "DEBUG": BLUE,
        "CRITICAL": MAGENTA,
        "ERROR": RED,
    }

    RESET = "\x1b[0m"

    def __init__(self):
        super().__init__()

    def filter(self, record):
        if record.levelname in self.COLORS:
            color_start = self.COLORS[record.levelname]
            record.levelname = f"{color_start}[{record.levelname}]"
            record.msg = f"{record.msg}{self.RESET}"
        return True



import time
def to_device(batch, device):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    return batch




def main(args, extras) -> None:
    # set CUDA_VISIBLE_DEVICES if needed, then import pytorch-lightning
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    
    
    import pytorch_lightning as pl
    import torch
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
    from pytorch_lightning.utilities.rank_zero import rank_zero_only

    if args.typecheck:
        from jaxtyping import install_import_hook

        install_import_hook("threestudio", "typeguard.typechecked")

    import threestudio
    from threestudio.systems.base import BaseSystem
    from threestudio.utils.callbacks import (
        CodeSnapshotCallback,
        ConfigSnapshotCallback,
        CustomProgressBar,
        ProgressCallback,
    )
    from threestudio.utils.config import ExperimentConfig, load_config
    from threestudio.utils.misc import get_rank
    from threestudio.utils.typing import Optional
    n_gpus= 1
    device = torch.device(f'cuda:{args.gpu}')
    logger = logging.getLogger("pytorch_lightning")
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    for handler in logger.handlers:
        if handler.stream == sys.stderr:  # type: ignore
            if not args.gradio:
                handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
                handler.addFilter(ColoredFilter())
            else:
                handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    # parse YAML config to OmegaConf
    cfg: ExperimentConfig
    cfg = load_config(args.config, cli_args=extras, n_gpus=n_gpus)

    # set a different seed for each device
    pl.seed_everything(cfg.seed + get_rank(), workers=True)

    dm = threestudio.find(cfg.data_type)(cfg.data)
    system: BaseSystem = threestudio.find(cfg.system_type)(
        cfg.system, get_device(), resumed=cfg.resume is not None
    )
    date_str = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
    cfg.trial_dir = cfg.trial_dir + f'@{date_str}'
    system.set_save_dir(os.path.join(cfg.trial_dir, "save"))

    config_dir = os.path.join(cfg.trial_dir, "configs")
    os.makedirs(config_dir, exist_ok=True)
    dump_config(os.path.join(config_dir, "parsed.yaml"), cfg)
    
    train_loop(system, cfg, dm, device)

import time
def to_device(batch, device):

    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
        elif isinstance(v, dict):
            batch[k] = to_device(batch[k], device)
    return batch

def train_loop(system, config, datamodule, device):
    # other=system.clone()
    train_config = config.trainer
    
    datamodule.setup()
    train_dataset =datamodule.train_dataloader()
    val_dataset =datamodule.val_dataloader()
    test_dataset =datamodule.test_dataloader()
    system.to(device)
    system.on_fit_start(device)
    optimizer = system.configure_optimizers()
    start_time = time.time()
    
    dataiters = datamodule.get_all_train_iters()
    
    last_vis_time = 0
    for global_step in tqdm(range(train_config.max_steps)):
        elapsed = time.time() - start_time
        
        
        batch = next(dataiters.get_current_iter(global_step))
        batch = to_device(batch, device)
        
        update_if_possible(train_dataset.dataset, 0, global_step)
        system.do_update_step(0, global_step)
        
        res = system.training_step(batch, global_step)
        system.do_update_step_end(0, global_step)
        update_end_if_possible( train_dataset.dataset, 0, global_step )
        
        for k in optimizer:
            optimizer[k].zero_grad()
        res['loss'].backward()
        if 'loss_lora' in res:
            res['loss_lora'].backward()
        for k in optimizer:
            optimizer[k].step()
        

        if elapsed >= last_vis_time + train_config.val_check_interval and train_config.visualize_progress:
            elapsed = time.time() - start_time
            system.eval()
            batch = next(iter(val_dataset))
            batch = to_device(batch, device)
            update_if_possible(val_dataset.dataset, 0, global_step)
            system.do_update_step(0, global_step)
            system.validation_step(batch, global_step, [str(timedelta(seconds=elapsed)).split('.')[0]]  if train_config.display_time else None)
            system.do_update_step_end(0, global_step)
            update_end_if_possible( val_dataset.dataset, 0, global_step )
            system.train()
            last_vis_time = elapsed

    elapsed = time.time() - start_time
    system.eval()
    with torch.no_grad():
        for batch in tqdm(test_dataset):

            batch = to_device(batch, device)
            update_if_possible(test_dataset.dataset, 0, global_step)
            system.do_update_step(0, global_step)
            system.test_step(batch, global_step, [str(timedelta(seconds=elapsed)).split('.')[0]]  if train_config.display_time else None)
            system.do_update_step_end(0, global_step)
            update_end_if_possible( test_dataset.dataset, 0, global_step )
    system.on_test_epoch_end(global_step)
    save_dict = {'epoch': 0, 'global_step': global_step, 
                 'state_dict': system.state_dict()}
    torch.save(save_dict, os.path.join(system.get_save_dir(), 'last.ckpt'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument(
        "--gpu",
        default="0",
        help="GPU(s) to be used. 0 means use the 1st available GPU. "
        "1,2 means use the 2nd and 3rd available GPU. "
        "If CUDA_VISIBLE_DEVICES is set before calling `launch.py`, "
        "this argument is ignored and all available GPUs are always used.",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true")
    group.add_argument("--validate", action="store_true")
    group.add_argument("--test", action="store_true")
    group.add_argument("--export", action="store_true")

    parser.add_argument(
        "--gradio", action="store_true", help="if true, run in gradio mode"
    )

    parser.add_argument(
        "--verbose", action="store_true", help="if true, set logging level to DEBUG"
    )

    parser.add_argument(
        "--typecheck",
        action="store_true",
        help="whether to enable dynamic type checking",
    )

    args, extras = parser.parse_known_args()

    if args.gradio:
        # FIXME: no effect, stdout is not captured
        with contextlib.redirect_stdout(sys.stderr):
            main(args, extras)
    else:
        main(args, extras)
        
