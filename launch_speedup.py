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
)

import time

from datetime import datetime, timedelta

from threestudio.utils.config import dump_config
import numpy as np
logging.getLogger("lightning").setLevel(logging.ERROR)


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

def run(rank, total_ranks, queues):

    import argparse
    from threestudio.utils.config import ExperimentConfig, load_config
    
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



    # parse YAML config to OmegaConf
    config: ExperimentConfig
    config = load_config(args.config, cli_args=extras)

    device = torch.device(f"cuda:{rank}")
    print('Start process ', rank)

    if rank == 0:

        system,  dm, dataiters = prepare_train(args, extras, device, config, rank=rank)

        if args.test:
            system.load_state_dict(torch.load(config.resume, map_location='cuda:0')['state_dict'], strict=False)
            test_dataset =dm.test_dataloader()
            system.eval()
            global_step = config.trainer.max_steps
            with torch.no_grad():
                for batch in tqdm(test_dataset):
                    batch = to_device(batch, device)
                    update_if_possible(test_dataset.dataset, 0, global_step)
                    system.do_update_step(0, global_step)
                    system.test_step(batch, global_step)
                    system.do_update_step_end(0, global_step)
                    update_end_if_possible( test_dataset.dataset, 0, global_step )
            system.on_test_epoch_end(global_step)
                    
            return
        train_loop(system, config, device, dm, queues, config.seed)
        for i in range(total_ranks - 1):
            queues[0].put(None)
    else:

        if args.test:
            
            return
        system, dm, dataiters = prepare_train(args, extras, device, config,rank=rank)
        optimizers = system.configure_optimizers()
        
        run_worker(system, optimizers, dataiters, queues, device, config.seed)

def main():
    import torch.multiprocessing as mp

    torch.autograd.set_detect_anomaly(True)
    mp.set_start_method('spawn', force=True)
    # torch.set_num_threads(1)
    queues = mp.Queue(), mp.Queue(), mp.Queue()

    processes = []
    num_processes = torch.cuda.device_count()
    if num_processes == 1:
        run(0,1,queues)
        exit(0)

    
    for rank in range(num_processes):
        p = mp.Process(target=run, args=(rank, num_processes, queues))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()    # wait for all subprocesses to finish

def prepare_train(args, extras, device, cfg, rank=0) -> None:
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    if args.typecheck:
        from jaxtyping import install_import_hook

        install_import_hook("threestudio", "typeguard.typechecked")

    import threestudio
    from threestudio.systems.base import BaseSystem
    
    
    dm = threestudio.find(cfg.data_type)(cfg.data)
    dm.setup()
    if rank > 0:
        dataiters = dm.get_all_train_iters()
    else:
        dataiters = None
    if 'loggers' in cfg.system:
        cfg.system.pop('loggers')
    system: BaseSystem = threestudio.find(cfg.system_type)(
        cfg.system, device, resumed=cfg.resume is not None
    )

    if rank == 0:

        date_str = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
        cfg.trial_dir = cfg.trial_dir + f'@{date_str}'
        save_dir = os.path.join(cfg.trial_dir, "save")
        system.set_save_dir(save_dir)
        config_dir = os.path.join(cfg.trial_dir, "configs")
        os.makedirs(config_dir, exist_ok=True)
        dump_config(os.path.join(config_dir, "parsed.yaml"), cfg)
    
    system.to(device)
    system.on_fit_start(device)
    system.do_update_step(0, 0)
    system.do_update_step_end(0, 0)
    return system, dm, dataiters

def to_device(batch, device):

    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
        elif isinstance(v, dict):
            batch[k] = to_device(batch[k], device)
    return batch


def run_worker(system, optimizers,dataiters, queues, device, seed_offset):
    while True:
        ret = queues[0].get()
        if ret is None:
            return
    

        (params, step) = ret
        
        system.set_params(params, clone=False)
        
        res = take_step(system, optimizers, dataiters, device, step, seed_offset)
        myparams = system.get_params(include_occs=False)
        
        grads = [para.grad for para in myparams]
        
        queues[1].put(
            (grads, step)
        )
        if 'loss_lora' in res:
            optimizers['optimizer_guidance'].zero_grad()
            res['loss_lora'].backward()
            optimizers['optimizer_guidance'].step()

def take_step(system, optimizers, dataiters, device, step, seed_offset):

    np.random.seed(step+seed_offset)
    torch.manual_seed(step+seed_offset)
    torch.cuda.manual_seed(step+seed_offset)
    batch = next(dataiters.get_current_iter(step))
    batch = to_device(batch, device)
    
    system.do_update_step(0, step)
    res = system.training_step(batch, step)
    system.do_update_step_end(0, step)
    optimizers['optimizer'].zero_grad()
    res['loss'].backward()

    return res
    


def optimizer_state_clone(optimizer_from, optimizer_to):
    optimizer_to.load_state_dict(optimizer_from.state_dict())

def train_loop(system, config, device, datamodule, queues, seed_offset):
    
    train_config, speedup_config = config.trainer, config.speedup
    
    save_dir = system.get_save_dir()
    
    test_dataset =datamodule.test_dataloader()
    val_dataset =datamodule.val_dataloader()
    

    thresh = speedup_config.threshold
    T = train_config.max_steps
    P = speedup_config.P

    systems = [None for _ in range(T+1)]
    optimizers = [None for _ in range(T+1)]
    

    begin_idx, end_idx = 0, P
    total_iters = 0
    start_time = time.time()
    pbar = tqdm(total=T)

    for step in range(P+1):
        systems[step] = system.clone(device)
        optimizers[step] = systems[step].configure_optimizers()['optimizer']
    
    
    last_vis_time = 0
    
    while begin_idx < T:
        
        parallel_len = end_idx - begin_idx

        pred_f = [None for _ in range(parallel_len)]
        
        for i in range(parallel_len):
            step = begin_idx + i
            params =[p.data for p in systems[step].get_params(include_occs=True, include_binaries=True)]
            queues[0].put( (params,  step) )
        
        for i in range(parallel_len):
            _preds, _step = queues[1].get()
            _i = _step - begin_idx
            pred_f[_i] = _preds
            

        
        rollout_system = systems[begin_idx]
        rollout_optimizer = optimizers[begin_idx]

        ind = None
        
        errors_all = 0
        for i in range(parallel_len):
            step = begin_idx + i

            if ind is None and rollout_system.renderer.cfg.get('grid_prune', False) and step > 0:
                np.random.seed(step+seed_offset)
                torch.manual_seed(step+seed_offset)
                torch.cuda.manual_seed(step+seed_offset)
                rollout_system.renderer.do_update_step(0, step)
            rollout_system.set_grads_from_grads(pred_f[i])
            rollout_optimizer.step()
            rollout_optimizer.zero_grad()
            if ind is None and rollout_system.renderer.cfg.get('grid_prune', False) and step > 0:
                rollout_system.renderer.do_update_step_end(0, step)

            # compute error
            error = rollout_system.compute_error_from_system(systems[step+1])
            
            if speedup_config.adaptivity_type == 'median':
                if i == parallel_len // 2:
                    errors_all = error
            elif speedup_config.adaptivity_type == 'mean':
                errors_all += error / parallel_len
            else:
                raise ValueError('Adaptivity not supported')
            
            if ind is None and (error > thresh or i == parallel_len - 1):
                ind = step+1
                optimizer_state_clone(optimizer_from=rollout_optimizer, optimizer_to=optimizers[step+1])
            if ind is not None:

                systems[step+1] = rollout_system.clone(device, systems[step+1])
                
        thresh = thresh * speedup_config.ema_decay + (errors_all ) * (1 - speedup_config.ema_decay)
            

        new_begin_idx = ind
        new_end_idx = min(new_begin_idx + parallel_len, T)

        for step in range(end_idx+1, new_end_idx+1):
            systems[step] = rollout_system.clone(device, systems[step - 1 - parallel_len])
            optimizers[step] = optimizers[step - 1 - parallel_len]

        progress = new_begin_idx - begin_idx
        begin_idx = new_begin_idx
        end_idx = new_end_idx

        total_iters += 1
        pbar.update(progress)


        elapsed = time.time() - start_time
        if elapsed >= last_vis_time + train_config.val_check_interval and train_config.visualize_progress:
            elapsed = time.time() - start_time
            systems[begin_idx].eval()
            with torch.no_grad():
                systems[begin_idx].set_save_dir(save_dir)   
                batch = next(iter(val_dataset))
                batch = to_device(batch, device=device)
                systems[begin_idx].validation_step(batch, begin_idx, [str(timedelta(seconds=elapsed)).split('.')[0]]  if train_config.display_time else None)
            last_vis_time = elapsed
            systems[begin_idx].train()
            
    pbar.close()
        
    elapsed = time.time() - start_time
    
    print('Effective Iters:', total_iters)
    system = systems[T]
    
    system.set_save_dir(save_dir)

    system.eval()
    global_step = T
    with torch.no_grad():
        for batch in tqdm(test_dataset):
            batch = to_device(batch, device)
            update_if_possible(test_dataset.dataset, 0, global_step)
            system.do_update_step(0, global_step)
            system.test_step(batch, global_step, [ str(timedelta(seconds=elapsed)).split('.')[0]] if train_config.display_time else None )
            system.do_update_step_end(0, global_step)
            update_end_if_possible( test_dataset.dataset, 0, global_step )
    system.on_test_epoch_end(global_step)
    save_dict = {'epoch': 0, 'global_step': T, 
                 'state_dict': system.state_dict()}
    torch.save(save_dict, os.path.join(save_dir, 'last.ckpt'))



if __name__ == "__main__":
    main()