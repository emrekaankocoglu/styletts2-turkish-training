#!/usr/bin/env python3
"""
Turkish AuxiliaryASR Training Script (DDP)

- Leaves `train_turkish.py` untouched.
- Designed to be launched with torchrun, e.g.:
    torchrun --standalone --nproc_per_node 4 train_turkish_ddp.py -p ./Configs/config_turkish_ddp.yml
"""

import os
import os.path as osp
import sys
import yaml
import click
import numpy as np

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from meldataset_turkish import MelDataset, Collater
from models import build_model
from optimizers import build_optimizer
from trainer import Trainer
from utils import build_criterion, get_data_path_list, plot_image

import logging
from logging import StreamHandler


def _setup_logger(is_main: bool):
    logger = logging.getLogger("train_turkish_ddp")
    logger.setLevel(logging.DEBUG if is_main else logging.ERROR)
    if not logger.handlers:
        handler = StreamHandler()
        handler.setLevel(logging.DEBUG if is_main else logging.ERROR)
        logger.addHandler(handler)
    return logger


def _is_distributed():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _rank():
    return int(os.environ.get("RANK", "0"))


def _local_rank():
    return int(os.environ.get("LOCAL_RANK", "0"))


def _world_size():
    return int(os.environ.get("WORLD_SIZE", "1"))


def _init_distributed(device: str):
    if not _is_distributed():
        return
    backend = "nccl" if device.startswith("cuda") else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")


def _cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _state_dict_needs_module_prefix(model_state_keys, ckpt_state_keys):
    model_has = any(k.startswith("module.") for k in model_state_keys)
    ckpt_has = any(k.startswith("module.") for k in ckpt_state_keys)
    return model_has and (not ckpt_has)


def _state_dict_needs_strip_module_prefix(model_state_keys, ckpt_state_keys):
    model_has = any(k.startswith("module.") for k in model_state_keys)
    ckpt_has = any(k.startswith("module.") for k in ckpt_state_keys)
    return (not model_has) and ckpt_has


def _adapt_state_dict_for_model(model, state_dict):
    """Make checkpoint model keys compatible with current model state_dict keys."""
    model_keys = list(model.state_dict().keys())
    ckpt_keys = list(state_dict.keys())

    if _state_dict_needs_module_prefix(model_keys, ckpt_keys):
        return {f"module.{k}": v for k, v in state_dict.items()}
    if _state_dict_needs_strip_module_prefix(model_keys, ckpt_keys):
        return {k[len("module."):]: v for k, v in state_dict.items() if k.startswith("module.")}
    return state_dict


def load_checkpoint_flexible(trainer: Trainer, checkpoint_path: str, load_only_params: bool):
    state = torch.load(checkpoint_path, map_location="cpu")
    state["model"] = _adapt_state_dict_for_model(trainer.model, state["model"])
    trainer._load(state["model"], trainer.model)
    if not load_only_params:
        trainer.steps = state.get("steps", 0)
        trainer.epochs = state.get("epochs", 0)
        if "optimizer" in state:
            trainer.optimizer.load_state_dict(state["optimizer"])
        if "scheduler" in state:
            state["scheduler"].update(**trainer.config.get("scheduler_params", {}))
            trainer.scheduler.load_state_dict(state["scheduler"])


def save_checkpoint_portable(trainer: Trainer, checkpoint_path: str):
    """Save checkpoint with model *without* DDP 'module.' prefix."""
    model_obj = trainer.model
    model_state = model_obj.module.state_dict() if hasattr(model_obj, "module") else model_obj.state_dict()
    state = {
        "model": model_state,
        "optimizer": trainer.optimizer.state_dict(),
        "scheduler": trainer.scheduler.state_dict(),
        "steps": trainer.steps,
        "epochs": trainer.epochs,
    }
    os.makedirs(osp.dirname(checkpoint_path), exist_ok=True)
    torch.save(state, checkpoint_path)


@click.command()
@click.option("-p", "--config_path", default="./Configs/config_turkish_ddp.yml", type=str)
def main(config_path: str):
    config = yaml.safe_load(open(config_path))

    log_dir = config["log_dir"]
    epochs = int(config.get("epochs", 200))
    batch_size = int(config.get("batch_size", 32))  # per-GPU batch size
    save_freq = int(config.get("save_freq", 10))
    device = config.get("device", "cuda")

    is_main = (_rank() == 0)
    logger = _setup_logger(is_main)

    # DDP init
    _init_distributed(device=device)
    if device.startswith("cuda"):
        torch.cuda.set_device(_local_rank())
        device = f"cuda:{_local_rank()}"

    if is_main:
        logger.info("Training Turkish AuxiliaryASR (DDP)")
        logger.info(f"Config: {config_path}")
        logger.info(f"World size: {_world_size()}")
        logger.info(f"Batch size (per GPU): {batch_size}")
        logger.info(f"Epochs: {epochs}")
        logger.info(f"Device: {device}")

    # Data
    train_list, val_list = get_data_path_list(config.get("train_data"), config.get("val_data"))

    train_dataset = MelDataset(train_list)
    val_dataset = MelDataset(val_list)
    collate_fn = Collater()

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=_world_size(), rank=_rank(), shuffle=True, drop_last=True
    ) if _is_distributed() else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=int(config.get("num_workers", 2)),
        drop_last=True,
        collate_fn=collate_fn,
        pin_memory=(not device.startswith("cpu")),
    )

    # Eval only on rank 0 to keep logs simple
    val_loader = None
    if is_main:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(config.get("num_workers", 2)),
            drop_last=False,
            collate_fn=collate_fn,
            pin_memory=(not device.startswith("cpu")),
        )

    # Model/optim
    model = build_model(config["model_params"]).to(device)
    if _is_distributed() and device.startswith("cuda"):
        model = DDP(model, device_ids=[_local_rank()], output_device=_local_rank(), broadcast_buffers=False)
        # Trainer code expects these attributes on `self.model`, but DDP doesn't forward
        # arbitrary attributes to `.module`. Expose the ones Trainer uses.
        model.n_down = model.module.n_down
        model.get_future_mask = model.module.get_future_mask
        model.length_to_mask = model.module.length_to_mask

    blank_index = train_dataset.text_cleaner.word_index_dictionary.get("<blank>", 115)
    if is_main:
        logger.info(f"Blank index for CTC: {blank_index}")

    criterion = build_criterion(critic_params={"ctc": {"blank": blank_index}})

    scheduler_params = {
        "max_lr": float(config["optimizer_params"].get("lr", 5e-4)),
        "pct_start": float(config["optimizer_params"].get("pct_start", 0.0)),
        "epochs": epochs,
        "steps_per_epoch": len(train_loader),
    }
    optimizer, scheduler = build_optimizer(
        {"params": model.parameters(), "optimizer_params": {}, "scheduler_params": scheduler_params}
    )

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        logger=logger,
        config={"scheduler_params": scheduler_params},
    )

    writer = SummaryWriter(log_dir=osp.join(log_dir, "tensorboard")) if is_main else None

    pretrained = config.get("pretrained_model", "")
    load_only_params = bool(config.get("load_only_params", True))
    if pretrained:
        if is_main:
            logger.info(f"Loading checkpoint: {pretrained} (load_only_params={load_only_params})")
        load_checkpoint_flexible(trainer, pretrained, load_only_params=load_only_params)

    if is_main:
        logger.info("Starting training...")

    for epoch in range(1, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_results = trainer._train_epoch()

        # synchronize before eval/logging
        if _is_distributed():
            dist.barrier()

        if is_main:
            eval_results = trainer._eval_epoch()
            results = train_results.copy()
            results.update(eval_results)

            logger.info("--- epoch %d ---" % epoch)
            for key, value in results.items():
                if isinstance(value, float):
                    logger.info("%-15s: %.4f" % (key, value))
                    if writer is not None:
                        writer.add_scalar(key, value, epoch)
                else:
                    # attention images
                    for v in value:
                        if writer is not None:
                            writer.add_figure("eval_attn", plot_image(v), epoch)

            if (epoch % save_freq) == 0:
                ckpt_path = osp.join(log_dir, "epoch_%05d.pth" % epoch)
                save_checkpoint_portable(trainer, ckpt_path)

        if _is_distributed():
            dist.barrier()

    if writer is not None:
        writer.close()
    _cleanup_distributed()
    return 0


if __name__ == "__main__":
    main()

