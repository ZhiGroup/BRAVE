"""Train the voxel encoder that produces Brain Region Embeddings.

The objective is a joint InfoNCE loss with three equally weighted terms — one at
the fine-scale voxel level, one at the coarse-scale voxel level, and one at the
instance level (see `models.Model.training_step`). Optimisation is AdamW under a
one-cycle learning-rate schedule stepped once per mini-batch.

Two details matter for reproducing the published encoder:

* The schedule is built for `--scheduler-epochs` (40 in the paper) while
  training stops after `--train-epochs` (8). Passing 8 to both would complete a
  whole one-cycle in 8 epochs and give a completely different learning-rate
  trajectory.
* Batch size 4. Several derived sampling quantities depend on it, so changing it
  changes the negative-sample counts.

Paths come from `config/paths.yaml` (`embedding:` section) via `config.py`.

Usage
-----
    python train_encoder.py --seed 1                  # published settings
    python train_encoder.py --seed 1 --devices 0      # single GPU
"""
import argparse
import json
import os
import sys

# CUDA_VISIBLE_DEVICES and the cuBLAS workspace must be set before torch
# initialises CUDA, so the device list is parsed ahead of the real argparse.


def _pre_parse():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--devices", default="0,1")
    p.add_argument("--seed", type=int, default=0)
    known, _ = p.parse_known_args()
    return known


_PRE = _pre_parse()
os.environ["CUDA_VISIBLE_DEVICES"] = _PRE.devices
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["PYTHONHASHSEED"] = str(_PRE.seed)

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402
import experiment_configurations  # noqa: E402
import models  # noqa: E402
import preprocessing  # noqa: E402
import seeding  # noqa: E402

from lightning.pytorch import Trainer, seed_everything  # noqa: E402
from lightning.pytorch.callbacks import (Callback,  # noqa: E402
                                         LearningRateMonitor, ModelCheckpoint)
from lightning.pytorch.loggers import TensorBoardLogger  # noqa: E402


class StopAfterEpochs(Callback):
    """Stop training after N epochs without shrinking the LR schedule."""

    def __init__(self, n_epochs):
        super().__init__()
        self.n_epochs = int(n_epochs)

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch + 1 >= self.n_epochs:
            print("[stop] reached {0} epochs; stopping".format(self.n_epochs))
            trainer.should_stop = True


def build_args(cli):
    """Assemble the hyper-parameter dict the model and dataset expect.

    Mirrors the internal driver's argument assembly exactly; only the path
    sources differ (config/paths.yaml instead of hardcoded locations).
    """
    conf = experiment_configurations.conf_model_4_exp_3()
    conf.n_batch = cli.batch_size
    conf.max_epoch = cli.scheduler_epochs
    # label only; used for output naming, never reaches the model
    conf.sub_folder = "voxel_infonce"
    if cli.n_samples is not None:
        conf.n_samples = cli.n_samples

    # training images come from the path config
    conf.train_img_dir = config.train_dir()
    conf.val_img_dir = config.val_dir()

    check_point_path = os.path.join(config.model_weight_path(),
                                    conf.tensorboard_folder_name)
    os.makedirs(check_point_path, exist_ok=True)
    check_point_path = os.path.join(check_point_path, conf.last_chk_point_name)

    experiment_configurations.print_conf(conf)

    args = {'train_img_dir': conf.train_img_dir,
            'val_img_dir': conf.val_img_dir,
            'batch_size': conf.n_batch,
            'n_pos_voxel': conf.n_pos_voxel,
            'patch_size': conf.patch_size,
            'embedding_dim': conf.embedding_dim,
            'los_temp': conf.los_temp,
            'n_samples': conf.n_samples,
            'prob_overlapped_pairs': conf.prob_overlapped_pairs,
            'fg_pct': conf.fg_pct,
            'radius': conf.radius,
            'lr': conf.lr,
            'weight_decay': conf.weight_decay,
            'device': torch.device('cpu'),
            'local_loss_fract': conf.local_loss_fract,
            'epochs': conf.epochs,
            'weight_name': conf.weight_name,
            'min_loss_red': conf.min_loss_red,
            'apply_z_stride': conf.apply_z_stride,
            'images': [],
            'patience': conf.patience,
            'do_check_point': conf.do_check_point,
            'check_point_path': check_point_path,
            'use_scheduler': conf.use_scheduler,
            'sche_lamda': conf.sche_lamda,
            'perform_val_iter_cycle': conf.perform_val_iter_cycle,
            'tensorboard_folder_name': conf.tensorboard_folder_name,
            'do_epoch_wise_evaluation': conf.do_epoch_wise_evaluation,
            'factor_n_negative_samples': conf.factor_n_negative_samples,
            'model_name': conf.model_name,
            'sub_folder': conf.sub_folder,
            'use_gradient_clipping': conf.use_gradient_clipping,
            'gradient_clipping_value': conf.gradient_clipping_value,
            'pct_start': conf.pct_start,
            'div_factor': conf.div_factor,
            'final_div_factor': conf.final_div_factor,
            'num_instances': conf.num_instances,
            'max_epoch': conf.max_epoch,
            'los_temp_instance': conf.los_temp_instance,
            'memory_bank_size': conf.memory_bank_size,
            'momentum': conf.momentum,
            'reconstruction_loss': conf.reconstruction_loss,
            'seed': cli.seed,
            }
    return args, conf


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--devices", default="0,1",
                    help="CUDA_VISIBLE_DEVICES list, e.g. '0,1' or '0'")
    ap.add_argument("--train-epochs", type=int, default=8,
                    help="epochs actually trained (published encoder: 8)")
    ap.add_argument("--scheduler-epochs", type=int, default=40,
                    help="epochs the one-cycle schedule spans (published: 40)")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--n-samples", type=int, default=None,
                    help="cap the number of training images (smoke tests)")
    ap.add_argument("--limit-train-batches", type=int, default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--detect-anomaly", action="store_true", default=True)
    ap.add_argument("--no-detect-anomaly", dest="detect_anomaly",
                    action="store_false")
    cli = ap.parse_args()

    seed_everything(cli.seed, workers=True)
    seeding.set_global_seed(cli.seed)

    args, conf = build_args(cli)

    run_name = cli.run_name or "seed_{0}".format(cli.seed)
    ckpt_dir = os.path.join(config.model_weight_path(), run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    print("[run] checkpoints -> {0}".format(ckpt_dir))

    model = models.get_model(args, model_type='vanila')
    datamodule = preprocessing.VoxelEmbedDataModule(args)

    callbacks = [
        ModelCheckpoint(dirpath=ckpt_dir,
                        filename="{0}_{{epoch:03d}}".format(run_name),
                        save_top_k=-1, save_last=True),
        LearningRateMonitor(logging_interval="epoch"),
        StopAfterEpochs(cli.train_epochs),
    ]

    n_devices = len([d for d in cli.devices.split(",") if d.strip()])
    torch.autograd.set_detect_anomaly(bool(cli.detect_anomaly))

    trainer_kwargs = dict(
        logger=[TensorBoardLogger(save_dir=config.log_path(), name=run_name)],
        callbacks=callbacks,
        accelerator="cuda",
        devices=n_devices,
        sync_batchnorm=True,
        log_every_n_steps=20,
        benchmark=False,        # the cuDNN autotuner is nondeterministic
        deterministic=False,    # see seeding.set_global_seed
        max_epochs=cli.scheduler_epochs,
        gradient_clip_val=1.5,
        gradient_clip_algorithm="value",
    )
    if n_devices > 1:
        trainer_kwargs["strategy"] = "ddp"
    if cli.limit_train_batches is not None:
        trainer_kwargs["limit_train_batches"] = cli.limit_train_batches
        trainer_kwargs["limit_val_batches"] = 0

    trainer = Trainer(**trainer_kwargs)

    with open(os.path.join(ckpt_dir, "run_manifest.json"), "w") as fh:
        json.dump({"seed": cli.seed,
                   "train_epochs": cli.train_epochs,
                   "scheduler_epochs": cli.scheduler_epochs,
                   "batch_size": cli.batch_size,
                   "devices": cli.devices,
                   "environment": seeding.environment_report()},
                  fh, indent=2, sort_keys=True)

    trainer.fit(model=model, datamodule=datamodule)
    print("[done] checkpoints in {0}".format(ckpt_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
