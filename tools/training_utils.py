import torch
import numpy as np
import absl.flags as flags

FLAGS = flags.FLAGS
from mmengine import Config
from tools.solver_utils import build_lr_scheduler, build_optimizer_with_params


def build_lr_rate(optimizer, total_iters):
    # build cfg from flags
    cfg = dict(
        SOLVER=dict(
            IMS_PER_BATCH=FLAGS.batch_size,
            TOTAL_EPOCHS=FLAGS.total_epoch,
            LR_SCHEDULER_NAME=FLAGS.lr_scheduler_name,
            REL_STEPS=(0.5, 0.75),
            ANNEAL_METHOD=FLAGS.anneal_method,  # "cosine"
            ANNEAL_POINT=FLAGS.anneal_point,
            # REL_STEPS=(0.3125, 0.625, 0.9375),
            OPTIMIZER_CFG=dict(type=FLAGS.optimizer_type, lr=FLAGS.lr, weight_decay=0),
            WEIGHT_DECAY=FLAGS.weight_decay,
            WARMUP_FACTOR=FLAGS.warmup_factor,
            WARMUP_ITERS=FLAGS.warmup_iters,
            WARMUP_METHOD=FLAGS.warmup_method,
            GAMMA=FLAGS.gamma,
            POLY_POWER=FLAGS.poly_power,
        ),
    )
    cfg = Config(cfg)
    scheduler = build_lr_scheduler(cfg, optimizer, total_iters=total_iters)
    return scheduler

