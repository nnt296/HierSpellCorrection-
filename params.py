from dataclasses import dataclass

import torch


@dataclass
class Param:
    DEVICE = "cpu"

    # Data
    TRAIN_CORPUS_DIR = "/home/local/BM/Datasets/SpellNews/train"
    VAL_CORPUS_DIR = "/home/local/BM/Datasets/SpellNews/val"
    PERCENT_NOISE = 0.3

    # Training
    NUM_ITER = 500
    NUM_WARMUP_STEP = 100
    BATCH_SIZE = 4
    LOG_PER_ITER = 1
    NUM_WORKER = 2
    # lr will increase from 2e-5 to MAX_LR in iter 0 -> iter NUM_ITER * PCT_START, then decrease to 2e-5
    MAX_LR = 0.003
    WEIGHT_DECAY = 1e-2

    RUN_DIR = 'runs/'
