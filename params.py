from dataclasses import dataclass


@dataclass
class Param:
    DEVICE = "cpu"

    # Data
    TRAIN_CORPUS_DIR = "./data/train"
    VAL_CORPUS_DIR = "./data/val"
    PERCENT_NOISE = 0.2
    MIN_NUM_TOKENS = 5

    # Training
    NUM_ITER = int(219661 * 5)
    NUM_WARMUP_STEP = 6000
    BATCH_SIZE = 8
    NUM_WORKER = 8
    # lr will increase from 2e-5 to MAX_LR in iter 0 -> iter NUM_ITER * PCT_START, then decrease to 2e-5
    MAX_LR = 1e-4
    WEIGHT_DECAY = 1e-2

    # Logging & saving
    LOG_EVERY_N_STEPS = 1000
    DEBUG_PRED_EVERY_N_STEPS = 5000
    RUN_DIR = 'runs/'
    SAVE_N_STEP = 120000
