from dataclasses import dataclass


@dataclass
class Param:
    DEVICE: str = "cpu"
    DISTRIBUTED: bool = False

    # Data
    TRAIN_CORPUS_DIR: str = "./data/train"
    VAL_CORPUS_DIR: str = "./data/val"
    PERCENT_NOISE: float = 0.2
    MIN_NUM_TOKENS: int = 5

    # Training
    TOTAL_ITER: int = int(219661 * 20)  # Single machine
    # TOTAL_ITER: int = int(110342 * 40)  # 2 nodes
    NUM_WARMUP_STEP: int = 50000
    BATCH_SIZE: int = 8
    # EFFECTIVE_BZ: int = 16
    NUM_WORKER: int = 8
    # lr will increase from 2e-5 to MAX_LR in iter 0 -> iter NUM_ITER * PCT_START, then decrease to 2e-5
    MAX_LR: float = 1e-4
    MIN_LR: float = 1e-5
    POLY_LR_DECAY_POWER: float = 1.0
    WEIGHT_DECAY: float = 1e-2
    EXCLUDE_DECAY: bool = False

    # Change scheduler & optimizer
    IS_FINETUNE: bool = True

    # Logging & saving
    LOG_EVERY_N_STEPS: int = 1000
    DEBUG_PRED_EVERY_N_STEPS: int = 5000
    RUN_DIR: str = 'runs/'
    SAVE_N_STEP: int = 120000
