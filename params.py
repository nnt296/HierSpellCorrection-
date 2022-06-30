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
    TOTAL_ITER: int = int(219661 * 30)  # Single machine
    # TOTAL_ITER: int = int(110342 * 40)  # 2 nodes
    NUM_WARMUP_STEP: int = 50176
    BATCH_SIZE: int = 8
    # Batch accumulation might not work on BatchNorm layer,
    # but Albert uses LayerNorm, which does not depend on batch (???)
    BATCH_ACCUM: int = 32  # = None to disable
    NUM_WORKER: int = 8

    # Optimizer
    MAX_LR: float = 8.8e-4
    MIN_LR: float = 1e-6
    POLY_LR_DECAY_POWER: float = 1.0
    WEIGHT_DECAY: float = 1e-2
    EXCLUDE_DECAY: bool = False
    OPTIM: str = "lamb"

    # Change scheduler & optimizer
    IS_FINETUNE: bool = False

    # Logging & saving
    LOG_EVERY_N_STEPS: int = 1024
    DEBUG_PRED_EVERY_N_STEPS: int = 5120
    RUN_DIR: str = 'runs/'
    SAVE_N_STEP: int = 115200
