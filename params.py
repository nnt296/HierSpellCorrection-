from dataclasses import dataclass


@dataclass
class Param:
    # DEVICE: str = "cpu"
    DISTRIBUTED: bool = False

    # Data
    TRAIN_CORPUS_DIR: str = "./data/train"
    VAL_CORPUS_DIR: str = "./data/val"
    PERCENT_NOISE: float = 0.15
    MIN_NUM_TOKENS: int = 5
    BATCH_SIZE: int = 16
    NUM_WORKER: int = 8
    FP16: bool = True

    # Training

    # Batch accumulation might not work on BatchNorm layer,
    # but Albert uses LayerNorm, which does not depend on batch (???)
    # Batch accumulation affects global_step when training (so remember to divide steps by BATCH_ACCUM)
    BATCH_ACCUM: int = 512 // BATCH_SIZE  # Set to 1 to disable
    TOTAL_STEP: int = 200000  # Single machine
    # TOTAL_ITER: int = int(110342 * 40)  # 2 nodes
    # Change scheduler & optimizer
    IS_FINETUNE: bool = False
    CKPT_PATH: str = None

    # Optimizer
    MAX_LR: float = 1.76e-3
    MIN_LR: float = 1e-6
    POLY_LR_DECAY_POWER: float = 1.0
    WEIGHT_DECAY: float = 1e-2
    EXCLUDE_DECAY: bool = True
    OPTIM: str = "lamb"
    NUM_WARMUP_STEP: int = 10000

    # Logging & saving
    LOG_EVERY_N_STEPS: int = 100
    DEBUG_PRED_EVERY_N_ITER: int = 5000  # Gradient accumulation does not affect this
    RUN_DIR: str = 'runs/'
    SAVE_N_STEP: int = 3000
