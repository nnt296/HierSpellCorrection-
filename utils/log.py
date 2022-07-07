import logging
# from pytorch_lightning.loggers import WandbLogger


# # configure logging at the root level of Lightning
# logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
# configure logging on module level, redirect to file
logger = logging.getLogger("pytorch_lightning.debug")
logger.setLevel(logging.INFO)
logger.addHandler(logging.FileHandler("debug_prediction.log"))


# wandb_logger = WandbLogger(project="spell-checker")
