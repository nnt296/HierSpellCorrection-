from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_constant_schedule_with_warmup
import pytorch_lightning as pl

from models.baseline import AlbertConfig, AlbertSpellChecker, SpellCheckerOutput
from models.metrics import compute_detection_metrics, compute_correction_metrics
from data.dataset import MisspelledDataset, custom_collator, word_tokenizer, char_tokenizer
from utils.debug_prediction import debug_prediction

from params import Param


class SpellChecker(pl.LightningModule):

    def __init__(self,
                 char_config: AlbertConfig,
                 word_config: AlbertConfig,
                 params: Param):
        super().__init__()
        self.params = params

        self.char_cfg = char_config
        self.word_cfg = word_config
        self.model = AlbertSpellChecker(self.char_cfg, self.word_cfg)

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.params.MAX_LR,
                          betas=(0.9, 0.998), eps=1e-8, weight_decay=self.params.WEIGHT_DECAY)
        scheduler = get_constant_schedule_with_warmup(optimizer=optimizer,
                                                      num_warmup_steps=self.params.NUM_WARMUP_STEP,
                                                      last_epoch=-1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }

    def forward(self, inputs) -> SpellCheckerOutput:
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs["loss"]
        detection_loss = outputs["detection_loss"]
        correction_loss = outputs["correction_loss"]

        self.log("train_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, batch_size=self.params.BATCH_SIZE)
        self.log("det_loss", detection_loss, on_step=True, on_epoch=False,
                 prog_bar=True, logger=True, batch_size=self.params.BATCH_SIZE)
        self.log("corr_loss", correction_loss, on_step=True, on_epoch=False,
                 prog_bar=True, logger=True, batch_size=self.params.BATCH_SIZE)

        if (batch_idx + 1) % self.params.DEBUG_PRED_EVERY_N_STEPS == 0:
            debug_prediction(
                detection_logits=outputs["detection_logits"],
                correction_logits=outputs["correction_logits"],
                correction_labels=batch["correction_labels"],
                char_input_ids=batch["char_input_ids"],
                word_input_ids=batch["word_input_ids"],
                detection_labels=batch["detection_labels"],
                word_tokenizer=word_tokenizer,
                char_tokenizer=char_tokenizer
            )
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)

        loss = outputs["loss"]
        self.log("val_loss", loss, on_epoch=True, batch_size=self.params.BATCH_SIZE)

        self.compute_metrics(outputs=outputs,
                             detection_labels=batch["detection_labels"],
                             correction_labels=batch["correction_labels"])
        return outputs

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        # # Batch from wiki_spelling_collator()
        # _ = batch.pop("correction_labels")
        # detection_labels = batch.pop("detection_labels")
        #
        # outputs = self(batch)
        # # Infer detection metrics for now, since fixing seems complicated
        # self.compute_metrics(outputs=outputs,
        #                      detection_labels=detection_labels,
        #                      correction_labels=None)
        pass

    def compute_metrics(self,
                        outputs: SpellCheckerOutput,
                        detection_labels: torch.LongTensor = None,
                        correction_labels: torch.LongTensor = None):
        """
        Compute and log metrics based on model's outputs and labels
        """
        bz = self.params.BATCH_SIZE
        if detection_labels is not None:
            detection_metrics, batch_sz = compute_detection_metrics(detection_logits=outputs["detection_logits"],
                                                                    detection_labels=detection_labels)
            self.log("det_f1", detection_metrics["f1"], on_epoch=True, batch_size=bz)
            self.log("det_precision", detection_metrics["precision"], on_epoch=True, batch_size=bz)
            self.log("det_recall", detection_metrics["recall"], on_epoch=True, batch_size=bz)

        if correction_labels is not None and detection_labels is not None:
            correction_metrics, _ = compute_correction_metrics(correction_logits=outputs["correction_logits"],
                                                               detection_labels=detection_labels,
                                                               correction_labels=correction_labels)
            self.log("corr_f1", correction_metrics["f1"], on_epoch=True, batch_size=bz)
            self.log("corr_precision", correction_metrics["precision"], on_epoch=True, batch_size=bz)
            self.log("corr_recall", correction_metrics["recall"], on_epoch=True, batch_size=bz)


def main():
    # Define training config
    params = Param()

    # Define dataset
    train_ds = MisspelledDataset(corpus_dir=params.TRAIN_CORPUS_DIR,
                                 percent_err=params.PERCENT_NOISE,
                                 min_num_tokens=params.MIN_NUM_TOKENS)
    train_loader = DataLoader(train_ds,
                              batch_size=params.BATCH_SIZE,
                              collate_fn=custom_collator,
                              num_workers=params.NUM_WORKER,
                              drop_last=True)

    # This validation dataset is inconsistent between epoch,
    # since we randomly select a sentence, add noise and chunk
    val_ds = MisspelledDataset(corpus_dir=params.VAL_CORPUS_DIR,
                               percent_err=params.PERCENT_NOISE,
                               min_num_tokens=params.MIN_NUM_TOKENS)
    val_loader = DataLoader(val_ds,
                            batch_size=params.BATCH_SIZE,
                            collate_fn=custom_collator,
                            num_workers=params.NUM_WORKER,
                            drop_last=False)

    char_cfg = AlbertConfig().from_json_file("spell_model/char_model/config.json")
    word_cfg = AlbertConfig().from_json_file("spell_model/word_model/config.json")

    ckpt_callback = pl.callbacks.ModelCheckpoint(save_last=True,
                                                 every_n_train_steps=params.SAVE_N_STEP)

    checker = SpellChecker(char_cfg, word_cfg, params)
    trainer = pl.Trainer(
        default_root_dir=params.RUN_DIR,
        max_steps=params.NUM_ITER,  # Training steps only
        accelerator="gpu",
        devices=1,
        log_every_n_steps=params.LOG_EVERY_N_STEPS,
        callbacks=[ckpt_callback]
    )
    trainer.fit(checker, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == '__main__':
    main()
