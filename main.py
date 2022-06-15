from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_constant_schedule_with_warmup
import pytorch_lightning as pl

from models.baseline import AlbertConfig, AlbertSpellChecker, SpellCheckerOutput
from models.metrics import compute_detection_metrics, compute_correction_metrics
from data.dataset import MisspelledDataset, custom_collator

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

    def train_dataloader(self):
        pass

    def test_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def predict_dataloader(self):
        pass

    def forward(self, inputs) -> SpellCheckerOutput:
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs["loss"]

        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, batch_size=self.params.BATCH_SIZE)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)

        loss = outputs["loss"]
        self.log("val_loss", loss, on_epoch=True, batch_size=self.params.BATCH_SIZE)

        detection_metrics, batch_sz = compute_detection_metrics(detection_logits=outputs["detection_logits"],
                                                                detection_labels=batch["detection_labels"])
        correction_metrics, _ = compute_correction_metrics(correction_logits=outputs["correction_logits"],
                                                           detection_labels=batch["detection_labels"],
                                                           correction_labels=batch["correction_labels"])

        self.log("det_f1", detection_metrics["f1"], on_epoch=True)
        self.log("det_precision", detection_metrics["precision"], on_epoch=True)
        self.log("det_recall", detection_metrics["recall"], on_epoch=True)

        self.log("corr_f1", correction_metrics["f1"], on_epoch=True)
        self.log("corr_precision", correction_metrics["precision"], on_epoch=True)
        self.log("corr_recall", correction_metrics["recall"], on_epoch=True)

        return outputs


def main():
    # Define training config
    params = Param()

    # Define dataset
    train_ds = MisspelledDataset(corpus_dir=params.TRAIN_CORPUS_DIR)
    train_loader = DataLoader(train_ds,
                              batch_size=params.BATCH_SIZE,
                              collate_fn=custom_collator,
                              num_workers=2,
                              drop_last=True)

    val_ds = MisspelledDataset(corpus_dir=params.VAL_CORPUS_DIR)
    val_loader = DataLoader(val_ds,
                            batch_size=params.BATCH_SIZE,
                            collate_fn=custom_collator,
                            num_workers=2,
                            drop_last=False)

    char_cfg = AlbertConfig().from_json_file("spell_model/char_model/config.json")
    word_cfg = AlbertConfig().from_json_file("spell_model/word_model/config.json")

    checker = SpellChecker(char_cfg, word_cfg, params)
    trainer = pl.Trainer(
        default_root_dir="runs/",
        max_steps=params.NUM_ITER
    )
    trainer.fit(checker, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == '__main__':
    main()
