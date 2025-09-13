import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from pytorch_lightning.loggers import CSVLogger


CHECKPOINT_PATH = "checkpoints"


def setup_trainer(
    name,
    max_steps: int = 25_000,  # Adjust if needed
    max_epochs: int = 200,
    monitor: str = "val_acc",
    mode="max",
):
    logger = CSVLogger(save_dir="logs", name=name)
    filename = f""
    checkpoint_cb = ModelCheckpoint(
        dirpath=CHECKPOINT_PATH,
        monitor=monitor,
        filename=r"{epoch}-{" + f"{monitor}" + r":.2f}",
        save_top_k=2,
        mode=mode,
        save_last=True,
    )

    lr_monitor_cb = LearningRateMonitor("epoch")
    earlystop_cb = EarlyStopping(
        monitor=monitor, mode=mode, patience=25, min_delta=1e-4
    )

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_cb, lr_monitor_cb],  # , earlystop_cb],
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        log_every_n_steps=10,
        max_steps=max_steps,
        max_epochs=max_epochs if not max_steps else 0,
    )

    return trainer, checkpoint_cb, logger
