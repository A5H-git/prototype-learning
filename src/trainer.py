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
    max_epochs: int,  # = 200,
    max_steps: int,  # = 25_000,  # Adjust if needed
    monitor: str = "val_acc",
    mode: str = "max",
    grad_batches: int = 2,
):
    logger = CSVLogger(save_dir="logs", name=name)
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
        accumulate_grad_batches=grad_batches,
        max_steps=max_steps,
        max_epochs=max_epochs if max_steps == -1 else -1,
    )

    return trainer, checkpoint_cb, logger
