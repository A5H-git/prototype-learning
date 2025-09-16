from models.simclr import SimCLR, get_contrastive_transforms, ContrastiveTransform
from models.protonet import ProtoNet, FewShotBatchSampler, get_protonet_transforms
from models.dino import get_dino_transforms
from torch import nn
import torch
from trainer import setup_trainer
from dataloaders import MNISTData, TMEData
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision.models import DenseNet

DATAPATH = "data/pathmnist_224.npz"
TEST_DATAPATH = "data/HMU-GE-HE-30K"
NUM_WORKERS = 0  # not happy with more than 0, probably due to memory issues
BATCH_SIZE = 256 // 2
CHECKPOINT = "checkpoints/epoch=20-val_acc_top1=0.78.ckpt"

SIMCLR_CKPT = "checkpoints/SimCLR/epoch=35-val_acc_top1=0.86.ckpt"
SIMCLR_HPARAMS = "logs/convnet/version_4/hparams.yaml"
N_WAY = 5
K_SHOT = 4

# I forgot to seed when training oops...


def main():
    # base_transform = get_contrastive_transforms()
    base_transform = get_dino_transforms()
    proto_transform = get_protonet_transforms()
    # contrastive_transform = ContrastiveTransform(base_transform, 2)

    train_dataset = MNISTData(DATAPATH, "train", proto_transform)  # type: ignore
    val_dataset = MNISTData(DATAPATH, "val", base_transform)  # type: ignore

    test_batch_sampler = FewShotBatchSampler(
        dataset_targets=torch.from_numpy(train_dataset.labels),
        N_way=N_WAY,
        K_shot=K_SHOT,
        shuffle=True,
        include_query=True,
        shuffle_once=False,
    )

    val_batch_sampler = FewShotBatchSampler(
        dataset_targets=torch.from_numpy(val_dataset.labels),
        N_way=N_WAY,
        K_shot=K_SHOT,
        shuffle=False,
        include_query=True,
        shuffle_once=True,
    )

    train_dataloader = DataLoader(
        train_dataset,
        # batch_size=BATCH_SIZE, # Comment when using batch_sampler
        # shuffle=True,
        pin_memory=True,
        num_workers=NUM_WORKERS,
        # persistent_workers=True,
        batch_sampler=test_batch_sampler,
    )

    val_dataloader = DataLoader(
        val_dataset,
        # batch_size=BATCH_SIZE, # Comment when using batch_sampler
        # shuffle=False,
        pin_memory=True,
        num_workers=NUM_WORKERS,
        # persistent_workers=True,
        batch_sampler=val_batch_sampler,
    )

    trainer, checkpoint, _ = setup_trainer(
        "SimCLRNet",
        max_epochs=200,  # used 500 for simClr
        max_steps=-1,
        monitor="val_acc",
        grad_batches=2,
    )

    # Going to use the same as UvA Notebook
    # convnet = DenseNet(
    #     growth_rate=32,
    #     block_config=(6, 6, 6, 6),
    #     bn_size=2,
    #     num_init_features=64,
    #     num_classes=128,
    # )
    simclr = SimCLR.load_from_checkpoint(
        checkpoint_path=SIMCLR_CKPT, hparams_file=SIMCLR_HPARAMS
    )

    simclr_backbone = nn.Sequential(
        simclr.encoder,
        nn.Flatten(),
    )

    for p in simclr_backbone.parameters():
        p.requires_grad = False

    model = ProtoNet(backbone=simclr_backbone, learning_rate=5e-4)

    # model = SimCLR(
    #     hidden_dim=2048,
    #     out_dim=128,
    #     learning_rate=5e-4,
    #     temperature=0.07,
    #     weight_decay=1e-4,
    #     max_epochs=500,
    # )

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


if __name__ == "__main__":
    # import torch.multiprocessing as mp

    # mp.set_start_method("spawn", force=True)  # Windows default, but make it explicit
    main()
