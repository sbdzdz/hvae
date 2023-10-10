"""Training script."""
from pathlib import Path

import hydra
import torch
import torchvision
from omegaconf import DictConfig, OmegaConf
from torchinfo import summary
from torchvision import transforms

from hvae.callbacks import LoggingCallback, MetricsCallback, VisualizationCallback


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


@hydra.main(config_path="configs", config_name="main")
def train(cfg: DictConfig) -> None:
    """Train a model."""
    torch.set_float32_matmul_precision("high")
    train_dataloader, val_dataloader = get_dataloaders(cfg)
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=[LoggingCallback(), MetricsCallback(), VisualizationCallback()],
        logger={"config": config},
    )
    model = hydra.utils.instantiate(cfg.model)
    trainer.logger.watch(model, log_freq=cfg.trainer.log_every_n_steps)
    print_summary(cfg, model)
    trainer.fit(model, train_dataloader, val_dataloader)


def print_summary(cfg, model):
    """Print a summary of the model."""
    x = torch.zeros(
        cfg.training.batch_size,
        cfg.dataset.num_channels,
        cfg.dataset.img_size,
        cfg.dataset.img_size,
    ).to(model.device)

    y = torch.zeros(cfg.training.batch_size).to(model.device).long()

    summary(model, input_data=(x, y))


def get_dataloaders(cfg: DictConfig):
    root = Path(hydra.utils.get_original_cwd()) / Path(cfg.dataset.root)
    if cfg.dataset.name != "cifar10":
        raise ValueError(f"Invalid dataset name: {cfg.dataset.name}.")
    dataset = torchvision.datasets.CIFAR10(
        root=root,
        train=True,
        download=True,
    )
    # filter out everything except desired class
    if cfg.dataset.classes is not None:
        dataset = torch.utils.data.Subset(
            dataset,
            [i for i, (_, label) in enumerate(dataset) if label in cfg.dataset.classes],
        )
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [cfg.dataset.train_split, cfg.dataset.val_split]
    )
    train_dataset = CustomDataset(
        train_dataset,
        transform=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        ),
    )
    val_dataset = CustomDataset(
        val_dataset,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        ),
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
    )
    return train_dataloader, val_dataloader


if __name__ == "__main__":
    train()
