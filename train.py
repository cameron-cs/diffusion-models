from typing import Optional, Union
import torch
from tqdm import tqdm
from torchvision.utils import make_grid
from PIL import Image
from pathlib2 import Path

from dataset.cifar_dataset import create_cifar10_dataset
from model.callback.checkpoint import ModelCheckpoint
from model.diffusion_trainer import GaussianDiffusionTrainer
from model.unet import ConditionalUNet


def train_one_epoch(trainer, loader, optimizer, device, epoch):
    trainer.train()
    total_loss, total_num = 0., 0

    with tqdm(loader, dynamic_ncols=True, colour="#ff924a") as data:
        for images, _ in data:
            optimizer.zero_grad()

            x_0 = images.to(device)
            loss = trainer(x_0)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_num += x_0.shape[0]

            data.set_description(f"Epoch: {epoch}")
            data.set_postfix(ordered_dict={
                "train_loss": total_loss / total_num,
            })

    return total_loss / total_num


def train(config):
    consume = config["consume"]
    if consume:
        cp = torch.load(config["consume_path"])
        config = cp["config"]
    print(config)

    device = torch.device(config["device"])
    loader = create_cifar10_dataset(**config["Dataset"])
    start_epoch = 1

    model = ConditionalUNet(**config["Model"]).to(device)
    model = torch.nn.DataParallel(model)  # optional: use DataParallel if multiple GPUs are available
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    trainer = GaussianDiffusionTrainer(model, **config["Trainer"]).to(device)

    model_callback = ModelCheckpoint(**config["Callback"])

    if consume:
        model_callback.load_state_dict(cp["model_callback"])
        start_epoch = cp["start_epoch"] + 1

    for epoch in range(start_epoch, config["epochs"] + 1):
        total_loss = 0.0
        total_num = 0
        with tqdm(loader, dynamic_ncols=True, colour="#ff924a") as data:
            for images, labels in data:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = trainer(images, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_num += images.size(0)

                data.set_description(f"Epoch [{epoch}]")
                data.set_postfix(train_loss=total_loss / total_num)

        model_callback.step(total_loss / total_num, model=model.state_dict(), config=config,
                              optimizer=optimizer.state_dict(), start_epoch=epoch,
                              model_checkpoint=model_callback.state_dict())


if __name__ == '__main__':
    dataloader = create_cifar10_dataset(data_path='data/CIFAR-10/', batch_size=64)

    train_conf = {
        "Model": {
            "in_channels": 3,
            "out_channels": 3,
            "model_channels": 128,
            "attention_resolutions": [
                2
            ],
            "num_res_blocks": 2,
            "dropout": 0.1,
            "channel_mult": [
                1,
                2,
                2,
                2
            ],
            "conv_resample": True,
            "num_heads": 4
        },
        "Dataset": {
            "train": True,
            "data_path": "data",
            "download": True,
            "image_size": [
                32,
                32
            ],
            "mode": "RGB",
            "suffix": [
                "png",
                "jpg"
            ],
            "batch_size": 64,
            "shuffle": True,
            "drop_last": True,
            "pin_memory": True,
            "num_workers": 4
        },
        "Trainer": {
            "T": 1000,
            "beta": [
                0.0001,
                0.02
            ]
        },
        "Callback": {
            "filepath": "callback/ddimp_cifar10.pth",
            "save_freq": 1
        },
        "device": "cuda:0",
        "epochs": 750,
        "consume": False,
        "consume_path": "callback/ddimp_cifar10.pth",
        "lr": 0.0002
    }

    train(train_conf)