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

def save_generated_image(images: torch.Tensor, nrow: int = 8, show: bool = True, path: Optional[str] = None,
                         format: Optional[str] = None, to_grayscale: bool = False, **kwargs):
    """
    Concat all images into a single picture, considering additional dimensions for intermediate samples.

    Parameters:
        images: a tensor with shape (batch_size, sample, channels, height, width).
        nrow: decide how many images per row. Default `8`.
        show: whether to display the image after stitching. Default `True`.
        path: the path to save the image. if None (default), will not save image.
        format: image format. You can print the set of available formats by running `python3 -m PIL`.
        to_grayscale: convert PIL image to grayscale version of image. Default `False`.
        **kwargs: other arguments for `torchvision.utils.make_grid`.

    Returns:
        concat image, a tensor with shape (height, width, channels).
    """
    images = images * 0.5 + 0.5  # normalize images
    if images.ndim == 5:  # (batch_size, sample, channels, height, width)
        images = images.view(-1, images.shape[2], images.shape[3], images.shape[4])  # merge batch and sample dimensions

    grid = make_grid(images, nrow=nrow, **kwargs)  # (channels, height, width)
    grid = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()  # format for PIL

    im = Image.fromarray(grid)
    if to_grayscale:
        im = im.convert(mode="L")
    if path is not None:
        im.save(path, format=format)
    if show:
        im.show()
    return grid

def save_samples_image(images: torch.Tensor, show: bool = True, path: Optional[str] = None,
                       format: Optional[str] = None, to_grayscale: bool = False, **kwargs):
    """
    concat all image including intermediate process into a picture.

    Parameters:
        images: images including intermediate process,
            a tensor with shape (batch_size, sample, channels, height, width).
        show: whether to display the image after stitching. Default `True`.
        path: the path to save the image. if None (default), will not save image.
        format: image format. You can print the set of available formats by running `python3 -m PIL`.
        to_grayscale: convert PIL image to grayscale version of image. Default `False`.
        **kwargs: other arguments for `torchvision.utils.make_grid`.

    Returns:
        concat image, a tensor with shape (height, width, channels).
    """
    images = images * 0.5 + 0.5

    grid = []
    for i in range(images.shape[0]):
        # for each sample in batch, concat all intermediate process images in a row
        t = make_grid(images[i], nrow=images.shape[1], **kwargs)  # (channels, height, width)
        grid.append(t)
    # stack all merged images to a tensor
    grid = torch.stack(grid, dim=0)  # (batch_size, channels, height, width)
    grid = make_grid(grid, nrow=1, **kwargs)  # concat all batch images in a different row, (channels, height, width)
    #  (height, width, channels)
    grid = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()

    im = Image.fromarray(grid)
    if to_grayscale:
        im = im.convert(mode="L")
    if path is not None:
        im.save(path, format=format)
    if show:
        im.show()
    return grid


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