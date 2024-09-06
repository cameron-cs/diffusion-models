import torch
from typing import Optional, Union
from torchvision.utils import make_grid
from PIL import Image
from pathlib2 import Path

from model.diffusion_sampler import DDIM
from model.unet import ConditionalUNet

def remove_module_prefix(state_dict):
    """
    Remove 'module.' prefix from state dict keys
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '', 1)
        new_state_dict[new_key] = v
    return new_state_dict


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
    images = images * 0.5 + 0.5  # normalise images
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


if __name__ == '__main__':
    cifar10_labels = {
        0: "airplane",
        1: "automobile",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck"
    }

    generate_conf = {
        'callback_path': 'callback/ddimp_cifar10.pth',
        'device': 'cuda:0',
        'batch_size': 256,
        'interval': 50,
        'eta': 0.0,
        'steps': 200,
        'method': 'quadratic',
        'nrow': 64,
        'show': True,
        'samples_image_save_path': 'data/samples/ddim_cifar_samples.png',
        'generated_image_save_path': 'data/generated/ddim_cifar_generated.png',
        'to_grayscale': False
    }

    device = torch.device(generate_conf['device'])

    cp = torch.load(generate_conf['callback_path'])

    # load trained model and adjust state dict for DataParallel
    model = ConditionalUNet(**cp["config"]["Model"])
    state_dict = remove_module_prefix(cp["model"])
    model.load_state_dict(state_dict)
    model = model.cuda()
    model = model.eval()

    # assuming you want to generate images of a specific class
    target_label = 0  # example: class 0, modify this value to generate different classes
    batch_size = 4

    # create labels tensor
    labels = torch.full((batch_size,), target_label, dtype=torch.long, device=device)

    # initialise the sampler with model and configuration
    sampler = DDIM(model, **cp["config"]["Trainer"]).to(device)

    # generate Gaussian noise
    z_t = torch.randn((batch_size, 3, 32, 32), device=device)

    # pass the labels argument to the sampler along with the extra parameters
    extra_param = {'steps': 200, 'eta': 0.0, 'method': 'quadratic'}
    x = sampler(z_t, labels=labels, only_return_x_0=False, interval=50, **extra_param)

    # save and display the generated samples
    save_samples_image(x, show=generate_conf['show'], path=generate_conf['samples_image_save_path'],
                       to_grayscale=generate_conf['to_grayscale'])
    save_generated_image(x, nrow=generate_conf['nrow'], show=generate_conf['show'], path=generate_conf['generated_image_save_path'],
                         to_grayscale=generate_conf['to_grayscale'])

    # print out the class label name
    print(f"Generated {batch_size} samples of class: {cifar10_labels[target_label]}")