import torch

from model.diffusion_sampler import DDIM
from model.unet import ConditionalUNet
from train import save_samples_image, save_generated_image


def remove_module_prefix(state_dict):
    """
    Remove 'module.' prefix from state dict keys
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '', 1)
        new_state_dict[new_key] = v
    return new_state_dict


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

    # initialize the sampler with model and configuration
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