from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader


def get_transforms():
    """Configure and return a set of transforms for the CIFAR-10 dataset."""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def create_dataloader(dataset, batch_size, **kwargs):
    """Create and return a DataLoader with the specified dataset and parameters."""
    loader_params = {
        'shuffle': kwargs.get('shuffle', True),
        'drop_last': kwargs.get('drop_last', True),
        'pin_memory': kwargs.get('pin_memory', True),
        'num_workers': kwargs.get('num_workers', 4),
        'batch_size': batch_size
    }
    return DataLoader(dataset, **loader_params)


def create_cifar10_dataset(data_path, batch_size, **kwargs):
    """Create and return a DataLoader for the CIFAR-10 dataset with specified parameters."""
    train = kwargs.get('train', True)
    download = kwargs.get('download', True)

    dataset = CIFAR10(
        root=data_path,
        train=train,
        download=download,
        transform=get_transforms()
    )

    return create_dataloader(dataset, batch_size, **kwargs)