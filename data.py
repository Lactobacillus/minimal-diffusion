import os
from easydict import EasyDict
from torchvision import datasets, transforms


def get_metadata(name):
    
    if name == "mnist":
    
        metadata = EasyDict({"image_size": 28,
                            "num_classes": 10,
                            "train_images": 60000,
                            "val_images": 10000,
                            "num_channels": 1})
    
    elif name == "cifar10":
    
        metadata = EasyDict({"image_size": 32,
                            "num_classes": 10,
                            "train_images": 50000,
                            "val_images": 10000,
                            "num_channels": 3})
    
    else:
    
        raise ValueError(f"{name} dataset nor supported!")
    
    return metadata


# TODO: Add datasets imagenette/birds/svhn etc etc.
def get_dataset(name, data_dir, metadata):
    """
    Return a dataset with the current name. We only support two datasets with
    their fixed image resolutions. One can easily add additional datasets here.

    Note: To avoid learning the distribution of transformed data, don't use heavy
        data augmentation with diffusion models.
    """
    if name == "mnist":

        transform_train = transforms.Compose([transforms.RandomResizedCrop(metadata.image_size, scale=(0.8, 1.0), ratio=(0.8, 1.2)), transforms.ToTensor()])
        train_set = datasets.MNIST(root=data_dir,
                                    train=True,
                                    download=True,
                                    transform=transform_train)

    elif name == "cifar10":

        transform_train = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        train_set = datasets.CIFAR10(root=data_dir,
                                    train=True,
                                    download=True,
                                    transform=transform_train)

    else:
        
        raise ValueError(f"{name} dataset nor supported!")
    
    return train_set
