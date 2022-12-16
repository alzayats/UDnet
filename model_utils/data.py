from os.path import join
from torchvision.transforms import Compose, ToTensor
from model_utils.dataset import DatasetFromFolderEval,  DatasetFromFolder


def transform():
    return Compose([
        # Resize((512, 512)),
        ToTensor(),
    ])


def get_training_set(path, label, data, patch_size, data_augmentation):
    label_dir = join(path, label)
    data_dir = join(path, data)

    return DatasetFromFolder(label_dir, data_dir, patch_size, data_augmentation, transform=transform())


def get_eval_set(data_dir, label_dir):
    return DatasetFromFolderEval(data_dir, label_dir, transform=transform())


