from torchvision.transforms import Compose, ToTensor
from model_utils.dataset import DatasetFromFolderEval,  DatasetFromFolder


def transform():
    return Compose([
        # Resize((512, 512)),
        ToTensor(),
    ])


def get_training_set(data_dir, patch_size, data_augmentation, file_list=None):
    # UDnet is unsupervised: targets are generated from the input images on the fly.
    return DatasetFromFolder(data_dir, data_dir, patch_size, data_augmentation, transform=transform(),
                             file_list=file_list)


def get_eval_set(data_dir, label_dir):
    return DatasetFromFolderEval(data_dir, label_dir, transform=transform())


