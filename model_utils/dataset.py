import os
import random
import torch.utils.data as data
from os import listdir
from os.path import join
from PIL import Image, ImageOps

from model_utils.label_gen import *


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".bmp"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img


def rescale_img(img_in, scale):
    size_in = img_in.size
    new_size_in = tuple([int(x * scale) for x in size_in])
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return img_in


def get_patch(img_in, img_tar, patch_size, scale=1, ix=-1, iy=-1):
    (ih, iw) = img_in.size

    patch_mult = scale
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)

    img_in = img_in.crop((iy, ix, iy + ip, ix + ip))
    img_tar = img_tar.crop((ty, tx, ty + tp, tx + tp))
                
    info_patch = {
        'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return img_in, img_tar, info_patch


def augment(img_in, img_tar, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}
    
    if random.random() < 0.5 and flip_h:
        img_in = ImageOps.flip(img_in)
        img_tar = ImageOps.flip(img_tar)
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_in = ImageOps.mirror(img_in)
            img_tar = ImageOps.mirror(img_tar)
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_in = img_in.rotate(180)
            img_tar = img_tar.rotate(180)
            info_aug['trans'] = True
            
    return img_in, img_tar, info_aug


class DatasetFromFolder(data.Dataset):
    def __init__(self, label_dir, data_dir, patch_size, data_augmentation, transform=None):
        super(DatasetFromFolder, self).__init__()
        self.label_path = label_dir 
        data_filenames = [join(data_dir, x) for x in listdir(data_dir) if is_image_file(x)]
        data_filenames.sort()
        self.data_filenames = data_filenames
        self.patch_size = patch_size
        self.transform = transform
        self.data_augmentation = data_augmentation

    def __getitem__(self, index):
        _, file = os.path.split(self.data_filenames[index])

        input = load_img(self.data_filenames[index])
        input_arr = np.array(input)

        k = random.randint(0,1)
        if k == 0:
            # target = RecoverCLAHE(input_arr)
            target = skimageintens(input_arr)
        if k == 1 :
            target = RecoverHE(input_arr)
        # if k == 2 :
        #     target = RecoverGC(input_arr) #1.5


        target = Image.fromarray(np.uint8(target))
        # target.save(f"./debug/{k}_{index}.png")

        input = input.resize((512, 512), resample=Image.BICUBIC)
        target = target.resize((512, 512), resample=Image.BICUBIC)
        input, target, _ = get_patch(input, target, self.patch_size)
        
        if self.data_augmentation:
            input, target, _ = augment(input, target)
        
        if self.transform:
            input = self.transform(input)
            target = self.transform(target)

        return input, target, file

    def __len__(self):
        return len(self.data_filenames)


class DatasetFromFolderEval(data.Dataset):
    def __init__(self, data_dir, label_dir, transform=None):
        super(DatasetFromFolderEval, self).__init__()
        data_filenames = [join(data_dir, x) for x in listdir(data_dir) if is_image_file(x)]
        data_filenames.sort()
        self.data_filenames = data_filenames

        label_filenames = [join(label_dir, x) for x in listdir(label_dir) if is_image_file(x)]
        label_filenames.sort()
        self.label_filenames = label_filenames

        self.transform = transform

    def __getitem__(self, index):
        input = load_img(self.data_filenames[index])
        label = load_img(self.label_filenames[index])
        _, file = os.path.split(self.data_filenames[index])

        (ih, iw) = input.size
        dh = ih % 8
        dw = iw % 8
        new_h, new_w = ih - dh, iw - dw

        input = input.resize((new_h, new_w))
        label = label.resize((new_h, new_w))

        if self.transform:
            input = self.transform(input)
            label = self.transform(label)
            
        return input, label, file
      
    def __len__(self):
        return len(self.data_filenames)


