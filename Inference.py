"""Enhance underwater images with a pretrained UDnet model.

    python Inference.py --input path/to/images --output results/
    python Inference.py --input photo.jpg
"""
import os
import sys
import time
import argparse
import numpy as np
import torch
from PIL import Image

from model_utils.UDnet import load_udnet

WEIGHTS_URL = 'https://github.com/alzayats/UDnet/releases/download/v1.0/UDnet.pth'
WEIGHTS_GDRIVE = 'https://drive.google.com/file/d/1SsOoMmO1V2cuYDoFIWfQT_MPmx5lmYZu/view?usp=sharing'
IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def str2bool(v):
    # argparse type=bool is broken: bool("False") is True. Parse the string properly.
    return str(v).lower() in ("true", "1", "yes", "t", "y")


parser = argparse.ArgumentParser(description='Enhance underwater images with UDnet')
parser.add_argument('--input', '--input_dir', dest='input', default='./check_IQA/input/',
                    help='an image file or a folder of images')
parser.add_argument('--output', default='./results/', help='folder to write enhanced images to')
parser.add_argument('--weights', '--model', dest='weights', default='weights/UDnet.pth',
                    help='path to the pretrained model (downloaded automatically if missing)')
parser.add_argument('--device', default='auto', help="'auto', 'cpu', 'cuda', 'cuda:1', ...")
parser.add_argument('--format', default='png', choices=['png', 'jpg', 'same'],
                    help="output file format; 'same' keeps the input extension (png is lossless)")
parser.add_argument('--gpu_mode', type=str2bool, default=True, help='set False to run on CPU')


def list_images(path):
    if os.path.isfile(path):
        return [path]
    if not os.path.isdir(path):
        sys.exit(f'Input not found: {path}')
    files = sorted(os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(IMG_EXTENSIONS))
    if not files:
        sys.exit(f'No images ({", ".join(IMG_EXTENSIONS)}) found in {path}')
    return files


def ensure_weights(path):
    if os.path.isfile(path):
        return
    print(f'Weights not found at {path}, downloading from {WEIGHTS_URL} ...')
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    try:
        torch.hub.download_url_to_file(WEIGHTS_URL, path)
    except Exception as e:
        sys.exit(f'Download failed ({e}).\nPlease download UDnet.pth manually from {WEIGHTS_GDRIVE}\n'
                 f'and save it as {path}')


def pick_device(opt):
    if not opt.gpu_mode or opt.device == 'cpu':
        return torch.device('cpu')
    if opt.device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(opt.device)


def main():
    opt = parser.parse_args()
    files = list_images(opt.input)
    ensure_weights(opt.weights)
    device = pick_device(opt)
    model = load_udnet(opt.weights, device)
    os.makedirs(opt.output, exist_ok=True)
    print(f'Enhancing {len(files)} image(s) on {device} -> {opt.output}')

    for path in files:
        img = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255.0
        x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
        t0 = time.time()
        out = model.enhance(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        dt = time.time() - t0
        out = (out[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)

        stem, ext = os.path.splitext(os.path.basename(path))
        ext = ext if opt.format == 'same' else '.' + opt.format
        save_path = os.path.join(opt.output, stem + ext)
        Image.fromarray(out).save(save_path, **({'quality': 95} if ext.lower() in ('.jpg', '.jpeg') else {}))
        print(f'  {os.path.basename(path)}  {img.shape[1]}x{img.shape[0]}  {dt:.3f}s')


if __name__ == '__main__':
    main()
