"""Enhance underwater images with a pretrained UDnet model.

    python Inference.py --input path/to/images --output results/
    python Inference.py --input photo.jpg
    python Inference.py --input photo.jpg --model v2            # UDnet V2
    python Inference.py --input photo.jpg --model v2 --ai x2    # V2 + AI detail, 2x upscaled
    python Inference.py --input dive.mp4 --model v2             # videos too (written as .mp4, no audio)
"""
import os
import sys
import time
import argparse
import cv2
import numpy as np
import torch
from PIL import Image

from model_utils.UDnet import load_udnet
from model_utils.UDnet_v2 import load_udnet_v2

WEIGHTS = {  # default path -> download URL
    'v1': ('weights/UDnet.pth', 'https://huggingface.co/Alzayats/UDnet/resolve/main/UDnet.pth'),
    'v2': ('weights/UDnet_v2.pth', 'https://huggingface.co/Alzayats/UDnet/resolve/main/UDnet_v2.pth'),
    'x2': ('weights/RealESRGAN_x2plus.pth',
           'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'),
    'x4': ('weights/RealESRGAN_x4plus.pth',
           'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'),
}
IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi', '.mkv', '.m4v')


def str2bool(v):
    # argparse type=bool is broken: bool("False") is True. Parse the string properly.
    return str(v).lower() in ("true", "1", "yes", "t", "y")


parser = argparse.ArgumentParser(description='Enhance underwater images with UDnet')
parser.add_argument('--input', '--input_dir', dest='input', required=True,
                    help='an image or video file, or a folder of them')
parser.add_argument('--output', default='./results/', help='folder to write enhanced images to')
parser.add_argument('--model', default='v1', choices=['v1', 'v2'],
                    help="'v1': the paper model (fully unsupervised, fast on a CPU); "
                         "'v2': trained on paired images (stronger correction, best on a GPU)")
parser.add_argument('--weights', default=None,
                    help='path to the pretrained model (default: weights/UDnet.pth or weights/UDnet_v2.pth, '
                         'downloaded automatically if missing)')
parser.add_argument('--ai', default='none', choices=['none', 'x2', 'x4'],
                    help='add a Real-ESRGAN detail pass after UDnet: sharper and upscaled 2x or 4x. '
                         'It can invent fine texture, so use it for viewing, not measurement')
parser.add_argument('--device', default='auto', help="'auto', 'cpu', 'cuda', 'cuda:1', ...")
parser.add_argument('--format', default='png', choices=['png', 'jpg', 'same'],
                    help="output image format; 'same' keeps the input extension (png is lossless). "
                         "Videos are always written as .mp4")
parser.add_argument('--gpu_mode', type=str2bool, default=True, help='set False to run on CPU')


def list_images(path, extensions=IMG_EXTENSIONS):
    if os.path.isfile(path):
        return [path]
    if not os.path.isdir(path):
        sys.exit(f'Input not found: {path}')
    files = sorted(os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(extensions))
    if not files:
        sys.exit(f'No files ({", ".join(extensions)}) found in {path}')
    return files


def ensure_weights(path, url):
    if os.path.isfile(path):
        return
    print(f'Weights not found at {path}, downloading from {url} ...')
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    try:
        torch.hub.download_url_to_file(url, path)
    except Exception as e:
        sys.exit(f'Download failed ({e}).\nPlease download {os.path.basename(url)} manually from {url}\n'
                 f'and save it as {path}')


def pick_device(opt):
    if not opt.gpu_mode or opt.device == 'cpu':
        return torch.device('cpu')
    if opt.device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(opt.device)


def enhance_video(path, save_path, run):
    """Enhance every frame of a video and write it as .mp4 at the same frame rate (audio is not copied)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f'  {os.path.basename(path)}: cannot be read, skipped')
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer, n, t0 = None, 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out = run(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if writer is None:
            writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (out.shape[1], out.shape[0]))
        writer.write(cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        n += 1
        if n % 100 == 0:
            print(f'    frame {n}/{total}  {n / (time.time() - t0):.1f} frames/s', flush=True)
    cap.release()
    if writer is not None:
        writer.release()
    print(f'  {os.path.basename(path)}  {n} frames at {fps:.2f} fps -> {os.path.basename(save_path)}'
          f'  {time.time() - t0:.1f}s')


def main():
    opt = parser.parse_args()
    files = list_images(opt.input, IMG_EXTENSIONS + VIDEO_EXTENSIONS)
    default_path, url = WEIGHTS[opt.model]
    weights = opt.weights or default_path
    if weights == default_path:
        ensure_weights(weights, url)
    elif not os.path.isfile(weights):
        sys.exit(f'Weights not found: {weights}')
    device = pick_device(opt)
    model = (load_udnet if opt.model == 'v1' else load_udnet_v2)(weights, device)
    up = None
    if opt.ai != 'none':
        from model_utils.realesrgan import RealESRGAN
        ensure_weights(*WEIGHTS[opt.ai])
        up = RealESRGAN(int(opt.ai[1]), WEIGHTS[opt.ai][0]).to(device)
    os.makedirs(opt.output, exist_ok=True)
    print(f'Enhancing {len(files)} file(s) with UDnet {opt.model.upper()}'
          f'{" + AI " + opt.ai if up else ""} on {device} -> {opt.output}')

    def run(img):
        """HxWx3 uint8 RGB -> enhanced HxWx3 uint8 RGB (larger with --ai)."""
        x = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        out = model.enhance(x)
        if up is not None:
            out = up(out)
        return (out[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)

    for path in files:
        stem, ext = os.path.splitext(os.path.basename(path))
        if ext.lower() in VIDEO_EXTENSIONS:
            enhance_video(path, os.path.join(opt.output, stem + '.mp4'), run)
            continue
        img = np.array(Image.open(path).convert('RGB'))
        t0 = time.time()
        out = run(img)
        dt = time.time() - t0

        ext = ext if opt.format == 'same' else '.' + opt.format
        save_path = os.path.join(opt.output, stem + ext)
        Image.fromarray(out).save(save_path, **({'quality': 95} if ext.lower() in ('.jpg', '.jpeg') else {}))
        print(f'  {os.path.basename(path)}  {img.shape[1]}x{img.shape[0]} -> {out.shape[1]}x{out.shape[0]}  {dt:.3f}s')


if __name__ == '__main__':
    main()
