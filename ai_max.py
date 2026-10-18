"""UDnet V2 + AI Max: UDnet V2 followed by DiffBIR v2.1 (Lin et al., ECCV 2024) detail restoration.

    bash setup_ai_max.sh                                    # once: DiffBIR + its conda env
    python ai_max.py --input path/to/images --output results/

1. UDnet V2 corrects colour and haze.
2. DiffBIR, run in its own conda environment, restores fine detail and upscales 2x, guided by an
   underwater text prompt.
3. Diffusion models drift in colour and can add texture to flat regions, so the final image keeps
   only DiffBIR's fine detail, and only where the V2 output already has texture. Colour, tone and
   flat areas such as open water come from V2.

The result can contain detail that was not captured by the camera: use it for viewing, not for
measurement. See AI_MAX.md for requirements and limitations.
"""
import os
import sys
import argparse
import subprocess
import tempfile
import numpy as np
import cv2
import torch
from PIL import Image

from Inference import WEIGHTS, list_images, ensure_weights
from model_utils.UDnet_v2 import load_udnet_v2

POS = ('underwater photograph, clear water, natural colours, highly detailed, sharp focus, '
       'realistic texture of fish, coral, rock and sand')
NEG = ('painting, illustration, drawing, cartoon, CG style, 3D render, blurry, oversaturated, '
       'watermark, text, jpeg artifacts, deformed, low quality, over-smooth')

parser = argparse.ArgumentParser(description='UDnet V2 + AI Max (DiffBIR detail restoration)')
parser.add_argument('--input', required=True, help='an image file or a folder of images')
parser.add_argument('--output', default='./results_ai_max/', help='folder to write results to')
parser.add_argument('--weights', default=WEIGHTS['v2'][0], help='UDnet V2 weights')
parser.add_argument('--diffbir', default='third_party/DiffBIR', help='DiffBIR checkout (from setup_ai_max.sh)')
parser.add_argument('--env', default='udnet-aimax', help='conda environment for DiffBIR (from setup_ai_max.sh)')
parser.add_argument('--upscale', type=float, default=2)
parser.add_argument('--steps', type=int, default=10, help='diffusion sampling steps')
parser.add_argument('--cfg', type=float, default=4.0, help='prompt guidance scale')
parser.add_argument('--seed', type=int, default=231)


def texture_mask(base, lo=2.0, hi=8.0):
    """1 where `base` already has local texture, 0 in flat areas such as open water."""
    g = cv2.cvtColor(base.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    m = cv2.GaussianBlur(g, (0, 0), 3)
    s = np.sqrt(np.maximum(cv2.GaussianBlur(g * g, (0, 0), 3) - m * m, 0))
    return np.clip((cv2.GaussianBlur(s, (0, 0), 8) - lo) / (hi - lo), 0, 1)[..., None]


def keep_colour(detail, base, sigma):
    """High frequencies from `detail` where `base` has texture; everything else from `base`.
    Both are float32 HxWx3 in 0-255 and the same size."""
    blur = lambda a: cv2.GaussianBlur(a, (0, 0), sigma)
    m = texture_mask(base)
    high = m * (detail - blur(detail)) + (1 - m) * (base - blur(base))
    return np.clip(high + blur(base), 0, 255)


def env_python(env):
    try:
        out = subprocess.run(['conda', 'run', '-n', env, 'python', '-c', 'import sys; print(sys.executable)'],
                             capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        sys.exit(f"conda environment '{env}' not found. Run `bash setup_ai_max.sh` first (see AI_MAX.md).")
    return out.stdout.strip().splitlines()[-1]


def main():
    opt = parser.parse_args()
    diffbir = os.path.abspath(opt.diffbir)
    if not os.path.isfile(os.path.join(diffbir, 'inference.py')):
        sys.exit(f'DiffBIR not found at {diffbir}. Run `bash setup_ai_max.sh` first (see AI_MAX.md).')
    if not torch.cuda.is_available():
        sys.exit('AI Max needs a CUDA GPU (about 10 GB of memory).')
    py = env_python(opt.env)
    files = list_images(opt.input)
    ensure_weights(opt.weights, WEIGHTS['v2'][1])
    os.makedirs(opt.output, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        v2_dir, raw_dir = os.path.join(tmp, 'v2'), os.path.join(tmp, 'raw')
        os.makedirs(v2_dir)
        print(f'1/3  UDnet V2 on {len(files)} image(s)', flush=True)
        model = load_udnet_v2(opt.weights, 'cuda')
        for path in files:
            img = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255.0
            out = model.enhance(torch.from_numpy(img).permute(2, 0, 1)[None].cuda())
            out = (out[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
            Image.fromarray(out).save(os.path.join(v2_dir, os.path.splitext(os.path.basename(path))[0] + '.png'))
        del model
        torch.cuda.empty_cache()

        print('2/3  DiffBIR detail restoration (the first run downloads about 6.7 GB of weights)', flush=True)
        cmd = [py, 'inference.py', '--task', 'sr', '--version', 'v2.1', '--upscale', str(opt.upscale),
               '--captioner', 'none', '--pos_prompt', POS, '--neg_prompt', NEG, '--cfg_scale', str(opt.cfg),
               '--steps', str(opt.steps), '--seed', str(opt.seed), '--precision', 'fp16',
               '--cldm_tiled', '--vae_encoder_tiled', '--vae_decoder_tiled',
               '--input', v2_dir, '--output', raw_dir]
        subprocess.run(cmd, cwd=diffbir, check=True)

        print('3/3  keeping V2 colour', flush=True)
        for f in sorted(os.listdir(v2_dir)):
            d = os.path.join(raw_dir, f)
            if not os.path.exists(d):
                print(f'  {f}: no DiffBIR output, skipped')
                continue
            detail = cv2.imread(d).astype(np.float32)
            base = cv2.imread(os.path.join(v2_dir, f)).astype(np.float32)
            base = cv2.resize(base, (detail.shape[1], detail.shape[0]), interpolation=cv2.INTER_CUBIC)
            out = keep_colour(detail, base, 3.0 * opt.upscale)
            cv2.imwrite(os.path.join(opt.output, f), out.round().astype(np.uint8))
            print(f'  {f}  {detail.shape[1]}x{detail.shape[0]}')
    print('done ->', opt.output)


if __name__ == '__main__':
    main()
