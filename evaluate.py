"""Evaluate enhanced images.

Full-reference metrics (PSNR, SSIM) are computed when --ref is given; no-reference
metrics (UIQM, UCIQE) are always computed. Files are matched by name, ignoring the
extension (out/123.png <-> ref/123.jpg).

    python evaluate.py --pred results/UIEB --ref datasets/UIEB/reference-890
    python evaluate.py --pred results/RUIE --csv results/RUIE_metrics.csv
    python evaluate.py --pred results/UIEB --ref ... --list splits/UIEB_test90.txt

Note: UIQM/UCIQE absolute values depend on the implementation. UIQM follows the
widely used FUnIE-GAN `uqim_utils.py`; UCIQE follows the common OpenCV-Lab Python port.
Always compare methods with the same script.
"""
import os
import sys
import csv
import argparse
import numpy as np
import cv2
from scipy import ndimage
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


# ---------------------------------------------------------------- UIQM (Panetta et al. 2016)
def _trimmed_mean(x, alpha_l=0.1, alpha_r=0.1):
    # Asymmetric alpha-trimmed mean, indexed exactly like the reference implementation.
    x = np.sort(x)
    k = len(x)
    t_l, t_r = int(np.ceil(alpha_l * k)), int(np.floor(alpha_r * k))
    return x[t_l + 1:k - t_r].sum() / (k - t_l - t_r)


def _uicm(x):
    r, g, b = (x[..., i].ravel().astype(np.float64) for i in range(3))
    rg, yb = r - g, (r + g) / 2 - b
    mu_rg, mu_yb = _trimmed_mean(rg), _trimmed_mean(yb)
    s_rg, s_yb = np.mean((rg - mu_rg) ** 2), np.mean((yb - mu_yb) ** 2)
    return -0.0268 * np.hypot(mu_rg, mu_yb) + 0.1586 * np.sqrt(s_rg + s_yb)


def _blocks(x, w):
    k1, k2 = x.shape[1] // w, x.shape[0] // w
    x = x[:k2 * w, :k1 * w]
    # -> (k2, k1, w, w[, c]) then flatten each block
    b = x.reshape(k2, w, k1, w, *x.shape[2:]).swapaxes(1, 2)
    return b.reshape(k2 * k1, -1), k1 * k2


def _eme(x, w=10):
    b, n = _blocks(x, w)
    mx, mn = b.max(1), b.min(1)
    ok = (mn != 0) & (mx != 0)
    return 2.0 / n * np.log(mx[ok] / mn[ok]).sum()


def _sobel(x):
    mag = np.hypot(ndimage.sobel(x, 0), ndimage.sobel(x, 1))
    return mag * (255.0 / np.max(mag)) if np.max(mag) > 0 else mag


def _uism(x):
    weights = (0.299, 0.587, 0.144)
    return sum(wt * _eme(_sobel(x[..., i]) * x[..., i]) for i, wt in enumerate(weights))


def _uiconm(x, w=10):
    b, n = _blocks(x, w)
    mx, mn = b.max(1), b.min(1)
    top, bot = mx - mn, mx + mn
    ok = (top != 0) & (bot != 0)
    r = top[ok] / bot[ok]
    return -1.0 / n * (r * np.log(r)).sum()


def uiqm(rgb):
    x = rgb.astype(np.float32)
    return 0.0282 * _uicm(x) + 0.2953 * _uism(x) + 3.5753 * _uiconm(x)


# ---------------------------------------------------------------- UCIQE (Yang & Sowmya 2015)
def uciqe(rgb):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float64) / 255.0
    lum, a, b = lab[..., 0].ravel(), lab[..., 1].ravel(), lab[..., 2].ravel()
    chroma = np.sqrt(a ** 2 + b ** 2)
    order = np.argsort(lum)
    con_lum = lum[order[int(len(lum) * 0.99)]] - lum[order[int(len(lum) * 0.01)]]
    sat = np.divide(chroma, lum, out=np.zeros_like(chroma), where=lum != 0)
    return 0.4680 * chroma.std() + 0.2745 * con_lum + 0.2576 * sat.mean()


# ---------------------------------------------------------------- main
def read_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        sys.exit(f'Cannot read image: {path}')
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def index_images(folder):
    if not os.path.isdir(folder):
        sys.exit(f'Folder not found: {folder}')
    return {os.path.splitext(f)[0]: os.path.join(folder, f)
            for f in sorted(os.listdir(folder)) if f.lower().endswith(IMG_EXTENSIONS)}


def main():
    p = argparse.ArgumentParser(description='Evaluate enhanced underwater images')
    p.add_argument('--pred', required=True, help='folder of enhanced images')
    p.add_argument('--ref', default=None, help='folder of reference images (enables PSNR/SSIM)')
    p.add_argument('--list', default=None, help='optional split file; only these images are evaluated')
    p.add_argument('--csv', default=None, help='write per-image scores to this CSV file')
    opt = p.parse_args()

    preds = index_images(opt.pred)
    if opt.list:
        with open(opt.list) as f:
            wanted = [os.path.splitext(os.path.basename(l.strip()))[0] for l in f if l.strip()]
        missing = [w for w in wanted if w not in preds]
        if missing:
            sys.exit(f'{len(missing)} images from {opt.list} are missing in {opt.pred}, e.g. {missing[:3]}')
        preds = {k: preds[k] for k in wanted}
    refs = index_images(opt.ref) if opt.ref else {}
    if opt.ref:
        missing = [k for k in preds if k not in refs]
        if missing:
            sys.exit(f'{len(missing)} images have no reference in {opt.ref}, e.g. {missing[:3]}')

    rows = []
    for name, path in preds.items():
        img = read_rgb(path)
        row = {'image': name}
        if opt.ref:
            ref = read_rgb(refs[name])
            if ref.shape != img.shape:
                img_fr = cv2.resize(img, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_CUBIC)
            else:
                img_fr = img
            row['PSNR'] = peak_signal_noise_ratio(ref, img_fr, data_range=255)
            row['SSIM'] = structural_similarity(ref, img_fr, channel_axis=2, data_range=255)
        row['UIQM'] = uiqm(img)
        row['UCIQE'] = uciqe(img)
        rows.append(row)
        print(name, ' '.join(f'{k}={v:.4f}' for k, v in row.items() if k != 'image'))

    keys = [k for k in rows[0] if k != 'image']
    means = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    print(f'\n{len(rows)} images | ' + ' | '.join(f'{k} {v:.4f}' for k, v in means.items()))

    if opt.csv:
        os.makedirs(os.path.dirname(opt.csv) or '.', exist_ok=True)
        with open(opt.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['image'] + keys)
            w.writeheader()
            w.writerows(rows)
            w.writerow({'image': 'MEAN', **means})
        print('Saved', opt.csv)


if __name__ == '__main__':
    main()
