"""Train UDnet V2 on paired underwater images (UIEB and LSUI).

    pip install kornia lpips tensorboard     # training-only extras
    python Train_v2.py --uieb path/to/UIEB --lsui path/to/LSUI --name my_v2

--uieb must contain raw-890/ and reference-890/; --lsui must contain input/ and GT/. The training
and test lists are in splits/V2 (UIEB test: splits/UIEB/test90.txt). Training images that are
near-duplicates of any test image were removed from these lists.

The released UDnet_v2.pth is the EMA model at 80k of 100k iterations with the default settings
(the full run takes about 5 hours on one RTX 4090). Checkpoints go to weights/v2_runs/<name>/; best.pth can be used
directly with `Inference.py --model v2 --weights weights/v2_runs/<name>/best.pth`.
"""
import os
import copy
import json
import math
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from skimage.metrics import structural_similarity

from model_utils.UDnet_v2 import UDnetV2

SPLITS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'splits')

parser = argparse.ArgumentParser(description='Train UDnet V2 on paired images')
parser.add_argument('--uieb', required=True, help='UIEB folder with raw-890/ and reference-890/')
parser.add_argument('--lsui', required=True, help='LSUI folder with input/ and GT/')
parser.add_argument('--name', default='v2')
parser.add_argument('--iters', type=int, default=100000)
parser.add_argument('--bs', type=int, default=8)
parser.add_argument('--crop', type=int, default=256)
parser.add_argument('--lr', type=float, default=4e-4)
parser.add_argument('--warm', type=int, default=2000, help='linear warm-up iterations')
parser.add_argument('--width', type=int, default=32)
parser.add_argument('--middle', type=int, default=12)
parser.add_argument('--w_ssim', type=float, default=0.2)
parser.add_argument('--w_lpips', type=float, default=0.1)
parser.add_argument('--w_fft', type=float, default=0.05)
parser.add_argument('--ema', type=float, default=0.999)
parser.add_argument('--val_every', type=int, default=5000)
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--resume', action='store_true', help='continue from weights/v2_runs/<name>/last.pth')
parser.add_argument('--init', default=None, help='start from a V2 checkpoint, e.g. weights/UDnet_v2.pth (fine-tuning)')


# ---------------------------------------------------------------- data

def read_list(name):
    with open(os.path.join(SPLITS, name)) as f:
        return [l.strip() for l in f if l.strip()]


def load_rgb(path):
    return np.array(Image.open(path).convert('RGB'))


def to_tensor(a):
    return torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).float() / 255.0


def resize(a, size):
    return np.asarray(Image.fromarray(a).resize(size, Image.BICUBIC))


class PairedTrain(Dataset):
    """Random crops with scale jitter and flips. Each set is sampled in proportion to its size."""

    def __init__(self, sets, crop, length):
        self.items = [[(os.path.join(inp, n), os.path.join(ref, n)) for n in read_list(lst)]
                      for inp, ref, lst in sets]
        sizes = np.array([len(i) for i in self.items], dtype=np.float64)
        self.p = sizes / sizes.sum()
        self.crop, self.length = crop, length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        inp, ref = random.choice(self.items[np.random.choice(len(self.items), p=self.p)])
        x, y = load_rgb(inp), load_rgb(ref)
        h, w = x.shape[:2]
        c = self.crop
        if min(h, w) < c:  # upscale small images so a full crop fits
            sc = c / min(h, w)
            size = (max(c, round(w * sc)), max(c, round(h * sc)))
            x, y = resize(x, size), resize(y, size)
            h, w = x.shape[:2]
        if min(h, w) > 2 * c and random.random() < 0.5:  # scale jitter
            sc = random.uniform(c / min(h, w) * 1.2, 1.0)
            size = (round(w * sc), round(h * sc))
            x, y = resize(x, size), resize(y, size)
            h, w = x.shape[:2]
        i, j = random.randint(0, h - c), random.randint(0, w - c)
        x, y = x[i:i + c, j:j + c], y[i:i + c, j:j + c]
        if random.random() < 0.5:
            x, y = x[:, ::-1], y[:, ::-1]
        if random.random() < 0.5:
            x, y = x[::-1], y[::-1]
        return to_tensor(x), to_tensor(y)


# ---------------------------------------------------------------- loss

class UIELoss(nn.Module):
    """Charbonnier + SSIM + LPIPS (VGG) + frequency (FFT amplitude and phase) loss."""

    def __init__(self, w_ssim, w_lpips, w_fft):
        super().__init__()
        import lpips
        self.lp = lpips.LPIPS(net='vgg', verbose=False).eval()
        for q in self.lp.parameters():
            q.requires_grad = False
        self.w = dict(ssim=w_ssim, lpips=w_lpips, fft=w_fft)

    @staticmethod
    def fft(x, y):
        fx, fy = torch.fft.rfft2(x, norm='ortho'), torch.fft.rfft2(y, norm='ortho')
        return (fx.abs() - fy.abs()).abs().mean() + 0.1 * (torch.angle(fx) - torch.angle(fy)).abs().mean()

    def forward(self, out, gt):
        import kornia
        terms = {'char': torch.sqrt((out - gt) ** 2 + 1e-6).mean()}
        if self.w['ssim']:
            terms['ssim'] = kornia.losses.ssim_loss(out.clamp(0, 1), gt, window_size=11) * 2
        if self.w['lpips']:
            terms['lpips'] = self.lp(out * 2 - 1, gt * 2 - 1).mean()
        if self.w['fft']:
            terms['fft'] = self.fft(out, gt)
        total = terms['char'] + sum(self.w[k] * v for k, v in terms.items() if k != 'char')
        return total, {k: v.item() for k, v in terms.items()}


# ---------------------------------------------------------------- training

@torch.no_grad()
def evaluate(net, tests, dev):
    """Mean PSNR and SSIM on each held-out test set, computed on 8-bit outputs."""
    res = {}
    for name, items in tests.items():
        ps, ss = [], []
        for inp, ref in items:
            x = to_tensor(load_rgb(inp))[None].to(dev)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                o = net(x)
            o = (o.float().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
            g = load_rgb(ref)
            mse = ((o.astype(np.float64) - g) ** 2).mean()
            ps.append(10 * np.log10(255 ** 2 / max(mse, 1e-10)))
            ss.append(structural_similarity(o, g, channel_axis=2, data_range=255))
        res[name] = (float(np.mean(ps)), float(np.mean(ss)))
    return res


def main():
    opt = parser.parse_args()
    root = os.path.join('weights', 'v2_runs', opt.name)
    os.makedirs(root, exist_ok=True)
    json.dump(vars(opt), open(os.path.join(root, 'config.json'), 'w'), indent=1)
    dev = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(0)
    np.random.seed(0)

    uieb = (os.path.join(opt.uieb, 'raw-890'), os.path.join(opt.uieb, 'reference-890'))
    lsui = (os.path.join(opt.lsui, 'input'), os.path.join(opt.lsui, 'GT'))
    train = PairedTrain([(*uieb, 'V2/UIEB_train.txt'), (*lsui, 'V2/LSUI_train.txt')], opt.crop, opt.iters * opt.bs)
    tests = {name: [(os.path.join(inp, n), os.path.join(ref, n)) for n in read_list(lst)]
             for name, (inp, ref), lst in (('UIEB', uieb, 'UIEB/test90.txt'), ('LSUI', lsui, 'V2/LSUI_test400.txt'))}

    model = UDnetV2(width=opt.width, middle_blocks=opt.middle).to(dev)
    ema = copy.deepcopy(model).eval()
    for q in ema.parameters():
        q.requires_grad = False
    print(f'params: {sum(q.numel() for q in model.parameters()) / 1e6:.2f} M', flush=True)

    crit = UIELoss(opt.w_ssim, opt.w_lpips, opt.w_fft).to(dev)
    optim = torch.optim.AdamW(model.parameters(), lr=opt.lr, betas=(0.9, 0.99), weight_decay=1e-4)
    # linear warm-up, then cosine decay to 1% of the base learning rate
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lambda it: min(1, (it + 1) / opt.warm) * (
        0.5 * (1 + math.cos(math.pi * min(it, opt.iters) / opt.iters)) * 0.99 + 0.01))

    start, best = 0, -1
    if opt.init:
        s = torch.load(opt.init, map_location='cpu', weights_only=True)
        w = {k: v.float() for k, v in s.get('state_dict', s.get('ema')).items()}
        model.load_state_dict(w)
        ema.load_state_dict(w)
        print('initialised from', opt.init, flush=True)
    ck = os.path.join(root, 'last.pth')
    if opt.resume and os.path.exists(ck):
        s = torch.load(ck, map_location='cpu')
        model.load_state_dict(s['model'])
        ema.load_state_dict(s['ema'])
        optim.load_state_dict(s['optim'])
        sched.load_state_dict(s['sched'])
        start, best = s['it'], s['best']
        print('resumed at', start, flush=True)

    loader = DataLoader(train, batch_size=opt.bs, num_workers=opt.workers, drop_last=True,
                        persistent_workers=opt.workers > 0)
    tb = SummaryWriter(os.path.join(root, 'tb'))
    it, t0 = start, time.time()
    model.train()
    for x, y in loader:
        if it >= opt.iters:
            break
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(x)
        loss, terms = crit(out.float(), y)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optim.step()
        sched.step()
        with torch.no_grad():  # exponential moving average of the weights
            d = min(opt.ema, (1 + it) / (10 + it))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(d).add_(pm.detach(), alpha=1 - d)
        it += 1
        if it % 100 == 0:
            for k, v in terms.items():
                tb.add_scalar(f'loss/{k}', v, it)
            tb.add_scalar('lr', sched.get_last_lr()[0], it)
        if it % 1000 == 0:
            print(f'it {it} loss {loss.item():.4f} ' + ' '.join(f'{k}={v:.4f}' for k, v in terms.items()) +
                  f' | {(time.time() - t0) / 1000:.3f}s/it', flush=True)
            t0 = time.time()
        if it % opt.val_every == 0 or it == opt.iters:
            r = evaluate(ema, tests, dev)
            for n, (p_, s_) in r.items():
                tb.add_scalar(f'val/{n}_psnr', p_, it)
                tb.add_scalar(f'val/{n}_ssim', s_, it)
            print(f'VAL it {it}: ' + ' '.join(f'{n} {a:.2f}/{b:.4f}' for n, (a, b) in r.items()), flush=True)
            config = {'width': opt.width, 'middle': opt.middle}
            score = r['UIEB'][0] + r['LSUI'][0]
            if score > best:
                best = score
                torch.save({'ema': ema.state_dict(), 'it': it, 'val': r, 'config': config},
                           os.path.join(root, 'best.pth'))
            torch.save({'model': model.state_dict(), 'ema': ema.state_dict(), 'optim': optim.state_dict(),
                        'sched': sched.state_dict(), 'it': it, 'best': best, 'config': config}, ck)
            model.train()
    print('done', flush=True)


if __name__ == '__main__':
    main()
