"""Real-ESRGAN (Wang et al., ICCVW 2021) RRDBNet generator, re-implemented so the official
x2plus / x4plus weights load without basicsr. Used as the optional "AI" detail stage after UDnet V2.

    up = RealESRGAN(scale=2, weights='weights/RealESRGAN_x2plus.pth').cuda()
    y = up(x)        # x in [0, 1], (B, 3, H, W) -> (B, 3, 2H, 2W)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        return self.conv5(torch.cat((x, x1, x2, x3, x4), 1)) * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.rdb1, self.rdb2, self.rdb3 = RDB(nf, gc), RDB(nf, gc), RDB(nf, gc)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, scale=4, nf=64, nb=23, gc=32):
        super().__init__()
        self.scale = scale
        in_ch = 3 * (16 // scale ** 2) if scale < 4 else 3  # x2 model works on a pixel-unshuffled input
        self.conv_first = nn.Conv2d(in_ch, nf, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        if self.scale == 2:
            x = F.pixel_unshuffle(x, 2)
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


class RealESRGAN(nn.Module):
    """RRDBNet with tiled inference so large images fit in GPU memory."""

    def __init__(self, scale, weights, tile=512, pad=32):
        super().__init__()
        self.net = RRDBNet(scale)
        s = torch.load(weights, map_location='cpu', weights_only=True)
        self.net.load_state_dict(s.get('params_ema', s.get('params', s)))
        self.net.eval()
        self.scale, self.tile, self.pad = scale, tile, pad

    @torch.no_grad()
    def forward(self, x):
        b, c, h, w = x.shape
        s, t, p = self.scale, self.tile, self.pad
        # the x2 model unshuffles by 2, so work on even sizes
        ph, pw = h % 2, w % 2
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode='reflect')
        H, W = x.shape[-2:]
        out = x.new_zeros(b, c, H * s, W * s)
        for i in range(0, H, t):
            for j in range(0, W, t):
                i0, j0 = max(i - p, 0), max(j - p, 0)
                i1, j1 = min(i + t + p, H), min(j + t + p, W)
                with torch.autocast('cuda', dtype=torch.float16, enabled=x.is_cuda):
                    y = self.net(x[..., i0:i1, j0:j1]).float()
                ti, tj = min(i + t, H), min(j + t, W)
                out[..., i * s:ti * s, j * s:tj * s] = y[..., (i - i0) * s:(ti - i0) * s, (j - j0) * s:(tj - j0) * s]
        return out[..., :h * s, :w * s].clamp(0, 1)
