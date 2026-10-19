"""UDnet V2: the UDnet design (V1's conditional VAE with latent feature modulation), modernised
with NAFNet blocks (Chen et al., ECCV 2022) and a global colour branch, and trained on paired
underwater images.

    model = load_udnet_v2('weights/UDnet_v2.pth', device='cuda')
    out = model.enhance(x)      # x: (1, 3, H, W) in [0, 1] -> same shape, in [0, 1]

During training a posterior encoder sees the input and its SGMCSS-refined reference and samples the
latent code; at inference the prior encoder predicts the code from the input alone (as in V1).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

from model_utils.colournet import USLN   # SGMCSS, as in V1


class LayerNorm2d(nn.Module):
    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


class SimpleGate(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock(nn.Module):
    def __init__(self, c, dw_expand=2, ffn_expand=2):
        super().__init__()
        dw = c * dw_expand
        self.conv1 = nn.Conv2d(c, dw, 1)
        self.conv2 = nn.Conv2d(dw, dw, 3, padding=1, groups=dw)
        self.conv3 = nn.Conv2d(dw // 2, c, 1)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dw // 2, dw // 2, 1))
        self.sg = SimpleGate()
        ffn = c * ffn_expand
        self.conv4 = nn.Conv2d(c, ffn, 1)
        self.conv5 = nn.Conv2d(ffn // 2, c, 1)
        self.norm1, self.norm2 = LayerNorm2d(c), LayerNorm2d(c)
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, inp):
        x = self.conv2(self.conv1(self.norm1(inp)))
        x = self.sg(x)
        x = self.conv3(x * self.sca(x))
        y = inp + x * self.beta
        x = self.conv5(self.sg(self.conv4(self.norm2(y))))
        return y + x * self.gamma


class GlobalColour(nn.Module):
    """Predicts a per-image 3x3 colour matrix + offset from a downsampled view.
    Underwater casts are largely global, so this gives the U-Net a good starting point."""

    def __init__(self, width=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, width, 3, 2, 1), nn.GELU(),
            nn.Conv2d(width, width * 2, 3, 2, 1), nn.GELU(),
            nn.Conv2d(width * 2, width * 4, 3, 2, 1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(width * 4, 12))
        nn.init.zeros_(self.net[-1].weight)  # start as the identity colour transform
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        p = self.net(F.interpolate(x, size=(128, 128), mode='bilinear', align_corners=False))
        m = p[:, :9].view(-1, 3, 3) + torch.eye(3, device=x.device, dtype=p.dtype)
        b = p[:, 9:].view(-1, 3, 1, 1)
        return torch.einsum('bij,bjhw->bihw', m, x) + b


def pad_to(x, m):
    h, w = x.shape[-2:]
    ph, pw = (-h) % m, (-w) % m
    return (F.pad(x, (0, pw, 0, ph), mode='reflect') if (ph or pw) else x), h, w


class Encoder(nn.Module):
    def __init__(self, cin, width, blocks):
        super().__init__()
        self.intro = nn.Conv2d(cin, width, 3, padding=1)
        self.stages, self.downs = nn.ModuleList(), nn.ModuleList()
        c = width
        for n in blocks:
            self.stages.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))
            self.downs.append(nn.Conv2d(c, 2 * c, 2, 2))
            c *= 2
        self.out_ch = c

    def forward(self, x):
        x = self.intro(x)
        skips = []
        for st, dn in zip(self.stages, self.downs):
            x = st(x)
            skips.append(x)
            x = dn(x)
        return x, skips


class Latent(nn.Module):
    """Globally pooled features -> Normal distributions for the modulation codes u and s (as in V1)."""

    def __init__(self, cin, z):
        super().__init__()
        self.u = nn.Linear(cin, 2 * z)
        self.s = nn.Linear(cin, 2 * z)

    def forward(self, f):
        g = f.mean((2, 3))
        (um, ul), (sm, sl) = self.u(g).chunk(2, 1), self.s(g).chunk(2, 1)
        return Normal(um, ul.clamp(-7, 5).exp()), Normal(sm, sl.clamp(-7, 5).exp())


class UDnetV2(nn.Module):
    """Prior encoder (U-Net encoder of NAFBlocks, 4 scales) + middle blocks + decoder whose every stage
    is modulated by the latent code (PAdaIN-style per-channel scale and shift). A posterior encoder
    and SGMCSS are used only in train_step."""

    def __init__(self, width=32, enc=(2, 2, 2, 2), middle=4, dec=(1, 1, 1, 1), z=20):
        super().__init__()
        self.colour = GlobalColour()
        self.sgmcss = USLN()
        self.pr_enc = Encoder(3, width, enc)
        self.po_enc = Encoder(6, width, enc)
        c = self.pr_enc.out_ch
        self.middle = nn.Sequential(*[NAFBlock(c) for _ in range(middle)])
        self.pr_lat, self.po_lat = Latent(c, z), Latent(c, z)
        self.ups, self.decs, self.mod = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for n in dec:
            self.ups.append(nn.Sequential(nn.Conv2d(c, c * 2, 1, bias=False), nn.PixelShuffle(2)))
            c //= 2
            self.decs.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))
            self.mod.append(nn.Linear(2 * z, 2 * c))   # (u, s) -> per-channel shift and scale
        for m in self.mod:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        self.ending = nn.Conv2d(width, 3, 3, padding=1)
        self.pad = 2 ** len(enc)

    def _encode(self, base):
        f, skips = self.pr_enc(base)
        return self.middle(f), skips

    def _decode(self, base, f, skips, code):
        for up, dec, mod, skip in zip(self.ups, self.decs, self.mod, skips[::-1]):
            f = up(f) + skip
            shift, scale = mod(code).chunk(2, 1)
            f = f * (1 + scale[..., None, None]) + shift[..., None, None]
            f = dec(f)
        return self.ending(f) + base

    def train_step(self, x, t):
        """x: input, t: reference. Returns (output, KL(posterior || prior))."""
        base = self.colour(x)
        f, skips = self._encode(base)
        pr_u, pr_s = self.pr_lat(f)
        with torch.autocast(x.device.type, enabled=False):   # SGMCSS colour conversions need fp32
            tc = self.sgmcss(t.float())
        po_f, _ = self.po_enc(torch.cat([x, tc], 1))
        po_u, po_s = self.po_lat(po_f)
        code = torch.cat([po_u.rsample(), po_s.rsample()], 1)
        kl = kl_divergence(po_u, pr_u).mean() + kl_divergence(po_s, pr_s).mean()
        return self._decode(base, f, skips, code), kl

    def forward(self, x):
        x, h, w = pad_to(x, self.pad)
        base = self.colour(x)
        f, skips = self._encode(base)
        pr_u, pr_s = self.pr_lat(f)
        return self._decode(base, f, skips, torch.cat([pr_u.mean, pr_s.mean], 1))[..., :h, :w]

    @torch.no_grad()
    def enhance(self, x):
        """x: (B, 3, H, W) in [0, 1]. Runs in bfloat16 on a GPU, float32 on a CPU."""
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=x.is_cuda):
            out = self(x)
        return out.float().clamp(0, 1)


class UDnetV2Preview(nn.Module):
    """The V2 preview network (NAFNet U-Net, 29 M parameters), kept so its archived weights still load."""

    def __init__(self, width=32, enc_blocks=(2, 2, 4, 8), middle_blocks=12, dec_blocks=(2, 2, 2, 2)):
        super().__init__()
        self.colour = GlobalColour()
        self.intro = nn.Conv2d(3, width, 3, padding=1)
        self.ending = nn.Conv2d(width, 3, 3, padding=1)
        self.encoders, self.decoders, self.downs, self.ups = (nn.ModuleList() for _ in range(4))
        c = width
        for n in enc_blocks:
            self.encoders.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))
            self.downs.append(nn.Conv2d(c, 2 * c, 2, 2))
            c *= 2
        self.middle = nn.Sequential(*[NAFBlock(c) for _ in range(middle_blocks)])
        for n in dec_blocks:
            self.ups.append(nn.Sequential(nn.Conv2d(c, c * 2, 1, bias=False), nn.PixelShuffle(2)))
            c //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))
        self.pad = 2 ** len(enc_blocks)

    def forward(self, inp):
        h, w = inp.shape[-2:]
        ph, pw = (-h) % self.pad, (-w) % self.pad
        x0 = F.pad(inp, (0, pw, 0, ph), mode='reflect') if (ph or pw) else inp
        base = self.colour(x0)
        x = self.intro(base)
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            skips.append(x)
            x = down(x)
        x = self.middle(x)
        for dec, up, skip in zip(self.decoders, self.ups, skips[::-1]):
            x = dec(up(x) + skip)
        out = self.ending(x) + base
        return out[..., :h, :w]

    @torch.no_grad()
    def enhance(self, x):
        """x: (B, 3, H, W) in [0, 1]. Runs in bfloat16 on a GPU, float32 on a CPU."""
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=x.is_cuda):
            out = self(x)
        return out.float().clamp(0, 1)


def load_udnet_v2(weights, device='cpu'):
    """Build UDnet V2 and load the released UDnet_v2.pth (or a training checkpoint with an 'ema' key).
    The archived preview weights (UDnet_v2_preview.pth) load into UDnetV2Preview automatically."""
    s = torch.load(weights, map_location='cpu', weights_only=True)
    cfg = s.get('config', {})
    state = s.get('state_dict', s.get('ema'))
    if 'pr_enc.intro.weight' in state:
        model = UDnetV2()
    else:
        model = UDnetV2Preview(width=cfg.get('width', 32), middle_blocks=cfg.get('middle', 12))
    model.load_state_dict({k: v.float() for k, v in state.items()}, strict=True)
    return model.to(device).eval()
