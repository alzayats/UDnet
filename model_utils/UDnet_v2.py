"""UDnet V2: a NAFNet restoration network (Chen et al., ECCV 2022) with a global colour branch,
trained on paired underwater images.

    model = load_udnet_v2('weights/UDnet_v2.pth', device='cuda')
    out = model.enhance(x)      # x: (1, 3, H, W) in [0, 1] -> same shape, in [0, 1]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class UDnetV2(nn.Module):
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
    """Build UDnet V2 and load the released UDnet_v2.pth (or a training checkpoint with an 'ema' key)."""
    s = torch.load(weights, map_location='cpu', weights_only=True)
    cfg = s.get('config', {})
    model = UDnetV2(width=cfg.get('width', 32), middle_blocks=cfg.get('middle', 12))
    state = s.get('state_dict', s.get('ema'))
    model.load_state_dict({k: v.float() for k, v in state.items()}, strict=True)
    return model.to(device).eval()
