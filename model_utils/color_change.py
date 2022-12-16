import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.cuda.set_device(0)
# device='cpu'
def rgb2hsi(img):
    img = torch.clamp(img, 0, 1)
    r = img[:, 0, :, :]
    g = img[:, 1, :, :]
    b = img[:, 2, :, :]
    i = (r + g + b) / 3
    s = 1 - 3 * img.min(1)[0] / (r + g + b + 1e-5)
    x1 = (2 * r - b - g) / 2
    x2 = ((r - g) ** 2 + (r - b) * (g - b) + 1e-5) ** 0.5
    angle = torch.arccos(x1 / x2) / 2 / torch.pi
    # h = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(img.device)
    h = (b <= r) * angle + (b > r) * (1 - angle)
    h = h.unsqueeze(1)
    s = s.unsqueeze(1)
    i = i.unsqueeze(1)
    out = torch.cat((h, s, i), dim=1)
    return out


def hsi2rgb(img):
    img = torch.clamp(img, 0, 1)
    h = img[:, 0, :, :]
    s = img[:, 1, :, :]
    i = img[:, 2, :, :]
    r = torch.zeros_like(h)
    g = torch.zeros_like(h)
    b = torch.zeros_like(h)
    h1 = torch.zeros_like(h)
    hi0 = (h < 1 / 3)
    hi2 = (h >= 2 / 3)
    hi1 = 1 - hi0.int() - hi2.int()
    hi1 = (hi1 == 1)
    h1[hi0] = 2 * torch.pi * h[hi0]
    h1[hi1] = 2 * torch.pi * (h[hi1] - 1 / 3)
    h1[hi2] = 2 * torch.pi * (h[hi2] - 2 / 3)
    p = i * (1 - s)
    q = i * (1 + s * torch.cos(h1) / (torch.cos(torch.pi / 3 - h1) + 1e-5))

    r[hi0] = q[hi0]
    b[hi0] = p[hi0]
    g[hi0] = 3 * i[hi0] - r[hi0] - b[hi0]

    g[hi1] = q[hi1]
    r[hi1] = p[hi1]
    b[hi1] = 3 * i[hi1] - r[hi1] - g[hi1]

    b[hi2] = q[hi2]
    g[hi2] = p[hi2]
    r[hi2] = 3 * i[hi2] - g[hi2] - b[hi2]

    r = r.unsqueeze(1)
    g = g.unsqueeze(1)
    b = b.unsqueeze(1)
    out = torch.cat((r, g, b), dim=1)
    return out

def rgb2hsv(img):
        img = torch.clamp(img, 0, 1)
        hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(img.device)

        hue[img[:, 2] == img.max(1)[0]] = 4.0 + ((img[:, 0] - img[:, 1]) / (img.max(1)[0] - img.min(1)[0] + 1e-5))[
            img[:, 2] == img.max(1)[0]]
        hue[img[:, 1] == img.max(1)[0]] = 2.0 + ((img[:, 2] - img[:, 0]) / (img.max(1)[0] - img.min(1)[0] + 1e-5))[
            img[:, 1] == img.max(1)[0]]
        hue[img[:, 0] == img.max(1)[0]] = (0.0 + ((img[:, 1] - img[:, 2]) / (img.max(1)[0] - img.min(1)[0] + 1e-5))[
            img[:, 0] == img.max(1)[0]]) % 6

        hue[img.min(1)[0] == img.max(1)[0]] = 0.0
        hue = hue / 6

        saturation = (img.max(1)[0] - img.min(1)[0]) / (img.max(1)[0] + 1e-5)
        saturation[img.max(1)[0] == 0] = 0

        value = img.max(1)[0]

        hue = hue.unsqueeze(1)
        saturation = saturation.unsqueeze(1)
        value = value.unsqueeze(1)
        hsv = torch.cat([hue, saturation, value], dim=1)
        return hsv

def hsv2rgb(hsv):
        h, s, v = hsv[:, 0, :, :], hsv[:, 1, :, :], hsv[:, 2, :, :]
        # 对出界值的处理
        h = h % 1
        s = torch.clamp(s, 0, 1)
        v = torch.clamp(v, 0, 1)

        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)

        hi = torch.floor(h * 6)
        f = h * 6 - hi
        p = v * (1 - s)
        q = v * (1 - (f * s))
        t = v * (1 - ((1 - f) * s))

        hi0 = hi == 0
        hi1 = hi == 1
        hi2 = hi == 2
        hi3 = hi == 3
        hi4 = hi == 4
        hi5 = hi == 5

        r[hi0] = v[hi0]
        g[hi0] = t[hi0]
        b[hi0] = p[hi0]

        r[hi1] = q[hi1]
        g[hi1] = v[hi1]
        b[hi1] = p[hi1]

        r[hi2] = p[hi2]
        g[hi2] = v[hi2]
        b[hi2] = t[hi2]

        r[hi3] = p[hi3]
        g[hi3] = q[hi3]
        b[hi3] = v[hi3]

        r[hi4] = t[hi4]
        g[hi4] = p[hi4]
        b[hi4] = v[hi4]

        r[hi5] = v[hi5]
        g[hi5] = p[hi5]
        b[hi5] = q[hi5]

        r = r.unsqueeze(1)
        g = g.unsqueeze(1)
        b = b.unsqueeze(1)
        rgb = torch.cat([r, g, b], dim=1)
        return rgb


MAT_RGB2XYZ  = torch.Tensor([[0.412453, 0.357580, 0.180423],
                             [0.212671, 0.715160, 0.072169],
                             [0.019334, 0.119193, 0.950227]]).to(device)
MAT_XYZ2RGB = torch.Tensor([[ 3.2405, -1.5372, -0.4985],
                            [-0.9693,  1.8760,  0.0416],
                            [ 0.0556, -0.2040,  1.0573]]).to(device)

XYZ_REF_WHITE = torch.Tensor([0.95047, 1.0, 1.08883]).to(device)


def rgb2lab(rgb):
    rgb=torch.clamp(rgb,0,1)

    return xyz_to_lab(rgb_to_xyz(rgb))

def lab2rgb(lab):
    lab=torch.clamp(lab,0,1)
    return xyz_to_rgb(lab_to_xyz(lab))

def rgb_to_xyz(rgb):

    # convert dtype from uint8 to float
    # xyz = rgb.astype(np.float64) / 255.0
    # xyz = rgb.astype(np.float64)
    xyz = rgb
    # gamma correction
    mask = xyz > 0.04045
    abc=torch.zeros_like(xyz)
    abc[mask] = ((xyz[mask]  + 0.055) / 1.055)**2.4
    abc[~mask] = xyz[~mask]/12.92
    xyz = abc.permute(0, 2, 3, 1)
    # linear transform
    xyz = torch.matmul(xyz , MAT_RGB2XYZ.T)
    xyz = xyz.permute(0, 3, 1, 2)
    return xyz
def xyz_to_lab(xyz):


    xyz=xyz.permute(0, 2, 3, 1)
    xyz = xyz/XYZ_REF_WHITE

    # nonlinear transform
    mask = xyz > 0.008856
    xyz[mask] = torch.pow(xyz[mask], 1.0 / 3.0)
    xyz[~mask] = 7.787 * xyz[~mask] + 16.0 / 116.0
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]

    # linear transform
    lab = torch.zeros_like(xyz)
    # lab = torch.zeros(xyz.shape, requires_grad=True)
    lab[..., 0] = (116.0 * y) - 16.0  # L channel
    lab[..., 1] = 500.0 * (x - y)  # a channel
    lab[..., 2] = 200.0 * (y - z)  # b channel
    lab[..., 0] = lab[..., 0]/100  # L channel
    lab[..., 1] = (lab[..., 1]+86.183030)/184.416084  # a channel
    lab[..., 2] = (lab[..., 2]+107.857300)/202.335422 # b channel
    lab=lab.permute(0, 3, 1, 2)
    return lab

def lab_to_xyz(lab):


    lab=lab.permute(0, 2, 3, 1)
    l, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    l=l*100
    a=a*184.416084-86.183030
    b=b*202.335422-107.857300
    xyz = torch.zeros_like(lab)
    # xyz = torch.zeros(lab.shape,requires_grad=True)
    xyz[..., 1] = (l + 16.0) / 116.0
    xyz[..., 0] = a / 500.0 + xyz[..., 1]
    xyz[..., 2] = xyz[..., 1] - b / 200.0
    # index = xyz[..., 2] < 0
    # xyz[index, 2] = 0
    torch.clamp(xyz, min=0.0)


    # nonlinear transform
    mask = xyz > 0.2068966
    xyz[mask] = torch.pow(xyz[mask], 3.0)
    xyz[~mask] = (xyz[~mask] - 16.0 / 116.0) / 7.787

    # de-normalization
    xyz = xyz*XYZ_REF_WHITE
    xyz=xyz.permute(0, 3, 1, 2)
    return xyz

def xyz_to_rgb(xyz):


    rgb = xyz.permute(0, 2, 3, 1)
    rgb = torch.matmul(rgb, MAT_XYZ2RGB.T)

    # gamma correction
    mask = rgb > 0.0031308
    rgb[mask] = 1.055 * torch.pow(rgb[mask], 1.0 / 2.4) - 0.055
    rgb[~mask] = rgb[~mask] * 12.92

    # clip and convert dtype from float to uint8
    # rgb = np.round(255.0 * np.clip(rgb, 0, 1)).astype(np.uint8)
    rgb = torch.clip(rgb, 0, 1)
    rgb = rgb.permute(0, 3, 1, 2)
    return rgb
if __name__ == '__main__':
    with torch.autograd.set_detect_anomaly(True):
        rgb = torch.Tensor([[[0.8, 0.5, 0.5]]])
        rgb.requires_grad_()
        rgb = torch.unsqueeze(rgb.permute(2,0,1),0).to(device)

        xzy=rgb_to_xyz(rgb)

        lab=xyz_to_lab(xzy)

        xzy1=lab_to_xyz(lab)

        rgb1=xyz_to_rgb(xzy1)
        print(rgb1)
        rgb1 = rgb1.sum()
        rgb1.backward()


