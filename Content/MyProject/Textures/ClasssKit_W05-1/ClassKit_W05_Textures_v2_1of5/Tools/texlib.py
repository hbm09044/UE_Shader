"""ClassKit_W05 텍스처 생성 공용 함수.
모든 결과는 가로·세로로 이음매 없이 반복(tileable)된다.
노멀은 DirectX 규약(G 채널: 아래를 보는 면이 밝다)."""
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from PIL import Image

N = 2048


def fbm(n, beta, seed, lo=1, hi=None):
    """주기 프랙탈 노이즈(1/f^beta 스펙트럼). 반환 0~1."""
    rng = np.random.default_rng(seed)
    fx = np.fft.fftfreq(n) * n
    fy = np.fft.rfftfreq(n) * n
    f = np.sqrt(fx[:, None] ** 2 + fy[None, :] ** 2)
    m = f >= lo
    if hi is not None:
        m &= f <= hi
    amp = np.zeros_like(f)
    amp[m] = f[m] ** (-beta)
    ph = rng.uniform(0, 2 * np.pi, f.shape)
    x = np.fft.irfft2(amp * np.exp(1j * ph), s=(n, n))
    return norm01(x).astype(np.float32)


def norm01(x, p=0.5):
    a, b = np.percentile(x, [p, 100 - p])
    return np.clip((x - a) / (b - a + 1e-9), 0, 1)


def blur(x, s):
    return ndi.gaussian_filter(x, s, mode="wrap")


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def worley(n, npts, seed, k=2, warp=None):
    """주기 Worley. F1, F2(픽셀 단위), 가장 가까운 점 번호를 돌려준다."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0, n, (npts, 2))
    tree = cKDTree(pts, boxsize=n)
    yy, xx = np.mgrid[0:n, 0:n]
    if warp is not None:
        yy = yy + warp[0]
        xx = xx + warp[1]
    q = np.stack([yy.ravel() + 0.5, xx.ravel() + 0.5], 1) % n
    d, i = tree.query(q, k=k, workers=-1)
    return (d[:, 0].reshape(n, n).astype(np.float32),
            d[:, 1].reshape(n, n).astype(np.float32),
            i[:, 0].reshape(n, n))


def normal_dx(h, hpx, pre=0.6):
    """하이트(0~1) → DirectX 탄젠트 노멀(0~1 RGB). hpx = 하이트 1이 몇 픽셀 높이인지."""
    if pre:
        h = blur(h, pre)
    dx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) * 0.5 * hpx
    dy = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) * 0.5 * hpx
    nx, ny, nz = -dx, -dy, np.ones_like(h)
    l = np.sqrt(nx * nx + ny * ny + nz * nz)
    return np.stack([nx / l, ny / l, nz / l], -1) * 0.5 + 0.5


def ao_from_height(h, hpx, scales=(3, 10, 28), k=(1.0, 0.8, 0.6)):
    """주변보다 낮은 곳을 어둡게. 반환 0~1."""
    occ = np.zeros_like(h)
    for s, w in zip(scales, k):
        occ += w * np.clip(blur(h, s) - h, 0, None) * hpx / s
    return np.clip(1 - occ * 1.6, 0, 1)


def stamp_pebbles(H, ID, items, edge_noise, core=None, id0=0):
    """자갈(반타원체)을 하이트 필드에 max 합성한다. items: (cx,cy,a,b,th,z0,hh,k)."""
    n = H.shape[0]
    placed = []
    for it in items:
        cx, cy, a, b, th, z0, hh, k = it
        if core is not None and core[int(cy) % n, int(cx) % n]:
            continue
        R = int(max(a, b) * 1.25) + 2
        ys = np.arange(int(cy) - R, int(cy) + R + 1)
        xs = np.arange(int(cx) - R, int(cx) + R + 1)
        dy = (ys - cy)[:, None]
        dx = (xs - cx)[None, :]
        c, s = np.cos(th), np.sin(th)
        u = (dx * c + dy * s) / a
        v = (-dx * s + dy * c) / b
        idx = np.ix_(ys % n, xs % n)
        d2 = (u * u + v * v) * (1 + 0.45 * (edge_noise[idx] - 0.5))
        m = d2 < 1
        prof = np.zeros_like(d2)
        prof[m] = (1 - d2[m] ** (k / 2)) ** 0.5
        hloc = z0 + hh * prof
        sub = H[idx]
        upd = m & (hloc > sub)
        sub[upd] = hloc[upd]
        H[idx] = sub
        pid = id0 + len(placed)
        sid = ID[idx]
        sid[upd] = pid
        ID[idx] = sid
        if core is not None:
            cs = core[idx]
            cs[d2 < 0.80] = True
            core[idx] = cs
        placed.append(it)
    return placed


def to8(x):
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)


def save_rgba(path, rgb, a):
    Image.fromarray(np.dstack([to8(rgb), to8(a)]), "RGBA").save(path, optimize=True)


def save_rgb(path, rgb):
    Image.fromarray(to8(rgb), "RGB").save(path, optimize=True)


def srgb2lin(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lin2srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def shade(bc, nrm, ao, rough, L=(-0.55, -0.55, 0.62)):
    """검증용 간이 조명(위에서 내려다본 뷰). bc는 sRGB 0~1."""
    n = nrm * 2 - 1
    n = n / np.linalg.norm(n, axis=-1, keepdims=True)
    L = np.array(L, np.float32)
    L /= np.linalg.norm(L)
    # DirectX 노멀: G가 화면 아래쪽(+row) → 화면 좌표계 y로 그대로 쓴다
    ndl = np.clip(n[..., 0] * L[0] + n[..., 1] * L[1] + n[..., 2] * L[2], 0, 1)
    Hh = (L + np.array([0, 0, 1.0])) / np.linalg.norm(L + np.array([0, 0, 1.0]))
    ndh = np.clip(n[..., 0] * Hh[0] + n[..., 1] * Hh[1] + n[..., 2] * Hh[2], 0, 1)
    p = 2 / np.clip(rough, 0.08, 1) ** 4
    spec = 0.04 * (p + 2) / 8 * ndh ** p * ndl
    lin = srgb2lin(bc)
    out = lin * (ndl[..., None] * 2.6 + 0.35 * ao[..., None] * np.array([0.75, 0.85, 1.0])) + spec[..., None] * 2.6
    out = out / (1 + out * 0.35)
    return lin2srgb(out)
