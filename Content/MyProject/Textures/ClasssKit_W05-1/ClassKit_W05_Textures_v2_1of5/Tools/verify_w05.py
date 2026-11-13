"""검증: 타일 이음매, 하이트 분포, 조명 미리보기, Lerp vs HeightLerp 비교 렌더.
사용: python verify_w05.py <텍스처 폴더> <미리보기 폴더>"""
import sys, os
import numpy as np
from PIL import Image
from texlib import *

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)


def load(name):
    return np.asarray(Image.open(f"{src}/{name}.png"), np.float32) / 255


def layer(nm, orm=True):
    bc = load(f"T_Kit_{nm}_BC")
    n = load(f"T_Kit_{nm}_N")
    o = load(f"T_Kit_{nm}_ORM") if orm else np.dstack([np.ones(bc.shape[:2]), np.full(bc.shape[:2], 0.75), np.zeros(bc.shape[:2])]).astype(np.float32)
    return bc[..., :3], bc[..., 3], n, o


L = {k: layer(k, k != "Snow") for k in ["Dirt", "Gravel", "Grass", "Snow"]}

# 1) 이음매: 경계 열/행의 차이 vs 내부 인접 차이
for k, (bc, h, n, o) in L.items():
    for nm, a in [("BC", bc), ("H", h), ("N", n)]:
        edge = np.abs(a[:, 0] - a[:, -1]).mean() + np.abs(a[0] - a[-1]).mean()
        inner = np.abs(a[:, 1] - a[:, 0]).mean() + np.abs(a[1] - a[0]).mean()
        print(f"{k:7s} {nm:2s} seam {edge:.4f} / inner {inner:.4f}  ratio {edge / inner:.2f}")
    hist = np.percentile(h, [5, 25, 50, 75, 95])
    nz = (n[..., 2] * 2 - 1)
    print(f"{k:7s} height p5/25/50/75/95 = {np.round(hist, 2)}  normal tilt mean {np.degrees(np.arccos(np.clip(nz, -1, 1))).mean():.1f} deg"
          f"  BC mean {np.round(bc.mean((0, 1)), 2)} std {bc.std():.3f}")

# 2) 단독 미리보기 (2x2 타일, 조명)
for k, (bc, h, n, o) in L.items():
    c = slice(0, 1024)
    im = shade(bc[c, c], n[c, c], o[c, c, 0], o[c, c, 1])
    Image.fromarray(to8(im)).save(f"{dst}/preview_{k}.png")
    Image.fromarray(to8(np.tile(shade(bc[::4, ::4], n[::4, ::4], o[::4, ::4, 0], o[::4, ::4, 1]), (2, 2, 1)))).save(f"{dst}/tile2x2_{k}.png")
    Image.fromarray(to8(h[::2, ::2])).save(f"{dst}/height_{k}.png")


# 3) Lerp vs HeightLerp — 붓으로 칠한 것 같은 부드러운 마스크
def heightlerp_alpha(phase, height, contrast):
    a = np.clip(height - 1 + 2 * phase, 0, 1)
    return np.clip(-contrast + (1 + 2 * contrast) * a, 0, 1) if contrast > 0 else a


S = 1536
yy, xx = np.mgrid[0:S, 0:S] / S
wob = fbm(S, 2.5, 99)[..., None][..., 0]
mR = smoothstep(0.18, 0.62, xx + 0.10 * (wob - 0.5))                  # 자갈 마스크: 왼→오
mG = smoothstep(0.30, 0.80, 1 - np.hypot(xx - 0.78, yy - 0.5) * 2.0)  # 풀 마스크: 오른쪽 원


def crop(a):
    return np.tile(a, (1, 1) + (1,) * (a.ndim - 2))[:S, :S]


def blend(mode):
    dbc, dh, dn, do = [crop(a) for a in L["Dirt"]]
    gbc, gh, gn, go = [crop(a) for a in L["Gravel"]]
    rbc, rh, rn, ro = [crop(a) for a in L["Grass"]]
    if mode == "lerp":
        a1, a2 = mR, mG
    else:
        a1 = heightlerp_alpha(mR, gh, 4 / 4)
        a2 = heightlerp_alpha(mG, rh, 3 / 4)
    mix = lambda A, B, a: A + (B - A) * (a[..., None] if A.ndim == 3 else a)
    bc = mix(mix(dbc, gbc, a1), rbc, a2)
    n = mix(mix(dn, gn, a1), rn, a2)
    o = mix(mix(do, go, a1), ro, a2)
    return shade(bc, n, o[..., 0], o[..., 1])


A = blend("lerp")
B = blend("height")
gap = np.ones((S, 24, 3))
Image.fromarray(to8(np.hstack([A, gap, B]))).save(f"{dst}/compare_Lerp_vs_HeightLerp.png")
z = slice(500, 1012)
zx = slice(300, 812)
Image.fromarray(to8(np.hstack([A[z, zx], gap[:512], B[z, zx]]))).resize((2096, 1024), Image.LANCZOS).save(f"{dst}/compare_zoom_gravel.png")
zx = slice(900, 1412)
Image.fromarray(to8(np.hstack([A[z, zx], gap[:512], B[z, zx]]))).resize((2096, 1024), Image.LANCZOS).save(f"{dst}/compare_zoom_grass.png")
print("done")
