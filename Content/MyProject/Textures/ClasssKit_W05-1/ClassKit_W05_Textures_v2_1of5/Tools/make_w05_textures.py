"""ClassKit_W05 v2 — Dirt / Gravel / Grass / Snow 텍스처 세트 생성.
사용: python make_w05_textures.py <출력 폴더> [dirt gravel grass snow]
출력: T_Kit_<Name>_BC (RGB=색, A=Height) / _N (DirectX) / _ORM (R=AO G=Roughness B=Metallic), 2048x2048 PNG"""
import sys, os
import numpy as np
from PIL import Image, ImageDraw
from texlib import *

PAL = np.array([  # 자갈 색 (sRGB), 가중치
    (0.56, 0.55, 0.53, 20), (0.43, 0.43, 0.44, 18), (0.33, 0.33, 0.35, 10),
    (0.63, 0.56, 0.46, 12), (0.47, 0.38, 0.30, 10), (0.53, 0.37, 0.30, 5),
    (0.40, 0.44, 0.48, 7), (0.78, 0.76, 0.71, 6), (0.21, 0.21, 0.23, 6),
    (0.58, 0.50, 0.38, 6)], np.float32)


def pebble_items(rng, n, count, rmin, rmax, z0, hscale, flat=(0.55, 1.0)):
    it = []
    for _ in range(count):
        r = rng.uniform(rmin, rmax)
        asp = rng.uniform(*flat)
        it.append((rng.uniform(0, n), rng.uniform(0, n), r, r * asp, rng.uniform(0, np.pi),
                   rng.uniform(*z0), r * hscale * rng.uniform(0.75, 1.15), rng.uniform(2.0, 3.6)))
    return it


def pebble_colors(rng, count):
    w = PAL[:, 3] / PAL[:, 3].sum()
    idx = rng.choice(len(PAL), count, p=w)
    col = PAL[idx, :3] * rng.uniform(0.82, 1.15, (count, 1)) + rng.normal(0, 0.012, (count, 3))
    rough = rng.uniform(0.5, 0.78, count)
    return np.clip(col, 0.03, 0.95).astype(np.float32), rough.astype(np.float32)


# ------------------------------------------------------------------ GRAVEL
def gravel(out):
    n = N
    rng = np.random.default_rng(51)
    edge = fbm(n, 1.6, 1, lo=12)
    ground = 0.10 + 0.07 * fbm(n, 2.0, 2) + 0.03 * fbm(n, 1.0, 3, lo=60)
    H = ground.copy()
    ID = np.full((n, n), -1, np.int32)
    core = np.zeros((n, n), bool)
    items = []
    for cnt, r0, r1, z in [(1100, 30, 46, (0.10, 0.22)), (9000, 17, 30, (0.06, 0.20)),
                           (40000, 9, 17, (0.03, 0.16)), (90000, 4.5, 9, (0.02, 0.12))]:
        cand = pebble_items(rng, n, cnt, r0, r1, z, 0.0165)
        items += stamp_pebbles(H, ID, cand, edge, core, id0=len(items))
    print("gravel pebbles:", len(items), "coverage:", (ID >= 0).mean())
    peb = ID >= 0
    col_p, rough_p = pebble_colors(rng, len(items))
    # 표면 미세 요철(자갈 위만), 살짝 패인 자국
    fine = fbm(n, 1.2, 4, lo=40)
    pits = smoothstep(0.86, 0.97, fbm(n, 1.0, 5, lo=60))
    H = H + peb * (0.016 * (fine - 0.5) - 0.02 * pits) + (~peb) * 0.03 * (fbm(n, 0.6, 6, lo=120) - 0.5)
    H = norm01(H, 0.05).astype(np.float32)
    hpx = 62.0
    ao = ao_from_height(H, hpx)
    # 색
    gcol = np.array([0.36, 0.29, 0.22]) * (0.75 + 0.5 * fbm(n, 1.8, 7))[..., None]
    bc = np.where(peb[..., None], col_p[np.clip(ID, 0, None)], gcol)
    speck = fbm(n, 0.9, 8, lo=50)
    mott = fbm(n, 1.8, 9, lo=6)
    bc = bc * (0.86 + 0.28 * speck)[..., None] * (0.9 + 0.2 * mott)[..., None]
    # 틈의 흙먼지: 낮고 가려진 곳으로 갈수록 흙색
    dust = np.clip((1 - ao) * 1.5 + smoothstep(0.45, 0.15, H) * 0.5, 0, 1) * peb
    bc = bc * (1 - 0.55 * dust[..., None]) + np.array([0.40, 0.32, 0.24]) * 0.55 * dust[..., None]
    bc = bc * (0.62 + 0.38 * ao)[..., None]
    rough = np.where(peb, rough_p[np.clip(ID, 0, None)], 0.92) + 0.10 * (speck - 0.5) + 0.2 * dust
    nrm = normal_dx(H, hpx)
    save_rgba(f"{out}/T_Kit_Gravel_BC.png", bc, H)
    save_rgb(f"{out}/T_Kit_Gravel_N.png", nrm)
    save_rgb(f"{out}/T_Kit_Gravel_ORM.png", np.dstack([ao, np.clip(rough, 0.3, 1), np.zeros_like(ao)]))


# ------------------------------------------------------------------ DIRT
def dirt(out):
    n = N
    rng = np.random.default_rng(52)
    mound = fbm(n, 2.3, 11)
    mid = fbm(n, 1.7, 12, lo=6)
    f1, f2, _ = worley(n, 5200, 13)
    clod = np.clip(1 - f1 / 22.0, 0, 1) ** 0.7
    clodmask = smoothstep(0.35, 0.6, fbm(n, 2.0, 14, lo=3))
    wy = (fbm(n, 2.0, 25, lo=3) - 0.5) * 150 + (fbm(n, 1.4, 26, lo=20) - 0.5) * 26
    wx = (fbm(n, 2.0, 27, lo=3) - 0.5) * 150 + (fbm(n, 1.4, 28, lo=20) - 0.5) * 26
    c1, c2, _ = worley(n, 300, 15, warp=(wy, wx))
    warp = fbm(n, 1.5, 16, lo=10)
    crack_w = 1.6 + 5.5 * warp ** 2
    crack = 1 - smoothstep(0.0, 1.0, (c2 - c1) / crack_w)          # 1 = 균열 중심
    drymask = smoothstep(0.42, 0.62, fbm(n, 2.2, 17, lo=2))        # 마른 구역에만 균열
    crack *= drymask
    H = 0.30 * mound + 0.22 * mid + 0.20 * clod * clodmask + 0.05 * (fbm(n, 0.8, 18, lo=70) - 0.5)
    H = H * (1 - 0.0) - 0.26 * crack
    H = norm01(H, 0.2).astype(np.float32) * 0.62 + 0.05
    ID = np.full((n, n), -1, np.int32)
    core = np.zeros((n, n), bool)
    edge = fbm(n, 1.6, 19, lo=12)
    items = []
    for cnt, r0, r1 in [(26, 20, 34), (260, 9, 18), (1500, 3.5, 8)]:
        cand = pebble_items(rng, n, cnt, r0, r1, (0.0, 0.0), 0.0125)
        cand = [(cx, cy, a, b, th, float(H[int(cy) % n, int(cx) % n]) - 0.05, hh, k)
                for (cx, cy, a, b, th, z0, hh, k) in cand]
        items += stamp_pebbles(H, ID, cand, edge, core, id0=len(items))
    peb = ID >= 0
    col_p, rough_p = pebble_colors(rng, len(items))
    H = norm01(H, 0.02).astype(np.float32)
    hpx = 34.0
    ao = ao_from_height(H, hpx, scales=(2.5, 8, 24))
    # 색: 젖은 어두운 흙 ↔ 마른 밝은 흙
    wet = np.array([0.17, 0.115, 0.075])
    dry = np.array([0.46, 0.35, 0.24])
    red = np.array([0.36, 0.22, 0.13])
    t = np.clip(0.55 * drymask + 0.35 * blur(H, 6) + 0.25 * (fbm(n, 1.9, 20, lo=3) - 0.5) + 0.1, 0, 1)
    bc = wet + (dry - wet) * t[..., None]
    tint = smoothstep(0.55, 0.8, fbm(n, 2.1, 21, lo=2))[..., None]
    bc = bc * (1 - 0.35 * tint) + red * 0.35 * tint
    grain = fbm(n, 0.7, 22, lo=80)
    bc = bc * (0.80 + 0.40 * grain)[..., None] * (0.9 + 0.2 * mid)[..., None]
    bc = bc * (1 - 0.6 * crack[..., None])
    # 잔 부스러기(밝은 모래알·어두운 유기물)
    sand = 0.6 * smoothstep(0.90, 0.96, fbm(n, 0.5, 23, lo=110))[..., None]
    org = smoothstep(0.88, 0.95, fbm(n, 0.45, 24, lo=110))[..., None]
    bc = bc * (1 - sand) + np.array([0.62, 0.55, 0.44]) * sand
    bc = bc * (1 - 0.7 * org)
    bc = np.where(peb[..., None], col_p[np.clip(ID, 0, None)] * (0.85 + 0.3 * grain)[..., None], bc)
    bc = bc * (0.60 + 0.40 * ao)[..., None]
    rough = 0.93 - 0.22 * (1 - t) + 0.08 * (grain - 0.5)
    rough = np.where(peb, rough_p[np.clip(ID, 0, None)], rough)
    nrm = normal_dx(H, hpx)
    save_rgba(f"{out}/T_Kit_Dirt_BC.png", bc, H)
    save_rgb(f"{out}/T_Kit_Dirt_N.png", nrm)
    save_rgb(f"{out}/T_Kit_Dirt_ORM.png", np.dstack([ao, np.clip(rough, 0.3, 1), np.zeros_like(ao)]))


# ------------------------------------------------------------------ GRASS
def grass(out):
    n = N
    S = 2                      # 슈퍼샘플 배율
    W = n * S
    rng = np.random.default_rng(53)
    dens = fbm(n, 2.0, 31, lo=2)          # 포기 높이·밀도의 큰 변화
    img = Image.new("RGB", (W, W), (0, 0, 0))
    hgt = Image.new("L", (W, W), 0)
    cov = Image.new("L", (W, W), 0)
    dc, dh, dv = ImageDraw.Draw(img), ImageDraw.Draw(hgt), ImageDraw.Draw(cov)
    ntuft = 3800
    tc = rng.uniform(0, W, (ntuft, 2))
    blades = []
    lean = rng.uniform(0, 2 * np.pi)
    for (tx, ty) in tc:
        d = dens[int(ty / S) % n, int(tx / S) % n]
        th = 0.42 + 0.58 * np.clip(d * 1.25 - 0.1, 0, 1) * rng.uniform(0.8, 1.0)   # 포기 높이
        nb = int(rng.uniform(13, 24) * (0.6 + 0.7 * d))
        hue = rng.normal(0, 1)
        for _ in range(nb):
            ang = rng.uniform(0, 2 * np.pi)
            ang = ang + 0.55 * np.sin(lean - ang)       # 한쪽으로 살짝 눕는다
            L = rng.uniform(50, 125) * (0.65 + 0.6 * th)
            bx = tx + rng.normal(0, 9)
            by = ty + rng.normal(0, 9)
            bend = rng.normal(0, 0.22) * L
            w0 = rng.uniform(8.0, 13.0)
            tip_h = th * rng.uniform(0.7, 1.0)
            dryb = rng.random() < 0.09
            blades.append((tip_h, bx, by, ang, L, bend, w0, hue + rng.normal(0, 0.5), dryb))
    blades.sort(key=lambda b: b[0])
    print("grass blades:", len(blades))
    K = 5
    base = np.array([0.075, 0.13, 0.035])
    midc = np.array([0.20, 0.34, 0.085])
    tipc = np.array([0.40, 0.53, 0.17])
    dry_b = np.array([0.30, 0.25, 0.11])
    dry_t = np.array([0.62, 0.54, 0.30])
    for (tip_h, bx, by, ang, L, bend, w0, hue, dryb) in blades:
        dx, dy = np.cos(ang), np.sin(ang)
        px, py = -dy, dx
        ts = np.linspace(0, 1, K + 1)
        cx = bx + dx * L * ts + px * bend * ts ** 2
        cy = by + dy * L * ts + py * bend * ts ** 2
        ws = w0 * (1 - ts) ** 0.65 * 0.5 + 0.45
        lx, ly = cx + px * ws, cy + py * ws
        rx, ry = cx - px * ws, cy - py * ws
        shifts = [(0, 0)]
        x0, x1, y0, y1 = cx.min() - 10, cx.max() + 10, cy.min() - 10, cy.max() + 10
        sx = [0] + ([W] if x0 < 0 else []) + ([-W] if x1 >= W else [])
        sy = [0] + ([W] if y0 < 0 else []) + ([-W] if y1 >= W else [])
        shifts = [(a, b) for a in sx for b in sy]
        bright = 1 + 0.10 * hue
        for i in range(K):
            t = (ts[i] + ts[i + 1]) / 2
            if dryb:
                c = dry_b + (dry_t - dry_b) * t
            else:
                c = base + (midc - base) * min(t * 2, 1) + (tipc - midc) * max(t * 2 - 1, 0)
                c = c * np.array([1 + 0.10 * hue, 1.0, 1 - 0.12 * hue])
            c = tuple(int(v) for v in np.clip(c * bright, 0, 1) * 255)
            hv = int(255 * (0.16 + (tip_h - 0.16) * t ** 0.8))
            for (ox, oy) in shifts:
                poly = [(lx[i] + ox, ly[i] + oy), (lx[i + 1] + ox, ly[i + 1] + oy),
                        (rx[i + 1] + ox, ry[i + 1] + oy), (rx[i] + ox, ry[i] + oy)]
                dc.polygon(poly, fill=c)
                dh.polygon(poly, fill=hv)
                dv.polygon(poly, fill=255)
    col = np.asarray(img.resize((n, n), Image.LANCZOS), np.float32) / 255
    Hb = np.asarray(hgt.resize((n, n), Image.LANCZOS), np.float32) / 255
    cv = np.asarray(cov.resize((n, n), Image.LANCZOS), np.float32) / 255
    print("grass coverage:", cv.mean())
    # 바닥 흙(풀 사이 틈)
    sg = fbm(n, 0.8, 32, lo=60)
    soil = np.array([0.15, 0.10, 0.055]) * (0.7 + 0.6 * sg)[..., None]
    soilh = 0.05 + 0.06 * fbm(n, 1.6, 33, lo=8)
    bc = soil * (1 - cv[..., None]) + col
    Hraw = np.maximum(Hb, soilh * (1 - cv))
    tuft = norm01(blur(Hraw, 14))
    H = norm01(0.72 * Hraw + 0.28 * tuft * Hraw.max(), 0.05).astype(np.float32)
    hpx = 30.0
    ao = ao_from_height(H, hpx, scales=(2, 6, 18), k=(0.9, 0.9, 0.7))
    bc = bc * (0.50 + 0.50 * ao)[..., None]
    bc = bc * (0.92 + 0.16 * fbm(n, 2.0, 34, lo=2))[..., None]
    rough = 0.90 - 0.28 * cv * smoothstep(0.2, 0.9, H) + 0.06 * (sg - 0.5)
    nrm = normal_dx(H, hpx, pre=0.8)
    save_rgba(f"{out}/T_Kit_Grass_BC.png", bc, H)
    save_rgb(f"{out}/T_Kit_Grass_N.png", nrm)
    save_rgb(f"{out}/T_Kit_Grass_ORM.png", np.dstack([ao, np.clip(rough, 0.3, 1), np.zeros_like(ao)]))


# ------------------------------------------------------------------ SNOW
def snow(out):
    n = N
    drift = fbm(n, 2.7, 41, hi=14)
    mid = fbm(n, 2.0, 42, lo=5, hi=60)
    yy, xx = np.mgrid[0:n, 0:n] / n
    w1 = fbm(n, 2.2, 43, hi=10)
    w2 = fbm(n, 2.2, 44, hi=10)
    rip = np.sin(2 * np.pi * (11 * xx + 4 * yy) + 5.0 * w1) * 0.5 + 0.5
    rip2 = np.sin(2 * np.pi * (23 * xx - 7 * yy) + 4.0 * w2) * 0.5 + 0.5
    ripmask = smoothstep(0.35, 0.7, fbm(n, 2.3, 45, hi=8))
    f1, f2, _ = worley(n, 900, 46)
    clump = np.clip(1 - f1 / 34.0, 0, 1) ** 1.5 * smoothstep(0.5, 0.75, fbm(n, 2.0, 47, lo=2, hi=20))
    grain = fbm(n, 0.4, 48, lo=160)
    H = 0.52 * drift + 0.16 * mid + 0.13 * rip * ripmask + 0.05 * rip2 * ripmask + 0.14 * clump + 0.025 * (grain - 0.5)
    H = norm01(H, 0.05).astype(np.float32)
    hpx = 44.0
    ao = ao_from_height(H, hpx, scales=(4, 14, 40), k=(0.5, 0.7, 0.9))
    lit = np.array([0.93, 0.95, 0.97])
    shd = np.array([0.66, 0.76, 0.90])
    t = np.clip(0.55 * ao + 0.55 * blur(H, 3) - 0.08, 0, 1)
    bc = shd + (lit - shd) * t[..., None]
    bc = bc * (0.965 + 0.07 * grain)[..., None]
    spark = smoothstep(0.93, 0.97, fbm(n, 0.2, 49, lo=300))[..., None]
    bc = np.clip(bc + 0.06 * spark, 0, 1)
    nrm = normal_dx(H + 0.004 * (grain - 0.5), hpx, pre=0.5)
    save_rgba(f"{out}/T_Kit_Snow_BC.png", bc, H)
    save_rgb(f"{out}/T_Kit_Snow_N.png", nrm)


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    which = sys.argv[2:] or ["dirt", "gravel", "grass", "snow"]
    for w in which:
        print("==", w)
        globals()[w](out)
