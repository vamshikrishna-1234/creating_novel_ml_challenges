"""
Generator for Synthetic Visual Feature Imputation Under Hidden Relational Constraints.

Creates 10,000 puzzle images. Each shows a 3x4 grid of geometric
compositions with one cell blank (random position). The task is to
predict 5 visual properties of the blank cell from the surrounding
context of 11 visible cells.

Properties: shape(6), color(6), size(3), count(3), fill(2).
Constraints: 9 within-attribute spatial constraint types + conditional
cross-attribute dependencies. 15% label noise on answers.
"""

from pathlib import Path
import math
import numpy as np
from PIL import Image, ImageDraw

SEED = 42
ROWS, COLS = 3, 4

SHAPE_N = 6
COLOR_N = 6
SIZE_N = 3
COUNT_N = 3
FILL_N = 2
PROP_RANGES = [SHAPE_N, COLOR_N, SIZE_N, COUNT_N, FILL_N]
N_PROPS = 5

PALETTE = [
    (215, 48, 39),
    (69, 117, 180),
    (49, 163, 84),
    (254, 178, 76),
    (123, 50, 148),
    (227, 119, 194),
]
OUTLINES = [tuple(max(c - 55, 0) for c in rgb) for rgb in PALETTE]
BASE_R = [11, 17, 23]

CELL = 86
GAP = 4
MARGIN = 10
IMG_BG = (242, 242, 244)
CELL_BG = (255, 255, 255)
BLANK_BG = (210, 210, 215)
BORDER_C = (110, 110, 120)


# ── Shape drawing ────────────────────────────────────────────────

def _verts(cx, cy, r, n, off_deg=0):
    return [
        (cx + r * math.cos(math.radians(off_deg + 360 * i / n)),
         cy + r * math.sin(math.radians(off_deg + 360 * i / n)))
        for i in range(n)
    ]


def _draw_one(draw, cx, cy, shape, ci, r, fi):
    fc = PALETTE[ci] if fi == 0 else None
    oc = OUTLINES[ci] if fi == 0 else PALETTE[ci]
    w = 2 if fi == 0 else max(2, int(r * 0.16))
    if shape == 0:
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fc, outline=oc, width=w)
    elif shape == 1:
        draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=fc, outline=oc, width=w)
    else:
        sides, off = {2: (3, -90), 3: (5, -90), 4: (6, 0), 5: (4, -90)}[shape]
        v = _verts(cx, cy, r, sides, off)
        if fi == 0:
            draw.polygon(v, fill=fc, outline=oc)
        else:
            draw.line(v + [v[0]], fill=oc, width=w)


def _draw_cell(draw, x, y, sz, props):
    si, ci, zi, ni, fi = int(props[0]), int(props[1]), int(props[2]), int(props[3]), int(props[4])
    cnt = ni + 1
    mid = sz / 2.0
    br = BASE_R[zi] * sz / 86.0

    if cnt == 1:
        _draw_one(draw, x + mid, y + mid, si, ci, br, fi)
    elif cnt == 2:
        off = sz * 0.22
        r = br * 0.72
        _draw_one(draw, x + mid - off, y + mid, si, ci, r, fi)
        _draw_one(draw, x + mid + off, y + mid, si, ci, r, fi)
    else:
        off = sz * 0.21
        r = br * 0.56
        _draw_one(draw, x + mid, y + mid - off, si, ci, r, fi)
        _draw_one(draw, x + mid - off, y + mid + off * 0.7, si, ci, r, fi)
        _draw_one(draw, x + mid + off, y + mid + off * 0.7, si, ci, r, fi)


# ── Constraint generators (return 3×4 int grids) ────────────────

def _prog_r(rng, R):
    s = rng.randint(1, R)
    g = np.zeros((ROWS, COLS), int)
    for r in range(ROWS):
        b = rng.randint(0, R)
        g[r] = [(b + c * s) % R for c in range(COLS)]
    return g

def _prog_c(rng, R):
    s = rng.randint(1, R)
    g = np.zeros((ROWS, COLS), int)
    for c in range(COLS):
        b = rng.randint(0, R)
        g[:, c] = [(b + r * s) % R for r in range(ROWS)]
    return g

def _const_r(rng, R):
    g = np.zeros((ROWS, COLS), int)
    for r in range(ROWS):
        g[r, :] = rng.randint(0, R)
    return g

def _const_c(rng, R):
    g = np.zeros((ROWS, COLS), int)
    for c in range(COLS):
        g[:, c] = rng.randint(0, R)
    return g

def _dist_r(rng, R):
    if R >= COLS:
        vals = sorted(rng.choice(R, COLS, replace=False))
    else:
        vals = list(range(R))
        while len(vals) < COLS:
            vals.append(int(rng.randint(0, R)))
        vals = sorted(vals)
    g = np.zeros((ROWS, COLS), int)
    for r in range(ROWS):
        g[r] = rng.permutation(vals)
    return g

def _dist_c(rng, R):
    if R >= ROWS:
        vals = sorted(rng.choice(R, ROWS, replace=False))
    else:
        vals = list(range(R))
        while len(vals) < ROWS:
            vals.append(int(rng.randint(0, R)))
        vals = sorted(vals)
    g = np.zeros((ROWS, COLS), int)
    for c in range(COLS):
        g[:, c] = rng.permutation(vals)
    return g

def _xor(rng, R):
    if R < 2:
        return _const_r(rng, R)
    v = rng.choice(R, 2, replace=False)
    g = np.zeros((ROWS, COLS), int)
    for r in range(ROWS):
        for c in range(COLS):
            g[r, c] = v[0] if (r + c) % 2 == 0 else v[1]
    return g

def _additive(rng, R):
    rv = [rng.randint(0, R) for _ in range(ROWS)]
    cv = [rng.randint(0, R) for _ in range(COLS)]
    g = np.zeros((ROWS, COLS), int)
    for r in range(ROWS):
        for c in range(COLS):
            g[r, c] = (rv[r] + cv[c]) % R
    return g

def _mirror_h(rng, R):
    """Left half mirrors right half (columns 0,1 mirror 3,2)."""
    g = np.zeros((ROWS, COLS), int)
    for r in range(ROWS):
        v0 = rng.randint(0, R)
        v1 = rng.randint(0, R)
        g[r, 0] = v0
        g[r, 1] = v1
        g[r, 2] = v1
        g[r, 3] = v0
    return g

def _conditional(rng, R, dep_grid):
    """Attribute value is a deterministic function of another attribute."""
    dep_vals = np.unique(dep_grid)
    mapping = {int(v): int(rng.randint(0, R)) for v in dep_vals}
    g = np.zeros((ROWS, COLS), int)
    for r in range(ROWS):
        for c in range(COLS):
            g[r, c] = mapping.get(int(dep_grid[r, c]), 0)
    return g

SPATIAL_RULES = [_prog_r, _prog_c, _const_r, _const_c,
                 _dist_r, _dist_c, _xor, _additive, _mirror_h]


# ── Grid generation ──────────────────────────────────────────────

def make_grid(rng, n_rules, use_conditional):
    rule_props = sorted(rng.choice(N_PROPS, n_rules, replace=False))

    grids = [None] * N_PROPS
    cond_prop = None
    if use_conditional and len(rule_props) >= 2:
        cond_prop = rule_props[-1]

    for p in rule_props:
        if p == cond_prop:
            continue
        fn = SPATIAL_RULES[rng.randint(0, len(SPATIAL_RULES))]
        grids[p] = fn(rng, PROP_RANGES[p])

    if cond_prop is not None:
        deps = [pp for pp in rule_props if pp != cond_prop and grids[pp] is not None]
        if deps:
            dep_p = deps[rng.randint(0, len(deps))]
            grids[cond_prop] = _conditional(rng, PROP_RANGES[cond_prop], grids[dep_p])
        else:
            fn = SPATIAL_RULES[rng.randint(0, len(SPATIAL_RULES))]
            grids[cond_prop] = fn(rng, PROP_RANGES[cond_prop])

    for p in range(N_PROPS):
        if grids[p] is not None:
            continue
        R = PROP_RANGES[p]
        if rng.random() < 0.40:
            fn = SPATIAL_RULES[rng.randint(0, len(SPATIAL_RULES))]
            tmp = fn(rng, R)
            br, bc = rng.randint(0, ROWS), rng.randint(0, COLS)
            orig = int(tmp[br, bc])
            alt = rng.randint(0, R)
            while alt == orig and R > 1:
                alt = rng.randint(0, R)
            tmp[br, bc] = alt
            grids[p] = tmp
        else:
            grids[p] = np.full((ROWS, COLS), rng.randint(0, R), int)

    cells = np.stack(grids, axis=-1)
    return cells


# ── Image rendering ──────────────────────────────────────────────

def render(cells, blank_r, blank_c):
    gw = COLS * CELL + (COLS - 1) * GAP
    gh = ROWS * CELL + (ROWS - 1) * GAP
    w = gw + 2 * MARGIN
    h = gh + 2 * MARGIN

    img = Image.new("RGB", (w, h), IMG_BG)
    dr = ImageDraw.Draw(img)
    gx = MARGIN
    gy = MARGIN

    for r in range(ROWS):
        for c in range(COLS):
            x = gx + c * (CELL + GAP)
            y = gy + r * (CELL + GAP)
            if r == blank_r and c == blank_c:
                dr.rectangle([x, y, x + CELL, y + CELL],
                             fill=BLANK_BG, outline=BORDER_C, width=2)
                dr.text((x + CELL // 2 - 4, y + CELL // 2 - 6), "?",
                        fill=(80, 80, 85))
            else:
                dr.rectangle([x, y, x + CELL, y + CELL],
                             fill=CELL_BG, outline=BORDER_C, width=1)
                _draw_cell(dr, x, y, CELL, cells[r, c])

    return img


# ── Main ─────────────────────────────────────────────────────────

def generate(out_dir="raw_data", seed=SEED):
    rng = np.random.RandomState(seed)
    N = 10_000
    NOISE = 0.15

    out = Path(out_dir)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx in range(N):
        p = rng.random()
        n_rules = 1 if p < 0.05 else (2 if p < 0.25 else 3)
        use_cond = n_rules >= 3 and rng.random() < 0.35

        cells = make_grid(rng, n_rules, use_cond)

        blank_r = rng.randint(0, ROWS)
        blank_c = rng.randint(0, COLS)

        answer = cells[blank_r, blank_c].tolist()

        noisy = 0
        if rng.random() < NOISE:
            n_perturb = rng.randint(1, 3)
            perturb_props = rng.choice(N_PROPS, n_perturb, replace=False)
            for pp in perturb_props:
                answer[pp] = int(rng.randint(0, PROP_RANGES[pp]))
            noisy = 1

        img = render(cells, blank_r, blank_c)
        img.save(str(img_dir / f"{idx:05d}.png"), "PNG")

        row = [idx, blank_r, blank_c] + answer + [n_rules, int(use_cond), noisy]
        rows.append(",".join(str(v) for v in row))
        if (idx + 1) % 1000 == 0:
            print(f"  {idx + 1}/{N}")

    header = "image_id,blank_row,blank_col,prop_0,prop_1,prop_2,prop_3,prop_4,difficulty,has_conditional,is_noisy"
    with open(str(out / "labels.csv"), "w") as f:
        f.write(header + "\n")
        f.write("\n".join(rows) + "\n")

    print(f"Done: {N} images saved to {img_dir}")


if __name__ == "__main__":
    generate()
