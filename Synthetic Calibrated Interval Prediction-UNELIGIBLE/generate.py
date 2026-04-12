"""
Data generator for Synthetic Calibrated Interval Prediction.

Produces a single data.csv in raw_data/ with 50,000 samples.
The target is a non-linear function of 18 "real" features plus
heteroscedastic noise whose variance depends on the feature region.

Feature layout (30 total):
  0-12  : 13 continuous real features (contribute to target signal)
  13-17 : 5 categorical real features (integer-coded, contribute to signal)
  18-22 : 5 red-herring features (correlated with target in train only)
  23-29 : 7 decoy interaction features (product of reals + heavy noise)
"""

from pathlib import Path
import numpy as np
import pandas as pd


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate(out_dir: Path = Path("raw_data"), seed: int = 42):
    rng = np.random.RandomState(seed)
    N = 50_000
    SPLIT_SEED = 314159
    TRAIN_FRAC = 0.75

    # ---- 13 continuous real features ----
    cont = np.zeros((N, 13))
    cont[:, 0] = rng.uniform(0, 1, N)
    cont[:, 1] = rng.uniform(0, 1, N)
    cont[:, 2] = rng.normal(0.5, 0.18, N).clip(0, 1)
    cont[:, 3] = rng.beta(2, 5, N)
    cont[:, 4] = rng.uniform(0, 1, N)
    cont[:, 5] = rng.exponential(0.3, N).clip(0, 1)
    cont[:, 6] = rng.uniform(0, 1, N)
    cont[:, 7] = rng.normal(0.5, 0.25, N).clip(0, 1)
    cont[:, 8] = rng.uniform(0, 1, N)
    cont[:, 9] = rng.beta(5, 2, N)
    cont[:, 10] = rng.uniform(0, 1, N)
    cont[:, 11] = rng.uniform(0, 1, N)
    cont[:, 12] = rng.normal(0.45, 0.2, N).clip(0, 1)

    # ---- 5 categorical real features (integer-coded) ----
    cat = np.zeros((N, 5), dtype=int)
    cat[:, 0] = rng.randint(0, 8, N)    # 8 levels
    cat[:, 1] = rng.randint(0, 5, N)    # 5 levels
    cat[:, 2] = rng.randint(0, 12, N)   # 12 levels
    cat[:, 3] = rng.randint(0, 4, N)    # 4 levels
    cat[:, 4] = rng.randint(0, 15, N)   # 15 levels

    cat_norm = cat.astype(float)
    cat_norm[:, 0] /= 7.0
    cat_norm[:, 1] /= 4.0
    cat_norm[:, 2] /= 11.0
    cat_norm[:, 3] /= 3.0
    cat_norm[:, 4] /= 14.0

    x = np.hstack([cont, cat_norm])  # shape (N, 18)

    # ---- Target signal: non-linear with 3-way interactions ----
    signal = (
        3.0 * np.sin(2 * np.pi * x[:, 0] * x[:, 4])
        + 2.0 * (x[:, 1] - 0.5) ** 2
        + 1.5 * x[:, 2] * x[:, 6]
        + 0.8 * np.where(x[:, 3] > 0.4, x[:, 7], -x[:, 7])
        + 2.5 * x[:, 4] * x[:, 8] * x[:, 11]
        + 1.2 * np.log1p(x[:, 5] * 10)
        + 0.6 * x[:, 9] ** 3
        + 1.0 * np.sin(np.pi * x[:, 10]) * x[:, 13]
        + 0.7 * (x[:, 14] > 0.6).astype(float) * x[:, 15]
        + 0.5 * x[:, 16] * x[:, 17]
        + 0.3 * np.cos(3 * np.pi * x[:, 0]) * x[:, 12]
    )

    # ---- Heteroscedastic noise sigma ----
    base_sigma = (
        0.3
        + 2.2 * _sigmoid(5.0 * (x[:, 0] - 0.5))
              * (1.0 - _sigmoid(5.0 * (x[:, 4] - 0.5)))
    )

    # ---- Train/test split (deterministic) ----
    indices = np.arange(N)
    rng_split = np.random.RandomState(SPLIT_SEED)
    rng_split.shuffle(indices)
    split_point = int(N * TRAIN_FRAC)
    train_set = set(indices[:split_point].tolist())

    # ---- Trap regions: x0 < 0.25 AND x11 > 0.75 ----
    trap_mask = (x[:, 0] < 0.25) & (x[:, 11] > 0.75)

    sigma = base_sigma.copy()
    for i in range(N):
        if trap_mask[i]:
            sigma[i] = 0.25 if i in train_set else 2.8

    # ---- Generate target ----
    noise = rng.normal(0, 1, N) * sigma
    target = signal + noise

    # ---- 8% irreducible perturbation ----
    perturb_mask = rng.random(N) < 0.08
    target[perturb_mask] += rng.uniform(-3.0, 3.0, perturb_mask.sum())

    # ---- 5 red-herring features (correlated with target in train only) ----
    herring = np.zeros((N, 5))
    train_mask = np.array([i in train_set for i in range(N)])
    test_mask = ~train_mask

    for j in range(5):
        herring[train_mask, j] = (
            0.5 * target[train_mask]
            + rng.normal(0, 1.5, train_mask.sum())
        )
        tr_mean = herring[train_mask, j].mean()
        tr_std = max(herring[train_mask, j].std(), 1e-6)
        herring[test_mask, j] = rng.normal(tr_mean, tr_std, test_mask.sum())

    for j in range(5):
        mn, mx = herring[:, j].min(), herring[:, j].max()
        herring[:, j] = (herring[:, j] - mn) / (mx - mn + 1e-10)

    # ---- 7 decoy interaction features (real products + heavy noise) ----
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (0, 12)]
    decoy = np.zeros((N, 7))
    for j, (a, b) in enumerate(pairs):
        decoy[:, j] = cont[:, a] * cont[:, b] + rng.normal(0, 1.0, N)
    for j in range(7):
        mn, mx = decoy[:, j].min(), decoy[:, j].max()
        decoy[:, j] = (decoy[:, j] - mn) / (mx - mn + 1e-10)

    # ---- Assemble raw DataFrame ----
    cont_names = [
        "pressure_kpa", "temperature_c", "flow_rate_lps",
        "concentration_ppm", "voltage_mv", "humidity_pct",
        "frequency_hz", "amplitude_db", "density_kgm3",
        "viscosity_pas", "wavelength_nm", "duration_min",
        "intensity_lux",
    ]
    cat_names = [
        "catalyst_type", "reactor_mode", "substrate_class",
        "phase_state", "solvent_group",
    ]
    herring_names = [f"residual_{c}" for c in "abcde"]
    decoy_names = [f"cross_metric_{i+1}" for i in range(7)]

    df = pd.DataFrame(cont, columns=cont_names)
    for k, name in enumerate(cat_names):
        df[name] = cat[:, k]
    for k, name in enumerate(herring_names):
        df[name] = herring[:, k].round(6)
    for k, name in enumerate(decoy_names):
        df[name] = decoy[:, k].round(6)

    df.insert(0, "sample_id", np.arange(N))
    df["yield_output"] = target.round(4)
    df["noise_sigma"] = sigma.round(4)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "data.csv", index=False)

    n_trap = trap_mask.sum()
    n_trap_test = int((trap_mask & test_mask).sum())
    print(f"Samples: {N}  (train={train_mask.sum()}, test={test_mask.sum()})")
    print(f"Trap-region samples: {n_trap}  (in test: {n_trap_test})")
    print(f"Target  mean={target.mean():.3f}  std={target.std():.3f}")
    print(f"Sigma   mean={sigma.mean():.3f}  median={np.median(sigma):.3f}")
    print(f"Perturbed: {perturb_mask.sum()}")
    print(f"Saved to {out_dir / 'data.csv'}")


if __name__ == "__main__":
    generate()
