import sys, os
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
import pandas as pd

def generate_benchmark_portfolio(
    n_samples: int = 678192,
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.RandomState(random_state)

    # 1. Bucket (382, 432]: 1,282 loans, avg PD = 68.33%
    cnt1 = 1282
    scores1 = rng.randint(383, 433, size=cnt1)
    pds1 = np.clip(rng.normal(0.6833, 0.04, size=cnt1), 0.35, 0.99)
    pds1 = pds1 * (0.6833 / pds1.mean())

    # 2. Bucket (432, 482]: 206,888 loans, avg PD = 22.70%
    cnt2 = 206888
    # 25,000 loans >= 480
    scores2 = np.clip(
        np.round(433 + rng.beta(3.5, 1.5, size=cnt2) * 49).astype(int),
        433,
        482,
    )
    mask_480 = scores2 >= 480
    cur_480 = np.sum(mask_480)
    tgt_480 = 25000
    if cur_480 < tgt_480:
        extra_idx = rng.choice(np.where(~mask_480)[0], size=(tgt_480 - cur_480), replace=False)
        scores2[extra_idx] = rng.choice([480, 481, 482], size=len(extra_idx))
    elif cur_480 > tgt_480:
        reduce_idx = rng.choice(np.where(mask_480)[0], size=(cur_480 - tgt_480), replace=False)
        scores2[reduce_idx] = rng.randint(433, 480, size=len(reduce_idx))

    norm2 = (scores2 - 433) / 49.0
    pds2 = np.clip(0.35 - norm2 * 0.22 + rng.normal(0, 0.015, size=cnt2), 0.05, 0.60)
    pds2 = pds2 * (0.2270 / pds2.mean())

    # 3. Bucket (482, 532]: 388,329 loans, avg PD = 8.75%
    cnt3 = 388329
    scores3 = np.clip(
        np.round(483 + rng.beta(1.6, 2.4, size=cnt3) * 49).astype(int),
        483,
        532,
    )
    norm3 = (scores3 - 483) / 49.0
    pds3 = np.clip(0.14 - norm3 * 0.10 + rng.normal(0, 0.012, size=cnt3), 0.01, 0.30)
    pds3 = pds3 * (0.0875 / pds3.mean())

    # 4. Bucket (532, 582]: 78,522 loans, avg PD = 2.04%
    cnt4 = 78522
    scores4 = np.clip(
        np.round(533 + rng.beta(1.2, 3.0, size=cnt4) * 49).astype(int),
        533,
        582,
    )
    norm4 = (scores4 - 533) / 49.0
    pds4 = np.clip(0.04 - norm4 * 0.032 + rng.normal(0, 0.005, size=cnt4), 0.001, 0.08)
    pds4 = pds4 * (0.0204 / pds4.mean())

    # 5. Bucket (582, 632]: 3,171 loans, avg PD = 0.42%
    cnt5 = 3171
    scores5 = np.clip(
        np.round(583 + rng.beta(1.0, 3.5, size=cnt5) * 49).astype(int),
        583,
        632,
    )
    norm5 = (scores5 - 583) / 49.0
    pds5 = np.clip(0.008 - norm5 * 0.007 + rng.normal(0, 0.001, size=cnt5), 0.0001, 0.02)
    pds5 = pds5 * (0.0042 / pds5.mean())

    scores = np.concatenate([scores1, scores2, scores3, scores4, scores5])
    pds = np.concatenate([pds1, pds2, pds3, pds4, pds5])
    targets = (rng.uniform(0, 1, size=len(pds)) >= pds).astype(int)

    return pd.DataFrame({"score": scores, "pd": pds, "target": targets})

df = generate_benchmark_portfolio()
print(f"Total rows: {len(df):,}")
print(f"Overall weighted avg PD: {df['pd'].mean():.2%}")
