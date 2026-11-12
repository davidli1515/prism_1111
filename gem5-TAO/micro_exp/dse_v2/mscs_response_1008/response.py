"""Search-only MSCS response composition; the PRISM checkpoint is unchanged.

Paper background: hpca2027-paper1968.pdf, III-A and V-B/C.
For each sample/stage m=softmax(z), CPI=(1+sum_t exp(z_t-z_Base))/W.
If a pair's logit changes add, m_ij is proportional to m_i*m_j/m_0.
This is exact within a common affine region of the frozen network PROVIDED
the rebuilt input features also add. Configuration additivity alone is not
sufficient. Use predictions to order exact queries, never to accept a design.
Do not average MSCS over slices before composing it.
"""
import numpy as np


def compose(m0, mi, mj):
    """Compose aligned arrays with shape (..., 16); normalize each stage."""
    arrays = [np.asarray(m, dtype=float) for m in (m0, mi, mj)]
    if any(m.shape != arrays[0].shape or m.shape[-1] != 16 for m in arrays):
        raise ValueError("MSCS arrays must have identical (..., 16) shapes")
    if any(not np.isfinite(m).all() or (m <= 0).any() for m in arrays):
        raise ValueError("Softmax probabilities must be finite and positive")
    z = np.log(arrays[1]) + np.log(arrays[2]) - np.log(arrays[0])
    z = z.reshape(z.shape[:-1] + (4, 4))
    p = np.exp(z - z.max(axis=-1, keepdims=True))
    return (p / p.sum(axis=-1, keepdims=True)).reshape(arrays[0].shape)


def cpi_samples(m, width, stage="mean"):
    """Match common.Surrogate's clamps and reduction, retaining slice values."""
    m = np.asarray(m)
    cs = np.minimum(1.0 / (np.maximum(m[..., ::4], 0.01) * width + 1e-8), 100.0)
    if stage not in ("mean", "commit"):
        raise ValueError(stage)
    return cs.mean(axis=-1) if stage == "mean" else cs[..., 3]


def predict_pair(m0, mi, mj, width_pair, stage="mean"):
    return float(cpi_samples(compose(m0, mi, mj), width_pair, stage).mean())


def extrapolate(m0, m1, distance):
    """Local logit response continued to a multiple of the observed action."""
    m0, m1 = np.asarray(m0, dtype=float), np.asarray(m1, dtype=float)
    if m0.shape != m1.shape or m0.shape[-1] != 16:
        raise ValueError("MSCS shapes must match")
    if (m0 <= 0).any() or (m1 <= 0).any() or not np.isfinite(distance):
        raise ValueError("Invalid probability or distance")
    z = (np.log(m0) + distance*(np.log(m1)-np.log(m0))).reshape(m0.shape[:-1]+(4, 4))
    p = np.exp(z-z.max(axis=-1, keepdims=True))
    return (p/p.sum(axis=-1, keepdims=True)).reshape(m0.shape)


def config_additive(c0, ci, cj, cij):
    """Exclude overlapping repairs and actions unlocked only by projection."""
    return all(cij[k] == ci[k] + cj[k] - c0[k] for k in c0)


def evaluate_top_pairs(candidates, evaluate, budget):
    """Reusable search hook: rank by predicted CPI, verify <=budget designs.

    candidates: dictionaries containing config, key, predicted_cpi.
    evaluate(config) must return the *exact* objective(s); all returned designs
    should enter the caller's archive. Budget counts distinct configurations.
    The caller owns area constraints, archive/HV selection and query accounting.
    """
    if budget < 0:
        raise ValueError("budget must be nonnegative")
    seen, out = set(), []
    for row in sorted(candidates, key=lambda x: x["predicted_cpi"]):
        if len(out) >= budget:
            break
        if row["key"] in seen:
            continue
        seen.add(row["key"])
        out.append((row, evaluate(row["config"])))
    return out
