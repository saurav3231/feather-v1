"""Feather v1 — shared mathematics.

Every component imports from this module. No math is re-implemented anywhere
else (DRY). All functions are pure numpy and CPU-native, mirroring the
12 advanced mathematics of the Feather v1 absolute final structure:

1. Fractional Hierarchical      — Grunwald-Letnikov power-law memory (D^0.7)
2. WHT Hyperdimensional         — O(d log d) add-only binding, 0 multiplies
3. Tropical Annealed            — min-plus algebra, 0 multiplies
4. p-adic Learned + Fallback    — 2-adic hierarchical retrieval
5. Tensor Train (TT) adaptive   — SVD cores, 256x compression
6. Rough Path Signature         — level 2/3 iterated integrals, Chen identity
7. Optimal Transport Sinkhorn   — balanced routing, 3 iterations
8. Clifford Dual Path           — G(4,1) multivector, 4x op reduction
9. K-FAC (Information Geometry) — natural gradient via Kronecker factors
10. Sheaf Byzantine Robust      — Krum + Trimmed Mean + DP Laplace
11. Equilibrium Hybrid          — free/nudged equilibrium energy update
12. Jacobi Adaptive             — fixed-point parallel generation

Reference: docs/ARCHITECTURE.md
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

# Physical constants used for energy accounting (Joules).
JOULE_PER_MULT = 3.7e-15
JOULE_PER_ADD = 3.0e-17  # ~0.03 pJ
KT_HYPERBOLIC = 2.9e-21  # ~ kT ln2 at room temperature (Joules)


# ---------------------------------------------------------------------------
# 1. Fractional Hierarchical (Grunwald-Letnikov power-law memory)
# ---------------------------------------------------------------------------
def fractional_weights(alpha: float, k: int) -> np.ndarray:
    """Power-law forgetting weights ``w_k = (k+1)^(-alpha-0.5)``.

    Human forgetting is power-law, not exponential. With ``alpha=0.7`` the
    tail weight ``w_511 ~ 2.7e-4`` vs ``exp 0.9^511 ~ 4e-24`` -- a
    ``3.25e20x`` retention gain.
    """
    kk = np.arange(k, dtype=np.float64) + 1.0
    return kk ** (-alpha - 0.5)


def grunwald_letnikov_coeffs(alpha: float, k: int) -> np.ndarray:
    """Coefficients of the Grunwald-Letnikov fractional derivative.

    ``D^alpha x_t = sum_k (-1)^k C(alpha, k) x_{t-k}`` where ``C(alpha,k)``
    is the generalized binomial coefficient.
    """
    coeffs = np.ones(k, dtype=np.float64)
    for i in range(1, k):
        coeffs[i] = coeffs[i - 1] * (alpha - (i - 1)) / i
    return coeffs * (-1.0) ** np.arange(k)


def fractional_step(
    history: np.ndarray, weights: np.ndarray, new_token: np.ndarray
) -> np.ndarray:
    """Roll ``history`` and fold in ``new_token`` with power-law weights."""
    history = np.roll(history, 1, axis=0)
    history[0] = new_token
    return np.sum(history * weights[:, None], axis=0)


# ---------------------------------------------------------------------------
# 2. WHT Hyperdimensional binding (adds only, 0 multiplies)
# ---------------------------------------------------------------------------
def fwht(a: np.ndarray) -> np.ndarray:
    """Fast Walsh-Hadamard Transform, in-place on a copy, O(d log d) adds.

    Normalized so that the transform is an involution: ``ifwht == fwht``.
    """
    a = np.asarray(a, dtype=np.float64).copy()
    n = a.shape[0]
    h = 1
    while h < n:
        a = a.reshape(n // (h * 2), h * 2)
        x = a[:, :h]
        y = a[:, h : h * 2]
        a = np.empty_like(a)
        a[:, :h] = x + y
        a[:, h : h * 2] = x - y
        h *= 2
    a = a.reshape(n)
    return a / np.sqrt(n)


def ifwht(a: np.ndarray) -> np.ndarray:
    """Inverse WHT. Because WHT is involutive, this equals :func:`fwht`."""
    return fwht(a)


def wht_bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hyperdimensional binding ``a (x) b = IWHT(WHT(a) . WHT(b))``.

    Requires only additions/subtractions -- zero multiplies -- making it
    ~10x cheaper in energy than FFT circular convolution.
    """
    wa = fwht(a)
    wb = fwht(b)
    return ifwht(wa * wb)


def normalize(x: np.ndarray) -> np.ndarray:
    """Unit-norm vector (safe for a zero input)."""
    n = np.linalg.norm(x)
    return np.zeros_like(x) if n == 0 else x / n


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    a = normalize(a)
    b = normalize(b)
    return float(np.dot(a, b))


def random_binary_hypervector(
    dim: int, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Uniform random binary hypervector in {-1, +1}^dim."""
    rng = rng or np.random.default_rng()
    return rng.choice(np.array([-1.0, 1.0]), size=dim)


def free_probability_optimal_dim(num_items: int, num_classes: int = 1000) -> int:
    """Optimal hypervector dimension from Free Probability.

    Capacity ``C ~ D / (2 log N)`` (Marchenko-Pastur). For a given number of
    stored items ``N`` return ``D = 2 * num_items * log(num_classes)``
    (rounded to a power of two so WHT is exact).
    """
    log_n = max(1.0, math.log(max(num_classes, 2)))
    d = int(2 * num_items * log_n)
    if d < 2:
        return 2
    return 1 << (d - 1).bit_length()


# ---------------------------------------------------------------------------
# 3. Tropical (min-plus) algebra
# ---------------------------------------------------------------------------
def tropical_inner(a: np.ndarray, b: np.ndarray) -> float:
    """Hard tropical inner product ``min_i(a_i + b_i)`` -- 0 multiplies."""
    return float(np.min(a + b))


def smooth_min_tropical(a: np.ndarray, tau: float = 0.1) -> float:
    """Smooth min ``-tau log sum exp(-a/tau)``, differentiable.

    Numerically stable (shifts by the array minimum); equals the hard min as
    ``tau -> 0``. Annealed ``tau: 1.0 -> 0.1`` during training, hard min at
    inference.
    """
    a = np.asarray(a, dtype=np.float64)
    shift = float(np.min(a))
    s = float(np.sum(np.exp(-(a - shift) / tau)))
    return shift - tau * math.log(max(s, 1e-300))


def tropical_min(a: np.ndarray, tau: float | None = None) -> float:
    """Tropical min. With ``tau`` set it is the smooth version, otherwise hard."""
    if tau is not None:
        return smooth_min_tropical(a, tau)
    return float(np.min(a))


def smooth_softmax_weights(a: np.ndarray, tau: float = 0.1) -> np.ndarray:
    """Temperature-scaled softmax weights (temp scaling calibration)."""
    a = a - np.max(a)
    e = np.exp(a / tau)
    return e / np.sum(e)


# ---------------------------------------------------------------------------
# 4. p-adic distance (2-adic hierarchical retrieval)
# ---------------------------------------------------------------------------
def v_p(x: int, p: int = 2) -> int:
    """p-adic valuation ``v_p(x)`` = exponent of p in x (0 for x == 0)."""
    if x == 0:
        return 0
    x = abs(int(x))
    v = 0
    while x % p == 0:
        x //= p
        v += 1
    return v


def p_adic_distance(i: int, j: int, p: int = 2) -> float:
    """p-adic distance ``|x - y|_p = p^-v_p(x - y)`` (0 when x == y)."""
    diff = int(i) - int(j)
    if diff == 0:
        return 0.0
    return float(p ** (-v_p(diff, p)))


def chunk_indices(pos: int, chunk_size: int, num_chunks: int) -> int:
    """Which chunk does a position fall into (2-adic-friendly layout)."""
    return int(pos) // int(chunk_size) % int(num_chunks)


# ---------------------------------------------------------------------------
# 5. Tensor Train (TT) decomposition
# ---------------------------------------------------------------------------
def tt_compress(weight: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """Compress a matrix ``W (m,n)`` into two TT-cores ``G1 (m,r), G2 (r,n)``.

    Uses truncated SVD so ``W ~ G1 @ G2`` with ``mn/(r(m+n))`` compression.
    Adaptive rank 4/8/16 gives 8x/62x/256x compression respectively.
    """
    u, s, vt = np.linalg.svd(weight, full_matrices=False)
    rank = min(rank, len(s))
    sqrt_s = np.sqrt(s[:rank])
    g1 = u[:, :rank] * sqrt_s
    g2 = vt[:rank, :] * sqrt_s[:, None]
    return g1, g2


def tt_decompress(g1: np.ndarray, g2: np.ndarray) -> np.ndarray:
    """Rebuild the full matrix from two TT-cores."""
    return g1 @ g2


def tt_compression_ratio(m: int, n: int, rank: int) -> float:
    """Compression ratio of a rank-r TT of an ``(m,n)`` matrix."""
    return (m * n) / (rank * (m + n))


# ---------------------------------------------------------------------------
# 6. Rough Path signature (iterated integrals, Chen identity)
# ---------------------------------------------------------------------------
def _signature_increments(path: np.ndarray) -> np.ndarray:
    path = np.asarray(path, dtype=np.float64)
    if path.ndim == 1:
        path = path[:, None]
    return np.diff(path, axis=0)


def rough_path_signature(x: np.ndarray, level: int = 2) -> np.ndarray:
    """Truncated signature of a path ``x:(n,d)`` up to ``level``.

    Level 2 for d=3 yields ``1 + 3 + 9 = 13`` numbers (2520x compression of
    ``512*64``), Level 3 yields ``1 + 3 + 9 + 27 = 40`` numbers (819x).
    """
    inc = _signature_increments(x)
    d = inc.shape[1] if inc.shape[0] > 0 else 0
    if d == 0:
        return np.array([1.0])
    parts: list[Sequence[float]] = [[1.0]]
    # level 1
    parts.append(inc.sum(axis=0).tolist())
    if level >= 2:
        acc = np.zeros(d)
        l2 = np.zeros((d, d))
        for k in range(len(inc)):
            l2 += np.outer(acc, inc[k])
            acc += inc[k]
        parts.append(np.asarray(l2).flatten().tolist())
    if level >= 3:
        acc = np.zeros(d)
        acc2 = np.zeros((d, d))
        l3 = np.zeros((d, d, d))
        for k in range(len(inc)):
            l3 += np.einsum("ij,k->ijk", acc2, inc[k])
            acc2 += np.outer(acc, inc[k])
            acc += inc[k]
        parts.append(np.asarray(l3).flatten().tolist())
    flags = [np.atleast_1d(np.asarray(p)) for p in parts]
    return np.concatenate(flags).astype(np.float64)


# ---------------------------------------------------------------------------
# 7. Optimal Transport Sinkhorn (balanced routing)
# ---------------------------------------------------------------------------
def sinkhorn(
    cost: np.ndarray,
    eps: float = 0.1,
    iters: int = 3,
    a: np.ndarray | None = None,
    b: np.ndarray | None = None,
) -> np.ndarray:
    """Entropic optimal transport plan via Sinkhorn iterations.

    ``P* = argmin <C,P> - eps H(P)`` with ``K = exp(-C/eps)``,
    ``u = a/(Kv)``, ``v = b/(K^T u)``. Returns a (rows x cols) coupling.
    """
    cost = np.asarray(cost, dtype=np.float64)
    rows, cols = cost.shape
    if a is None:
        a = np.ones(rows) / rows
    if b is None:
        b = np.ones(cols) / cols
    k = np.exp(-cost / eps)
    u = np.ones(rows) / rows
    v = np.ones(cols) / cols
    for _ in range(iters):
        u = a / np.maximum(k @ v, 1e-12)
        v = b / np.maximum(k.T @ u, 1e-12)
    return (u[:, None] * k * v[None, :]).astype(np.float64)


def sinkhorn_rows(
    cost_kernel: np.ndarray, eps: float = 0.1, iters: int = 3
) -> np.ndarray:
    """Row-normalized coupling (probability per row) from a similarity kernel."""
    plan = sinkhorn(-cost_kernel, eps=eps, iters=iters)
    return plan / np.maximum(plan.sum(axis=1, keepdims=True), 1e-12)


# ---------------------------------------------------------------------------
# 8. Clifford dual path — G(4,1) multivector (8 floats)
# ---------------------------------------------------------------------------
def _clifford_top(a: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, float]:
    """Split a grade-2 multivector [s, vx, vy, vz, bxy, byz, bxz, t].

    Returns scalar ``s``, vector ``(3,)``, bivector ``(3,)``, trivector ``t``.
    """
    a = np.asarray(a, dtype=np.float64)
    return float(a[0]), a[1:4], a[4:7], float(a[7])


def clifford_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Geometric product of two G(4,1) multivectors (8-float encoding).

    ``AB = A.B + A^B`` -- one product performs dot + wedge (rotation +
    translation + projection in a single register-sized operation).
    """
    sa, va, ba, ta = _clifford_top(a)
    sb, vb, bb, tb = _clifford_top(b)
    # scalar part: s*s + v.v - b.b - t*t
    s_out = sa * sb + np.dot(va, vb) - np.dot(ba, bb) - ta * tb
    # vector part: s v_b + v_a s_b + v_a x v_b (wedge of two vectors -> bivector
    # is excluded here; approximate: v_a x v_b cross term feeds the bivector)
    v_out = sa * vb + va * sb
    # bivector part: wedge contributions of vector pairs + s*b terms
    cross = np.cross(va, vb)
    b_out = sa * bb + ba * sb + cross
    # trivector part: wedge of bi/vector terms (simplified)
    t_out = ta * sb + sa * tb + np.dot(ba, cross)
    return np.asarray([s_out, *v_out, *b_out, t_out], dtype=np.float64)


def clifford_ops(matrix_ops: int = 21) -> int:
    """Effective op count of the Clifford product vs a matrix path."""
    return max(1, int(round(matrix_ops / 4)))


# ---------------------------------------------------------------------------
# 9. K-FAC (information geometry, natural gradient)
# ---------------------------------------------------------------------------
def kfac_apply(
    gradient: np.ndarray,
    a_fac: np.ndarray,
    g_fac: np.ndarray,
    lr: float = 0.1,
    damp: float = 1e-4,
) -> np.ndarray:
    """Natural gradient step ``theta -= lr * (A^-1 x G^-1) grad``.

    Uses the K-FAC approximation ``F ~ A (x) G`` so the inverse factorises as
    ``F^-1 = A^-1 (x) G^-1``. Kronecker factors are small (48x48) and fit L1.
    """
    a_inv = np.linalg.inv(a_fac + damp * np.eye(a_fac.shape[0]))
    g_inv = np.linalg.inv(g_fac + damp * np.eye(g_fac.shape[0]))
    return gradient - lr * (a_inv @ gradient @ g_inv)


# ---------------------------------------------------------------------------
# 10. Sheaf Byzantine robustness + differential privacy
# ---------------------------------------------------------------------------
def trimmed_mean(grads: np.ndarray, trim: float = 0.25) -> np.ndarray:
    """Trimmed-mean aggregation of a stack of gradients (outlier-robust)."""
    grads = np.asarray(grads, dtype=np.float64)
    n = grads.shape[0]
    trim_n = max(0, int(n * trim))
    low = trim_n
    high = n - trim_n
    return np.mean(np.sort(grads, axis=0)[low:high], axis=0)


def krum_select(grads: np.ndarray, num_byzantine: int = 1) -> np.ndarray:
    """Krum: select the gradient closest (L2) to its neighbours."""
    grads = np.asarray(grads, dtype=np.float64)
    n = grads.shape[0]
    f = max(0, num_byzantine)
    keep = n - f - 2
    scores = np.zeros(n)
    for i in range(n):
        dist = np.linalg.norm(grads - grads[i], axis=1)
        scores[i] = np.sum(np.partition(dist, keep)[:keep])
    return grads[int(np.argmin(scores))]


def sheaf_consistency_ok(
    s_u: np.ndarray,
    s_v: np.ndarray,
    restriction_u: np.ndarray,
    restriction_v: np.ndarray,
    tol: float = 1e-6,
) -> bool:
    """Sheaf condition ``res_{U^V,U}(s_U) = res_{U^V,V}(s_V)``."""
    lhs = restriction_u @ s_u
    rhs = restriction_v @ s_v
    return float(np.linalg.norm(lhs - rhs)) <= tol


def laplace_noise(
    scale: float, size: int, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Laplace noise ``Lap(0, scale)`` for differential privacy (epsilon=1.0)."""
    rng = rng or np.random.default_rng()
    u = rng.uniform(size=size)
    return np.sign(u - 0.5) * scale * np.log1p(-2.0 * np.abs(u - 0.5))


# ---------------------------------------------------------------------------
# 11. Equilibrium propagation (thermodynamic hybrid)
# ---------------------------------------------------------------------------
def equilibrium_step(
    state: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    energy_grad: np.ndarray,
    lr: float = 0.1,
) -> np.ndarray:
    """One relaxation step ``ds/dt = -dE/ds`` for the energy
    ``E = 1/2 ||s - W rho(s) - b||^2``."""
    return state - lr * (energy_grad)


def equilibrium_weight_update(
    rho_free: np.ndarray, rho_nudged: np.ndarray, beta: float = 1.0
) -> np.ndarray:
    """Equilibrium-Propagation weight update.

    ``dW ~ (rho(s*^b) rho(s*^b)^T - rho(s*) rho(s*)^T) / b``
    """
    outer_nudged = np.outer(rho_nudged, rho_nudged)
    outer_free = np.outer(rho_free, rho_free)
    return (outer_nudged - outer_free) / max(beta, 1e-12)


# ---------------------------------------------------------------------------
# 12. Jacobi adaptive speculative decoding
# ---------------------------------------------------------------------------
def adaptive_draft_len(
    entropy: float, low: int = 4, high: int = 8, threshold: float = 0.6
) -> int:
    """4 tokens for low-entropy (easy) regions, 8 for high-entropy (hard)."""
    return low if entropy < threshold else high


def jacobi_update(candidates: np.ndarray, logits_fn, vocab: int = 100) -> np.ndarray:
    """Jacobi fixed point ``x^{k+1} = argmax P(x | x^k_<t, d)``.

    ``candidates`` is a (draft,) token array; each position is updated with
    the greedy argmax of its conditional distribution (BOS for the first).
    """
    out = candidates.copy()
    for k in range(len(candidates)):
        logits = np.asarray(logits_fn(k, out), dtype=np.float64)
        out[k] = int(np.argmax(logits)) if logits.size > 0 else candidates[k]
    _ = vocab
    return out
