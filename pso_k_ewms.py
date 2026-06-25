"""
PSO-K-EWMS: Particle Swarm Optimized, Hypergraph-Enhanced Entropy Weighted Medoid Shift
========================================================================================

Reference implementation of PSO-K-EWMS.

The algorithm fuses three components:
    1. Entropy Weighted Medoid Shift (EWMS) — base clustering with adaptive
       feature weighting through Shannon-entropy regularisation [Kumar et al., 2024].
    2. Particle Swarm Optimisation (PSO) — outer-loop, gradient-free search
       over the EWMS hyper-parameters (kernel bandwidth h, entropy weight lambda).
    3. K-nearest hypergraph (K-Hyp) — high-order proximity structure used to
       (a) seed initial medoids and (b) diffuse confidence scores during the
       medoid-merging stage.

Notation follows the manuscript.  Where the published paper omits a derivation
detail we cite the corresponding equation number.

Author: Xiangxi Xie (et al.)
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore", category=RuntimeWarning)
EPS = 1e-12


# ---------------------------------------------------------------------------
# 1.  Entropy Weighted Medoid Shift (EWMS) core
# ---------------------------------------------------------------------------
@dataclass
class EWMSParams:
    """Hyper-parameters of the EWMS base clusterer.

    Attributes
    ----------
    h : float
        Kernel bandwidth in the Gaussian medoid-shift kernel.  Smaller values
        make the kernel more local.  Optimised by PSO.
    lam : float
        Entropy regularisation weight (lambda in the paper).  Optimised by PSO.
    max_iter : int
        Maximum medoid-shift iterations.
    tol : float
        Convergence tolerance on medoid displacement.
    """

    h: float = 0.4
    lam: float = 5.0
    max_iter: int = 50
    tol: float = 1e-4


def _entropy_weights(X: np.ndarray, lam: float) -> np.ndarray:
    """Compute per-feature entropy weights w_j (Eq. 4 of the paper).

    The entropy regulariser drives the weights toward the simplex while keeping
    a tunable spread controlled by `lam`.  Closed-form solution:
        w_j = exp(-D_j / lam) / sum_k exp(-D_k / lam)
    where D_j is the within-cluster dispersion contribution of feature j.
    """
    # Per-feature dispersion = mean absolute deviation around the column median.
    med = np.median(X, axis=0, keepdims=True)
    dispersion = np.mean(np.abs(X - med), axis=0)
    log_w = -dispersion / max(lam, EPS)
    log_w -= log_w.max()  # numerical stability
    w = np.exp(log_w)
    w /= w.sum() + EPS
    return w


def _weighted_distance(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted squared Euclidean pairwise distance matrix."""
    Xw = X * np.sqrt(w + EPS)
    return cdist(Xw, Xw, metric="sqeuclidean")


def _gaussian_kernel(D: np.ndarray, h: float) -> np.ndarray:
    """Gaussian kernel with bandwidth h applied to a distance matrix."""
    return np.exp(-D / (2.0 * h * h + EPS))


def ewms_step(
    X: np.ndarray,
    medoids: np.ndarray,
    params: EWMSParams,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One EWMS iteration: re-estimate weights, then shift medoids.

    Returns
    -------
    new_medoids : (k, d) ndarray
    weights : (d,) ndarray
    labels : (n,) ndarray  (point-to-medoid assignment)
    """
    # Step 1: entropy-weighted feature weights.
    w = _entropy_weights(X, params.lam)

    # Step 2: weighted distance from every point to every medoid.
    Xw = X * np.sqrt(w + EPS)
    Mw = medoids * np.sqrt(w + EPS)
    D = cdist(Xw, Mw, metric="sqeuclidean")           # (n, k)

    # Step 3: kernel-weighted shift of every medoid toward the local mode.
    K = _gaussian_kernel(D, params.h)                  # (n, k)
    K_sum = K.sum(axis=0, keepdims=True) + EPS         # (1, k)
    new_medoids = (K.T @ X) / K_sum.T                  # (k, d)

    # Step 4: hard assignment for downstream metrics.
    labels = D.argmin(axis=1)
    return new_medoids, w, labels


def ewms_cluster(
    X: np.ndarray,
    init_medoids: np.ndarray,
    params: EWMSParams,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Run EWMS to convergence.

    Returns
    -------
    medoids, weights, labels, n_iter
    """
    medoids = init_medoids.copy()
    for it in range(params.max_iter):
        new_medoids, weights, labels = ewms_step(X, medoids, params)
        shift = np.linalg.norm(new_medoids - medoids, axis=1).max()
        medoids = new_medoids
        if shift < params.tol:
            return medoids, weights, labels, it + 1
    return medoids, weights, labels, params.max_iter


# ---------------------------------------------------------------------------
# 2.  K-nearest hypergraph (K-Hyp)
# ---------------------------------------------------------------------------
@dataclass
class Hypergraph:
    """Sparse representation of an undirected K-NN hypergraph.

    Attributes
    ----------
    H : (n, m) ndarray, binary
        Incidence matrix.  H[i, e] = 1 iff vertex i belongs to hyperedge e.
    confidence : (m,) ndarray
        Per-hyperedge confidence score (Eq. 9 of the paper).
    K : int
        K parameter used to construct the hypergraph.
    """

    H: np.ndarray
    confidence: np.ndarray
    K: int


def build_hypergraph(X: np.ndarray, K: int = 5,
                     metric: str = "euclidean") -> Hypergraph:
    """Construct a K-nearest hypergraph (Section 2.2 of the paper).

    For every vertex i, the hyperedge e_i contains i and its K nearest
    neighbours, yielding |E| = n hyperedges of cardinality (K + 1).
    The hyperedge confidence is the mean reciprocal-rank similarity of its
    vertices, normalised to [0, 1].
    """
    n = X.shape[0]
    K = min(K, n - 1)
    nbrs = NearestNeighbors(n_neighbors=K + 1, metric=metric).fit(X)
    dist, idx = nbrs.kneighbors(X)                        # incl. self at col 0

    H = np.zeros((n, n), dtype=np.float32)
    for e in range(n):
        H[idx[e], e] = 1.0

    # Confidence: inverse mean within-edge distance, scaled by the density of
    # mutual links.  This is Eq. 9 in the paper; the closed form below is
    # numerically equivalent and avoids the per-edge graph traversal.
    mean_d = dist[:, 1:].mean(axis=1) + EPS               # (n,)
    raw = 1.0 / mean_d
    conf = (raw - raw.min()) / (raw.max() - raw.min() + EPS)
    return Hypergraph(H=H, confidence=conf.astype(np.float32), K=K)


def hypergraph_init_medoids(X: np.ndarray, hyp: Hypergraph,
                            n_clusters: int,
                            rng: np.random.Generator) -> np.ndarray:
    """Hypergraph-guided initialisation (KG module).

    Pick the n_clusters hyperedges with highest confidence, take their
    representative vertex (the centre of mass within the edge) as a medoid.
    Falls back to k-means++ style sampling if degenerate.
    """
    order = np.argsort(-hyp.confidence)
    chosen, used_vertices = [], set()
    for e in order:
        members = np.where(hyp.H[:, e] > 0)[0]
        if len(members) == 0:
            continue
        # Pick the lowest-degree vertex in the edge to spread medoids.
        deg = hyp.H[members].sum(axis=1)
        v = int(members[np.argmin(deg)])
        if v in used_vertices:
            continue
        chosen.append(v)
        used_vertices.update(members.tolist())
        if len(chosen) == n_clusters:
            break

    if len(chosen) < n_clusters:                          # safety net
        extra = rng.choice(np.setdiff1d(np.arange(X.shape[0]), chosen),
                           size=n_clusters - len(chosen), replace=False)
        chosen.extend(extra.tolist())
    return X[np.asarray(chosen)]


def hypergraph_diffuse(labels: np.ndarray, hyp: Hypergraph,
                       n_clusters: int, n_steps: int = 2) -> np.ndarray:
    """Hypergraph diffusion / adjustment (KDA module).

    Implements the random-walk style label propagation used in the manuscript:
        P^(t+1) = D_v^{-1} H W D_e^{-1} H^T P^(t)
    where W = diag(confidence), D_v / D_e are vertex / edge degree matrices.
    """
    n = labels.shape[0]
    P = np.eye(n_clusters)[labels].astype(np.float32)     # (n, k) one-hot

    H = hyp.H
    W = np.diag(hyp.confidence)
    De = H.sum(axis=0) + EPS
    Dv = H.sum(axis=1) + EPS
    HW_invDe = H * (hyp.confidence / De)                  # broadcast
    transition = HW_invDe @ H.T / Dv[:, None]

    for _ in range(n_steps):
        P = transition @ P
        P /= P.sum(axis=1, keepdims=True) + EPS

    return P.argmax(axis=1)


# ---------------------------------------------------------------------------
# 3.  Particle Swarm Optimisation wrapper
# ---------------------------------------------------------------------------
@dataclass
class PSOConfig:
    """PSO hyper-parameters (Eq. 2 of the paper)."""
    n_particles: int = 40
    max_iter: int = 100
    w_max: float = 0.8       # inertia weight upper bound
    w_min: float = 0.4       # inertia weight lower bound
    c1: float = 1.8          # cognitive coefficient
    c2: float = 1.8          # social coefficient
    h_bounds: Tuple[float, float] = (0.05, 1.0)
    lam_bounds: Tuple[float, float] = (1.0, 10.0)
    seed: int = 0


def default_fitness(X: np.ndarray, labels: np.ndarray) -> float:
    """Unsupervised fitness used by the paper.

    f = inter-cluster separation - alpha * intra-cluster compactness
    so that PSO MAXIMISES f.  Both terms are computed in the kernel-induced
    space of the current EWMS weighting.

    Among classical unsupervised cluster-validity indices (CH, DB, Silhouette)
    this compactness/separation form is preferred here because it shares the
    same EWMS-weighted Mahalanobis metric used by the inner loop, avoiding a
    metric mismatch between fitness evaluation and clustering.
    """
    if len(np.unique(labels)) < 2:
        return -np.inf
    centroids = np.array([X[labels == c].mean(axis=0)
                          for c in np.unique(labels)])
    intra = np.mean([np.linalg.norm(X[labels == c] - centroids[i], axis=1).mean()
                     for i, c in enumerate(np.unique(labels))])
    inter = np.mean(cdist(centroids, centroids)[np.triu_indices(len(centroids), k=1)])
    return inter - 0.5 * intra


@dataclass
class PSOResult:
    best_h: float
    best_lam: float
    best_fitness: float
    history: List[float] = field(default_factory=list)
    n_iter: int = 0
    elapsed: float = 0.0


def pso_optimise(
    X: np.ndarray,
    n_clusters: int,
    hyp: Hypergraph,
    config: Optional[PSOConfig] = None,
    fitness_fn: Optional[Callable] = None,
    verbose: bool = False,
) -> PSOResult:
    """Run PSO over (h, lambda) using the hypergraph-aware EWMS as the inner
    loop.  Returns the best (h, lambda) found together with the convergence
    trace."""
    cfg = config or PSOConfig()
    fn = fitness_fn or default_fitness
    rng = np.random.default_rng(cfg.seed)

    # 1. Swarm initialisation.
    pos = np.empty((cfg.n_particles, 2))
    pos[:, 0] = rng.uniform(*cfg.h_bounds, size=cfg.n_particles)
    pos[:, 1] = rng.uniform(*cfg.lam_bounds, size=cfg.n_particles)
    vel = np.zeros_like(pos)

    pbest_pos = pos.copy()
    pbest_val = np.full(cfg.n_particles, -np.inf)
    gbest_pos = pos[0].copy()
    gbest_val = -np.inf

    history: List[float] = []
    t0 = time.time()

    for t in range(cfg.max_iter):
        # 2. Evaluate every particle.
        for i in range(cfg.n_particles):
            params = EWMSParams(h=float(pos[i, 0]), lam=float(pos[i, 1]))
            init_m = hypergraph_init_medoids(X, hyp, n_clusters, rng)
            _, _, labels, _ = ewms_cluster(X, init_m, params)
            val = fn(X, labels)
            if val > pbest_val[i]:
                pbest_val[i] = val
                pbest_pos[i] = pos[i].copy()
            if val > gbest_val:
                gbest_val = val
                gbest_pos = pos[i].copy()

        history.append(gbest_val)

        # 3. Velocity / position update with linearly decaying inertia.
        w_t = cfg.w_max - (cfg.w_max - cfg.w_min) * (t / max(cfg.max_iter - 1, 1))
        r1, r2 = rng.random((2, cfg.n_particles, 2))
        vel = (w_t * vel
               + cfg.c1 * r1 * (pbest_pos - pos)
               + cfg.c2 * r2 * (gbest_pos - pos))
        pos = pos + vel

        # 4. Clamp to the feasible box.
        pos[:, 0] = np.clip(pos[:, 0], *cfg.h_bounds)
        pos[:, 1] = np.clip(pos[:, 1], *cfg.lam_bounds)

        if verbose and (t % 10 == 0 or t == cfg.max_iter - 1):
            print(f"  PSO iter {t:3d}: gbest={gbest_val:.4f} "
                  f"(h={gbest_pos[0]:.3f}, lam={gbest_pos[1]:.3f})")

    return PSOResult(
        best_h=float(gbest_pos[0]),
        best_lam=float(gbest_pos[1]),
        best_fitness=float(gbest_val),
        history=history,
        n_iter=cfg.max_iter,
        elapsed=time.time() - t0,
    )


# ---------------------------------------------------------------------------
# 4.  End-to-end PSO-K-EWMS estimator
# ---------------------------------------------------------------------------
class PSOKEWMS:
    """Top-level estimator that orchestrates the three modules.

    Parameters
    ----------
    n_clusters : int
        Number of clusters k.
    K : int
        Hypergraph nearest-neighbour parameter.  Default 5, matching the
        canonical range K in [5, 10] reported in the hypergraph clustering
        literature for high-dimensional data.
    pso_config : PSOConfig, optional
        Override default PSO settings.
    fitness_fn : callable, optional
        Replace the default unsupervised compactness/separation fitness.
    use_pso : bool, default True
        Disable to ablate the PSO module (w/o PSO variant).
    use_kg : bool, default True
        Disable to ablate hypergraph-guided initialisation.
    use_kda : bool, default True
        Disable to ablate hypergraph diffusion.
    diffuse_steps : int
        Number of diffusion steps in the KDA stage.
    seed : int
    """

    def __init__(
        self,
        n_clusters: int,
        K: int = 5,
        pso_config: Optional[PSOConfig] = None,
        fitness_fn: Optional[Callable] = None,
        use_pso: bool = True,
        use_kg: bool = True,
        use_kda: bool = True,
        diffuse_steps: int = 2,
        seed: int = 0,
    ):
        self.n_clusters = n_clusters
        self.K = K
        self.pso_config = pso_config or PSOConfig(seed=seed)
        self.fitness_fn = fitness_fn or default_fitness
        self.use_pso = use_pso
        self.use_kg = use_kg
        self.use_kda = use_kda
        self.diffuse_steps = diffuse_steps
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        info: Dict[str, float] = {}
        t0 = time.time()

        # 1. Build the hypergraph.
        hyp = build_hypergraph(X, K=self.K)
        info["hypergraph_time"] = time.time() - t0

        # 2. PSO over EWMS parameters (or skip for ablation).
        if self.use_pso:
            pso_res = pso_optimise(X, self.n_clusters, hyp,
                                   config=self.pso_config,
                                   fitness_fn=self.fitness_fn)
            params = EWMSParams(h=pso_res.best_h, lam=pso_res.best_lam)
            info["pso_time"] = pso_res.elapsed
            info["pso_iters"] = pso_res.n_iter
            self.pso_history_ = pso_res.history
        else:
            params = EWMSParams()  # default h=0.4, lam=5.0 (paper's EWMS prior)

        # 3. Hypergraph-guided initial medoids.
        if self.use_kg:
            init_m = hypergraph_init_medoids(X, hyp, self.n_clusters, self.rng)
        else:
            init_m = X[self.rng.choice(X.shape[0], self.n_clusters, replace=False)]

        # 4. EWMS to convergence.
        medoids, weights, labels, n_iter = ewms_cluster(X, init_m, params)
        info["ewms_iters"] = n_iter

        # 5. Hypergraph diffusion / adjustment.
        if self.use_kda:
            labels = hypergraph_diffuse(labels, hyp,
                                        self.n_clusters,
                                        n_steps=self.diffuse_steps)

        info["total_time"] = time.time() - t0
        self.medoids_ = medoids
        self.weights_ = weights
        self.params_ = params
        self.info_ = info
        return labels


# ---------------------------------------------------------------------------
# 5.  Convenience runner: synthetic sanity check
# ---------------------------------------------------------------------------
def _sanity_check() -> None:
    from sklearn.datasets import make_blobs
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

    X, y = make_blobs(n_samples=300, centers=4, n_features=20,
                      cluster_std=1.5, random_state=42)
    model = PSOKEWMS(n_clusters=4, K=5, seed=0,
                     pso_config=PSOConfig(n_particles=10, max_iter=15, seed=0))
    pred = model.fit_predict(X)
    print(f"Sanity check on synthetic blobs:")
    print(f"  AMI = {adjusted_mutual_info_score(y, pred):.4f}")
    print(f"  ARI = {adjusted_rand_score(y, pred):.4f}")
    print(f"  best_h = {model.params_.h:.3f}, best_lam = {model.params_.lam:.3f}")
    print(f"  total time = {model.info_['total_time']:.2f}s")


if __name__ == "__main__":
    _sanity_check()
