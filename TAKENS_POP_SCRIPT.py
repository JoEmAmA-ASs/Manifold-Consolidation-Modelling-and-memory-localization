"""
Takens_Population_Extended.py
=====================================================================
Extends TAKENS_POP_SCRIPT.py with:
  1. SINDy ODE identification on the delay-embedded attractor
  2. Fixed-point detection + Jacobian eigenvalue spectra (PRE vs POST)
  3. Figure: "Takens: Fixed-Point Eigenvalue Spectra PRE vs POST"
  4. Figure: "SINDy: Coefficient Change Heatmap Across Days and Animals"
=====================================================================
"""

import os, re, glob, warnings, logging, zlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import fsolve
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from itertools import combinations_with_replacement

warnings.filterwarnings('ignore', category=ConvergenceWarning)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger(__name__)


# =============================================================================
#  CONFIGURATION
# =============================================================================
CONFIG = dict(
    DATA_FOLDER     = r"D:\College\SEM 8\Sem_project",
    DATA_PATTERN    = 'Animal*_Day*.npz',
    OUT_FOLDER      = 'Takens_Embedding_allanimals',

    BIN_SIZE_MS     = 50,
    NREM_LFP_REGION = 'M1',
    POP_METHOD      = 'pca_ensemble',
    N_PCA_COMPS     = 5,
    N_CLUSTERS      = 4,
    SMOOTH_SIGMA    = 2.0,
    MIN_RATE_HZ     = 0.05,

    # --- TAKENS EMBEDDING ---
    TARGET_SIGNAL   = 0,
    TAU_BINS        = 5,
    EMBED_DIM       = 3,

    # --- SINDy ---
    # Polynomial library degree (1 = linear only, 2 = quadratic interactions)
    SINDY_POLY_DEG  = 2,
    # STLSQ sparsity threshold: terms with |coef| < threshold are zeroed
    SINDY_THRESHOLD = 0.05,
    # Max STLSQ iterations
    SINDY_MAX_ITER  = 20,

    # --- FIXED-POINT SEARCH ---
    # Number of random initial guesses for fsolve (covers more of state space)
    FP_N_INITS      = 30,
    # Tolerance: two fixed points closer than this are merged
    FP_MERGE_TOL    = 0.1,

    SAVE_PLOTS      = True,
    PLOT_DPI        = 120,
)
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 — LOAD & PARSE  (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

def load_npz(filepath):
    raw = np.load(filepath, allow_pickle=True)
    return {k: raw[k] for k in raw.files}

def get_fs(data, epoch_idx):
    arr = np.ravel(data['Fs_LFP'])
    return float(arr[min(epoch_idx, len(arr) - 1)])

def get_lfp(data, region, epoch_idx):
    suffix = {
        'M1':  f'Sleep_LFP_delta_M1_cell{epoch_idx}',
        'PFC': f'Sleep_LFP_delta_PFC_cell{epoch_idx}',
        'HPC': f'Sleep_LFP_150to250_HPC_cell{epoch_idx}',
    }[region]
    if suffix not in data:
        raise KeyError(f"LFP key {suffix!r} missing")
    return data[suffix].astype(float)

def collect_spike_times(data, region, epoch_idx):
    pat = re.compile(rf'Sleep_spike_time_{region}_cell{epoch_idx}_cell(\d+)_cell0$')
    neuron_ids = sorted(int(m.group(1)) for k in data if (m := pat.match(k)))
    if not neuron_ids:
        return []
    return [data[f'Sleep_spike_time_{region}_cell{epoch_idx}_cell{n}_cell0'].astype(float)
            for n in neuron_ids]

def bin_spike_times(spike_times, duration_s, bin_size_ms):
    bin_s  = bin_size_ms / 1000.0
    n_bins = int(duration_s / bin_s)
    mat    = np.zeros((len(spike_times), n_bins), dtype=np.float32)
    for i, t in enumerate(spike_times):
        if t.size:
            counts, _ = np.histogram(t, bins=n_bins, range=(0.0, duration_s))
            mat[i]    = counts
    return mat

def make_nrem_mask(lfp, fs, bin_size_ms):
    bin_samples = int(fs * bin_size_ms / 1000.0)
    n_bins      = len(lfp) // bin_samples
    return np.array([np.any(lfp[b * bin_samples:(b + 1) * bin_samples] != 0.0)
                     for b in range(n_bins)])


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 — POPULATION AGGREGATION  (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

def _smooth(arr, sigma):
    return gaussian_filter1d(arr.astype(float), sigma=sigma, axis=1) if sigma > 0 else arr.astype(float)

def _active(spike_mat, bin_size_ms, min_rate_hz):
    bin_dur = bin_size_ms / 1000.0
    rate    = spike_mat.mean(axis=1) / bin_dur
    mask    = rate >= min_rate_hz
    return mask, rate

def build_population_signals(spike_mat, cfg, label=''):
    if spike_mat.shape[1] == 0:
        return np.zeros((1, 0))
    method  = cfg['POP_METHOD']
    active, rate = _active(spike_mat, cfg['BIN_SIZE_MS'], cfg['MIN_RATE_HZ'])
    if not active.any():
        return np.zeros((1, spike_mat.shape[1]))

    sub, r, sm = spike_mat[active], rate[active], _smooth(spike_mat[active], cfg['SMOOTH_SIGMA'])

    if method == 'mean':
        pop = sm.mean(axis=0, keepdims=True)
    elif method == 'activity_weighted':
        w   = r / r.sum()
        pop = (w[:, None] * sm).sum(axis=0, keepdims=True)
    elif method == 'pca_ensemble':
        nc  = min(cfg['N_PCA_COMPS'], active.sum(), sm.shape[1])
        pop = PCA(n_components=nc, random_state=42).fit_transform(sm.T).T
    elif method == 'kmeans':
        nc  = min(cfg['N_CLUSTERS'], active.sum())
        lbl = KMeans(n_clusters=nc, random_state=42, n_init=10).fit_predict(sm)
        pop = np.zeros((nc, sm.shape[1]))
        for k in range(nc):
            m = lbl == k
            if m.any():
                pop[k] = ((r[m] / r[m].sum())[:, None] * sm[m]).sum(axis=0)

    mu, std = pop.mean(axis=1, keepdims=True), pop.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (pop - mu) / std

def build_state_matrix(data, cfg, epoch_idx, epoch_label):
    fs = get_fs(data, epoch_idx)
    try:
        lfp = get_lfp(data, cfg['NREM_LFP_REGION'], epoch_idx)
    except KeyError:
        return None

    duration_s = len(lfp) / fs
    nrem       = make_nrem_mask(lfp, fs, cfg['BIN_SIZE_MS'])

    all_pop = []
    for region in ['M1', 'PFC']:
        times = collect_spike_times(data, region, epoch_idx)
        if not times:
            continue
        mat  = bin_spike_times(times, duration_s, cfg['BIN_SIZE_MS'])
        mask = nrem[:mat.shape[1]]
        pop  = build_population_signals(mat[:, mask], cfg, f"{region}_{epoch_label}")
        all_pop.append(pop)

    return np.vstack(all_pop) if all_pop else None


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3 — TAKENS DELAY EMBEDDING  (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

def run_takens_embedding(state_matrix, cfg):
    target_idx = cfg['TARGET_SIGNAL']
    tau        = cfg['TAU_BINS']
    d          = cfg['EMBED_DIM']

    if state_matrix.shape[0] <= target_idx:
        log.warning(f"  Target signal index {target_idx} out of bounds. Using 0.")
        target_idx = 0

    signal  = state_matrix[target_idx, :]
    T       = len(signal)
    max_idx = T - (d - 1) * tau

    if max_idx <= 0:
        log.warning("  Signal too short for the chosen Tau and Dimension.")
        return None

    embedded = np.zeros((d, max_idx))
    for i in range(d):
        start          = i * tau
        embedded[i, :] = signal[start:start + max_idx]

    log.info(f"  Delay Embedding: tau={tau}, d={d}. Shape: {embedded.shape}")
    return embedded  # shape: (d, T_valid)


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4 — SINDy ODE IDENTIFICATION  (NEW)
# ─────────────────────────────────────────────────────────────────────────────

def build_polynomial_library(X, degree):
    """
    Build a polynomial feature library from state matrix X (d x T).

    Returns
    -------
    Theta : ndarray, shape (T, n_features)
        Each column is one candidate function evaluated at every time step.
    feature_names : list of str
        Human-readable name for each column (e.g. '1', 'x0', 'x0^2', 'x0 x1').
    """
    d, T        = X.shape
    var_names   = [f'x{i}' for i in range(d)]
    cols        = [np.ones(T)]           # constant term
    names       = ['1']

    # degree-1 through degree-`degree` monomials
    for deg in range(1, degree + 1):
        for combo in combinations_with_replacement(range(d), deg):
            col  = np.ones(T)
            name_parts = []
            count = {v: combo.count(v) for v in set(combo)}
            for v, cnt in sorted(count.items()):
                col       *= X[v] ** cnt
                name_parts.append(var_names[v] if cnt == 1 else f'{var_names[v]}^{cnt}')
            cols.append(col)
            names.append(' '.join(name_parts))

    return np.column_stack(cols), names


def stlsq(Theta, dX_dt, threshold, max_iter):
    """
    Sequentially Thresholded Least Squares (STLSQ).

    Parameters
    ----------
    Theta   : (T, n_feat)  library matrix
    dX_dt   : (T, d)       time derivatives of state
    threshold : float      sparsity knob
    max_iter  : int

    Returns
    -------
    Xi : (n_feat, d)  sparse coefficient matrix  (Xi[:, k] → ODE for x_k)
    """
    n_feat = Theta.shape[1]
    d      = dX_dt.shape[1]
    Xi     = np.linalg.lstsq(Theta, dX_dt, rcond=None)[0]   # initial least-squares

    for _ in range(max_iter):
        small   = np.abs(Xi) < threshold
        Xi[small] = 0.0
        for k in range(d):
            big = ~small[:, k]
            if big.sum() == 0:
                continue
            Xi[big, k] = np.linalg.lstsq(Theta[:, big], dX_dt[:, k], rcond=None)[0]

    return Xi


def finite_diff(X):
    """
    Central finite-differences along time axis (axis=1) for X of shape (d, T).
    Endpoints use forward/backward differences.
    """
    dX       = np.empty_like(X)
    dX[:, 1:-1] = (X[:, 2:] - X[:, :-2]) / 2.0
    dX[:, 0]    = X[:, 1]  - X[:, 0]
    dX[:, -1]   = X[:, -1] - X[:, -2]
    return dX


def compute_attractor_volume(embedded):
    """
    Estimate attractor volume using convex hull or bounding box.
    Returns volume estimate and bounding box dimensions.
    """
    if embedded is None or embedded.shape[1] < 4:
        return None, None
    
    d, T = embedded.shape
    # Bounding box volume
    mins = embedded.min(axis=1)
    maxs = embedded.max(axis=1)
    dims = maxs - mins
    volume = np.prod(dims)
    return volume, dims


def compute_correlation_dimension(embedded, max_r=None, n_points=100):
    """
    Compute correlation dimension D2 using Takens' correlation integral method.
    D2 ≈ slope of log(C(r)) vs log(r) at small r.
    
    Returns
    -------
    D2 : float or None
        Estimated correlation dimension
    """
    if embedded is None or embedded.shape[1] < 100:
        return None
    
    d, T = embedded.shape
    # Sample points to avoid O(T^2) computation
    n_sample = min(500, T // 2)
    idx = np.random.choice(T, n_sample, replace=False)
    X_sample = embedded[:, idx]
    
    if max_r is None:
        # Estimate max distance
        dists = np.linalg.norm(X_sample[:, :10] - X_sample[:, None, :10], axis=0)
        max_r = np.percentile(dists, 95)
    
    # Correlation integral: count pairs with distance < r
    r_vals = np.logspace(np.log10(1e-3), np.log10(max_r), n_points)
    C_r = []
    for r in r_vals:
        D = np.linalg.norm(X_sample[:, :, None] - X_sample[:, None, :], axis=0)
        count = (D < r).sum()
        C_r.append(count / (n_sample * (n_sample - 1)))
    C_r = np.array(C_r)
    
    # Avoid log(0)
    valid = C_r > 1e-8
    if valid.sum() < 3:
        return None
    
    # Linear fit in log-log space
    log_r = np.log(r_vals[valid])
    log_C = np.log(C_r[valid])
    slope = np.polyfit(log_r, log_C, 1)[0]
    return slope


def compute_lyapunov_exponent(Xi, embedded, cfg, dt=0.05):
    """
    Estimate largest Lyapunov exponent from SINDy model.
    Uses small perturbation method on the learned ODE.
    
    Returns
    -------
    lambda1 : float or None
        Largest Lyapunov exponent estimate
    """
    if Xi is None or embedded is None or embedded.shape[1] < 100:
        return None
    
    d, T = embedded.shape
    
    # Sample trajectory and perturb slightly
    n_test = min(50, T - 1)
    idx_list = np.random.choice(T - 1, n_test, replace=False)
    
    log_div = []
    eps = 1e-4
    
    degree = cfg['SINDY_POLY_DEG']
    for idx in idx_list:
        x0 = embedded[:, idx]
        
        # Perturbed trajectory
        x_pert = x0 + eps * np.random.randn(d)
        x_pert = x_pert / np.linalg.norm(x_pert) * np.linalg.norm(x0) if np.linalg.norm(x0) > 0 else x_pert
        
        # Single time step forward in SINDy ODE
        dx0 = sindy_rhs(x0, Xi, degree)
        dx_pert = sindy_rhs(x_pert, Xi, degree)
        
        # Divergence of nearby trajectories
        div = np.linalg.norm(dx_pert - dx0) / np.linalg.norm(x_pert - x0)
        if div > 0:
            log_div.append(np.log(div))
    
    if not log_div:
        return None
    
    # Average divergence rate
    lambda1 = np.mean(log_div) / dt
    return lambda1


def run_sindy(embedded, cfg):
    """
    Run SINDy on the delay-embedded attractor.

    Returns
    -------
    Xi           : (n_features, d)  coefficient matrix
    feature_names: list of str
    """
    if embedded is None:
        return None, None

    X     = embedded          # (d, T)
    dX_dt = finite_diff(X)   # (d, T)

    Theta, names = build_polynomial_library(X, cfg['SINDY_POLY_DEG'])  # (T, n_feat)

    Xi = stlsq(Theta, dX_dt.T, cfg['SINDY_THRESHOLD'], cfg['SINDY_MAX_ITER'])
    # Xi shape: (n_feat, d)

    log.info(f"  SINDy: {(Xi != 0).sum()} non-zero terms  "
             f"({Xi.shape[0]} features × {Xi.shape[1]} dims)")
    return Xi, names


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 5 — FIXED-POINT DETECTION & EIGENVALUE SPECTRA  (NEW)
# ─────────────────────────────────────────────────────────────────────────────

def sindy_rhs(x, Xi, degree):
    """
    Evaluate the SINDy ODE right-hand side at a single point x (shape: (d,)).
    Returns dx/dt of shape (d,).
    """
    X_pt     = x.reshape(-1, 1)          # (d, 1)
    Theta_pt, _ = build_polynomial_library(X_pt, degree)   # (1, n_feat)
    return (Theta_pt @ Xi).ravel()       # (d,)


def numerical_jacobian(x, Xi, degree, eps=1e-5):
    """
    Finite-difference Jacobian of the SINDy ODE at point x.
    J[i, j] = d(f_i)/d(x_j)
    """
    d  = len(x)
    J  = np.zeros((d, d))
    f0 = sindy_rhs(x, Xi, degree)
    for j in range(d):
        xp        = x.copy()
        xp[j]    += eps
        J[:, j]   = (sindy_rhs(xp, Xi, degree) - f0) / eps
    return J


def find_fixed_points(Xi, embedded, cfg):
    """
    Search for fixed points of the SINDy ODE in the neighbourhood of the
    embedded data cloud.  Returns list of (fp, eigenvalues) tuples.
    """
    degree  = cfg['SINDY_POLY_DEG']
    n_inits = cfg['FP_N_INITS']
    tol     = cfg['FP_MERGE_TOL']

    # Sample random initial guesses uniformly within data bounding box
    lo = embedded.min(axis=1)
    hi = embedded.max(axis=1)
    rng = np.random.default_rng(42)

    fps = []   # list of confirmed fixed-point vectors
    eigenvalues_all = []

    for _ in range(n_inits):
        x0   = rng.uniform(lo, hi)
        try:
            fp, info, ier, _ = fsolve(
                sindy_rhs, x0,
                args=(Xi, degree),
                full_output=True
            )
        except Exception:
            continue

        if ier != 1:
            continue

        # Verify it is actually close to zero
        residual = np.linalg.norm(sindy_rhs(fp, Xi, degree))
        if residual > 1e-6:
            continue

        # Merge duplicates
        is_new = all(np.linalg.norm(fp - fp_prev) > tol for fp_prev in fps)
        if not is_new:
            continue

        fps.append(fp)
        J    = numerical_jacobian(fp, Xi, degree)
        eigs = np.linalg.eigvals(J)
        eigenvalues_all.append(eigs)
        log.info(f"    Fixed point found: {fp.round(3)}  residual={residual:.2e}  "
                 f"eigs={eigs.real.round(3)}")

    return fps, eigenvalues_all


# ─────────────────────────────────────────────────────────────────────────────
#  PLOTTING  (original + two new figures)
# ─────────────────────────────────────────────────────────────────────────────

def plot_takens_attractor(embedded, label, out_dir, dpi):
    """3-D Takens attractor coloured by time (unchanged)."""
    if embedded is None or embedded.shape[0] < 3:
        return
    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111, projection='3d')
    n   = min(embedded.shape[1], 3000)
    x, y, z = embedded[0, :n], embedded[1, :n], embedded[2, :n]
    tc  = np.linspace(0, 1, n)
    ax.plot(x, y, z, color='gray', lw=0.5, alpha=0.5)
    sc  = ax.scatter(x, y, z, c=tc, cmap='viridis', s=2, alpha=0.8)
    ax.set_title(f"{label} — Reconstructed Takens Attractor", fontsize=11)
    ax.set_xlabel('x(t)')
    ax.set_ylabel(r'$x(t+\tau)$')
    ax.set_zlabel(r'$x(t+2\tau)$')
    plt.colorbar(sc, label='Normalised Time', ax=ax, pad=0.1)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{label}_takens_3d.png'), dpi=dpi, bbox_inches='tight')
    plt.close(fig)


# ── Figure 1: Fixed-point eigenvalue spectra PRE vs POST ─────────────────────

def plot_eigenvalue_spectra(eigs_pre, eigs_post, label, out_dir, dpi):
    """
    Complex-plane scatter of Jacobian eigenvalues at all fixed points,
    PRE (blue) vs POST (orange).  The unit circle (Re=0 line) is drawn
    as a stability boundary reference.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    def _scatter(eigs_list, color, marker, epoch_name):
        all_re, all_im = [], []
        for eigs in eigs_list:
            for e in eigs:
                all_re.append(e.real)
                all_im.append(e.imag)
        if all_re:
            ax.scatter(all_re, all_im, c=color, marker=marker, s=60,
                       alpha=0.75, edgecolors='k', linewidths=0.4,
                       label=epoch_name, zorder=3)

    _scatter(eigs_pre,  '#3B8BDE', 'o', 'PRE-sleep')
    _scatter(eigs_post, '#E8662A', 's', 'POST-sleep')

    # Stability boundary: Re(λ) = 0
    ylim = ax.get_ylim() if ax.get_ylim() != (0, 1) else (-3, 3)
    ax.axvline(0, color='gray', lw=1.2, ls='--', label='Re(λ)=0 (stability boundary)')
    ax.axhline(0, color='lightgray', lw=0.8, ls=':')

    ax.set_xlabel(r'Re($\lambda$)', fontsize=12)
    ax.set_ylabel(r'Im($\lambda$)', fontsize=12)
    ax.set_title(f'Fixed-Point Eigenvalue Spectra\nPRE vs POST — {label}', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{label}_eigenvalue_spectra.png'),
                dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    log.info(f"  Saved eigenvalue spectra → {label}_eigenvalue_spectra.png")


def plot_attractor_metrics(volume_pre, volume_post, D2_pre, D2_post, 
                          lambda1_pre, lambda1_post, label, out_dir, dpi):
    """
    Plot attractor metrics: volume, correlation dimension D2, and Lyapunov exponent.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Volume
    if volume_pre is not None and volume_post is not None:
        ax = axes[0]
        epochs = ['PRE', 'POST']
        vols = [volume_pre, volume_post]
        colors = ['steelblue', 'coral']
        ax.bar(epochs, vols, color=colors, alpha=0.7, edgecolor='black', lw=1.5)
        ax.set_ylabel('Attractor Volume')
        ax.set_title('Attractor Volume')
        ax.grid(axis='y', alpha=0.3)
        for i, v in enumerate(vols):
            ax.text(i, v * 1.02, f'{v:.3e}', ha='center', fontsize=9)
    
    # Correlation Dimension D2
    if D2_pre is not None and D2_post is not None:
        ax = axes[1]
        epochs = ['PRE', 'POST']
        dims = [D2_pre, D2_post]
        colors = ['steelblue', 'coral']
        ax.bar(epochs, dims, color=colors, alpha=0.7, edgecolor='black', lw=1.5)
        ax.set_ylabel('D2 (Correlation Dimension)')
        ax.set_title('Correlation Dimension D2')
        ax.grid(axis='y', alpha=0.3)
        for i, d in enumerate(dims):
            ax.text(i, d * 1.02, f'{d:.2f}', ha='center', fontsize=9)
    
    # Lyapunov Exponent
    if lambda1_pre is not None and lambda1_post is not None:
        ax = axes[2]
        epochs = ['PRE', 'POST']
        exps = [lambda1_pre, lambda1_post]
        colors = ['steelblue', 'coral']
        bars = ax.bar(epochs, exps, color=colors, alpha=0.7, edgecolor='black', lw=1.5)
        # Highlight if positive (chaotic)
        for bar, exp in zip(bars, exps):
            if exp > 0:
                bar.set_hatch('//')
        ax.axhline(0, color='red', ls='--', lw=1, label='Chaos threshold')
        ax.set_ylabel('Largest Lyapunov Exponent')
        ax.set_title('Lyapunov Exponent (λ1)')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        for i, l in enumerate(exps):
            ax.text(i, l * 1.02, f'{l:.3f}', ha='center', fontsize=9)
    
    fig.suptitle(f'{label} — Attractor Dynamics Metrics PRE vs POST', fontsize=12)
    plt.tight_layout()
    save_path = os.path.join(out_dir, f'{label}_attractor_metrics.png')
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    log.info(f"  Saved attractor metrics → {os.path.basename(save_path)}")


def plot_top_coefficient_changes(coef_records, out_dir, dpi):
    """
    Plot top coefficient changes across all animals and regions.
    Shows which SINDy terms changed most between PRE and POST.
    """
    if not coef_records:
        return
    
    # Aggregate all coefficient changes
    all_changes = {}  # feature_name -> list of changes
    
    for rec in coef_records:
        if rec['pre_Xi'] is None or rec['post_Xi'] is None:
            continue
        
        pre_Xi, post_Xi = rec['pre_Xi'], rec['post_Xi']
        feat_names = rec['feature_names']
        
        # For each dimension and feature, compute change
        for dim in range(pre_Xi.shape[1]):
            for feat_idx, feat_name in enumerate(feat_names):
                if feat_idx >= pre_Xi.shape[0]:
                    continue
                change = post_Xi[feat_idx, dim] - pre_Xi[feat_idx, dim]
                key = f'{feat_name}'
                if key not in all_changes:
                    all_changes[key] = []
                all_changes[key].append(abs(change))
    
    if not all_changes:
        return
    
    # Compute mean absolute change per feature
    mean_changes = {k: np.mean(v) for k, v in all_changes.items()}
    sorted_features = sorted(mean_changes.items(), key=lambda x: x[1], reverse=True)
    
    # Plot top 15
    n_top = min(15, len(sorted_features))
    top_features = sorted_features[:n_top]
    feat_names_top = [f[0] for f in top_features]
    changes_top = [f[1] for f in top_features]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(feat_names_top))
    ax.barh(y_pos, changes_top, color='steelblue', alpha=0.7, edgecolor='black', lw=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feat_names_top, fontsize=9)
    ax.set_xlabel('Mean Absolute Coefficient Change (POST − PRE)', fontsize=10)
    ax.set_title('Top SINDy Coefficient Changes\nAcross All Animals and Regions', fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    
    plt.tight_layout()
    save_path = os.path.join(out_dir, 'top_coefficient_changes.png')
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    log.info(f"  Saved top coefficient changes → {os.path.basename(save_path)}")

def plot_sindy_heatmap(coef_records, feature_names, out_dir, dpi):
    """
    coef_records : list of dicts, each with keys
        'animal', 'day', 'pre_Xi', 'post_Xi'   (both shape n_feat × d)

    Plots  ΔXi = Xi_post − Xi_pre  for the first ODE dimension (x0-dot),
    with animals×days on the y-axis and library features on the x-axis.
    """
    if not coef_records:
        log.warning("No SINDy coefficient records to plot.")
        return

    # Collect delta-coefficients for dim-0 (x0-dot)
    row_labels = []
    delta_rows = []
    for rec in coef_records:
        if rec['pre_Xi'] is None or rec['post_Xi'] is None:
            continue
        delta = rec['post_Xi'][:, 0] - rec['pre_Xi'][:, 0]
        delta_rows.append(delta)
        row_labels.append(f"{rec['animal']} | {rec['day']}")

    if not delta_rows:
        log.warning("All SINDy coefficient records are None — skipping heatmap.")
        return

    delta_mat = np.array(delta_rows)   # (n_sessions, n_features)

    # ── prune all-zero columns for readability ──────────────────────────────
    nonzero_cols = np.where(np.abs(delta_mat).max(axis=0) > 1e-8)[0]
    if len(nonzero_cols) == 0:
        log.warning("All SINDy coefficient deltas are zero — heatmap would be blank.")
        return
    delta_mat   = delta_mat[:, nonzero_cols]
    feat_labels = [feature_names[i] for i in nonzero_cols]

    # ── symmetric colour scale ───────────────────────────────────────────────
    vmax = np.abs(delta_mat).max()
    vmax = vmax if vmax > 0 else 1.0

    fig_h = max(4, 0.45 * len(row_labels) + 1.5)
    fig_w = max(6, 0.55 * len(feat_labels) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(delta_mat, aspect='auto', cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax, interpolation='nearest')

    ax.set_xticks(range(len(feat_labels)))
    ax.set_xticklabels(feat_labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(r'$\Delta\Xi$ (POST $-$ PRE)', fontsize=9)

    ax.set_title(r'SINDy Coefficient Changes ($\dot{x}_0$ equation)'
                 '\nAcross Days and Animals', fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'sindy_coefficient_heatmap.png'),
                dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    log.info(f"  Saved SINDy heatmap → sindy_coefficient_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
#  SAVE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def save_results(animal, epoch_label, day, embedded, out_dir):
    if embedded is None:
        return
    fname = os.path.join(out_dir, f'{animal}_{day}_{epoch_label}_takens.npz')
    np.savez_compressed(fname, delay_embedded_manifold=embedded)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

EPOCHS = {'pre': 0, 'post': 1}


def process_animal(animal_name, cfg, coef_records):
    """
    Process one animal.  Appends to coef_records in-place for the
    cross-animal heatmap.
    """
    folder  = cfg['DATA_FOLDER']
    pattern = os.path.join(folder, cfg['DATA_PATTERN'].replace('Animal*', animal_name))
    files   = sorted(glob.glob(pattern))

    if not files:
        log.error(f"No files: {animal_name}  pattern={pattern}")
        return

    out_dir = os.path.join(folder, cfg['OUT_FOLDER'], animal_name)
    os.makedirs(out_dir, exist_ok=True)

    log.info(f"\n{'='*68}")
    log.info(f"  {animal_name}  |  {len(files)} days")
    log.info(f"{'='*68}")

    for fpath in files:
        day = os.path.splitext(os.path.basename(fpath))[0]
        log.info(f"\n  -- {day} --")

        try:
            data = load_npz(fpath)
        except (zlib.error, ValueError) as e:
            log.error(f"  [SKIP] {day} — corrupted file: {type(e).__name__}: {e}")
            continue

        # ── collect both epochs first so we can do PRE vs POST comparison ──
        epoch_embedded = {}
        epoch_sindy    = {}
        epoch_eigs     = {}
        epoch_metrics_dict = {}

        for epoch, idx in EPOCHS.items():
            state = build_state_matrix(data, cfg, idx, epoch)
            if state is None or state.size == 0:
                epoch_embedded[epoch] = None
                epoch_sindy[epoch]    = (None, None)
                epoch_eigs[epoch]     = []
                epoch_metrics_dict[epoch] = {'volume': None, 'D2': None, 'lambda1': None}
                continue

            # 1. Takens embedding
            embedded = run_takens_embedding(state, cfg)
            epoch_embedded[epoch] = embedded

            # 2. Original 3-D plot + save .npz
            if cfg['SAVE_PLOTS'] and embedded is not None:
                plot_takens_attractor(
                    embedded,
                    f"{animal_name}_{day}_{epoch}",
                    out_dir,
                    cfg['PLOT_DPI']
                )
            save_results(animal_name, epoch, day, embedded, out_dir)

            # 3. SINDy
            Xi, feat_names = run_sindy(embedded, cfg)
            epoch_sindy[epoch] = (Xi, feat_names)

            # 4. Fixed-point eigenvalues
            if Xi is not None and embedded is not None:
                fps, eigs_list = find_fixed_points(Xi, embedded, cfg)
                epoch_eigs[epoch] = eigs_list
            else:
                epoch_eigs[epoch] = []
            
            # 5. Attractor metrics
            volume, dims = compute_attractor_volume(embedded)
            D2 = compute_correlation_dimension(embedded)
            lambda1 = compute_lyapunov_exponent(Xi, embedded, cfg)
            epoch_metrics_dict[epoch] = {'volume': volume, 'D2': D2, 'lambda1': lambda1}

        # ── Figure 1: eigenvalue spectra PRE vs POST (per day) ──────────────
        if cfg['SAVE_PLOTS']:
            plot_eigenvalue_spectra(
                epoch_eigs.get('pre', []),
                epoch_eigs.get('post', []),
                label=f"{animal_name}_{day}",
                out_dir=out_dir,
                dpi=cfg['PLOT_DPI']
            )
        
        # ── Figure 1b: attractor metrics PRE vs POST (per day) ──────────────
        metrics_pre = epoch_metrics_dict.get('pre', {})
        metrics_post = epoch_metrics_dict.get('post', {})
        if cfg['SAVE_PLOTS'] and (metrics_pre or metrics_post):
            plot_attractor_metrics(
                metrics_pre.get('volume'), metrics_post.get('volume'),
                metrics_pre.get('D2'), metrics_post.get('D2'),
                metrics_pre.get('lambda1'), metrics_post.get('lambda1'),
                label=f"{animal_name}_{day}",
                out_dir=out_dir,
                dpi=cfg['PLOT_DPI']
            )

        # ── Accumulate SINDy records for the cross-animal heatmap ───────────
        pre_Xi,  pre_names  = epoch_sindy.get('pre',  (None, None))
        post_Xi, post_names = epoch_sindy.get('post', (None, None))

        feat_names = pre_names if pre_names is not None else post_names
        coef_records.append({
            'animal':  animal_name,
            'day':     day,
            'pre_Xi':  pre_Xi,
            'post_Xi': post_Xi,
            'feature_names': feat_names,
        })

    log.info(f"\n  Done: {animal_name}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    cfg = CONFIG.copy()

    # coef_records accumulates across all animals for the heatmap
    coef_records = []

    if len(sys.argv) > 1:
        animal = sys.argv[1]
        cfg['DATA_PATTERN'] = f'{animal}_Day*.npz'
        process_animal(animal, cfg, coef_records)
    else:
        all_files = sorted(glob.glob(os.path.join(cfg['DATA_FOLDER'], cfg['DATA_PATTERN'])))
        animals   = sorted(set(os.path.basename(f).split('_')[0] for f in all_files))
        for a in animals:
            process_animal(a, cfg, coef_records)

    # ── Figure 2: cross-animal / cross-day heatmap ──────────────────────────
    # Use feature names from whichever record has them
    all_feat_names = next(
        (r['feature_names'] for r in coef_records if r['feature_names'] is not None),
        []
    )
    out_root = os.path.join(cfg['DATA_FOLDER'], cfg['OUT_FOLDER'])
    os.makedirs(out_root, exist_ok=True)
    plot_sindy_heatmap(coef_records, all_feat_names, out_root, cfg['PLOT_DPI'])
    
    # ── Figure 2b: top coefficient changes across all animals/regions ──────
    plot_top_coefficient_changes(coef_records, out_root, cfg['PLOT_DPI'])

    log.info("\nAll done.")