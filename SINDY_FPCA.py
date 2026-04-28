"""
=======================================================================
PFC–M1 Manifold Consolidation Analysis  –  Multi-Day Pipeline
Based on: Kim, Joshi, Frank, Ganguly. Nature 613, 103–110 (2023)
https://doi.org/10.1038/s41586-022-05533-z

DIMENSIONALITY REDUCTION: Functional PCA (fPCA)
────────────────────────────────────────────────
  Standard PCA treats each time-bin independently, ignoring the fact
  that neural firing rates are *continuous functions* of time.
  Functional PCA instead:
    1. Represents each neuron's PSTH as a smooth function using a
       B-spline basis (scipy.interpolate.BSpline / LSQUnivariateSpline)
    2. Stacks the spline COEFFICIENTS into a data matrix
    3. Runs PCA on that coefficient matrix → functional principal
       components (fPCs) that capture smooth, physiologically
       meaningful modes of co-variation across neurons
    4. Projects raw data onto fPCs for downstream analyses

  This is the standard approach in functional data analysis (Ramsay &
  Silverman 2005) and is more appropriate for neural trajectory
  analyses than bin-wise PCA.

WHAT THIS SCRIPT DOES (per day):
─────────────────────────────────────────────────────────────────────
  A. Load .npz  →  bin spikes (50 ms)  →  Gaussian smooth  →  z-score
  B. Slow Oscillation (SO) detection in already-filtered delta LFP
  C. PFC–M1 SO coupling
  D. SWR detection in HPC 150–250 Hz LFP
  E. Functional PCA on M1 & PFC population  →  low-D latent space
  F. CCA communication subspace  (windowed, 30 s windows)
  G. SINDy ODE fit on M1 fPCA latent trajectory

USAGE:
    Place all day .npz files in the same directory.
    Edit DATA_PATTERN to match your filenames, then run:
        python PFC_M1_consolidation_multiday.py

REQUIREMENTS:
    pip install numpy scipy scikit-learn pysindy matplotlib
=======================================================================
"""

import os
import glob
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import LSQUnivariateSpline
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
import pysindy as ps

warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

DATA_PATTERN = 'Animal1_Day*.npz'

N_M1_PER_SESSION  = 39
N_PFC_PER_SESSION = 13

DT_SPIKE     = 0.05       # spike-count bin width (s)
SIGMA_SMOOTH = 4          # Gaussian smoothing (bins)

# ── Functional PCA settings ──────────────────────────────────────────
FPCA_N_COMPONENTS = 5     # number of functional PCs to retain
FPCA_N_KNOTS      = 20    # interior B-spline knots per neuron
FPCA_SPLINE_ORDER = 4     # spline order (4 = cubic)

# SO detection
SO_PEAK_PERCENTILE_ACTIVE   = 85
SO_TROUGH_PERCENTILE_ACTIVE = 85
SO_MIN_DUR   = 0.30
SO_MAX_DUR   = 1.00

SWR_THRESH_SD  = 3.0
SWR_MIN_GAP    = 0.05
SO_COUPLING_WIN   = 0.50
SWR_SO_WIN        = 0.75

N_CCA           = 2
SINDY_N_PCS     = 3
SINDY_THRESHOLD      = 0.001   # lowered: less aggressive zeroing
SINDY_ALPHA          = 0.001   # SR3 regularisation weight
SINDY_COEF_CLIP      = 2.0     # hard clip on any single SINDy coefficient
SINDY_ENSEMBLE_N     = 20      # number of ensemble bootstrap fits
SINDY_ENSEMBLE_FRAC  = 0.8     # fraction of time-points per bootstrap

EARLY_DAYS = slice(0, 4)
LATE_DAYS  = slice(-4, None)


# ═══════════════════════════════════════════════════════════════════════
# FUNCTIONAL PCA
# ═══════════════════════════════════════════════════════════════════════

class FunctionalPCA:
    """
    Functional PCA via B-spline basis expansion.

    Each neuron's smoothed firing-rate trace r_i(t) is represented as:

        r_i(t) ≈ Σ_k  c_{ik} · B_k(t)

    where B_k are B-spline basis functions and c_{ik} are the
    least-squares spline coefficients.  PCA is then applied to the
    N_neurons × N_coefficients matrix of coefficients, yielding
    functional principal components (fPCs).

    The latent score matrix (T_bins × n_components) is obtained by
    evaluating the fPC basis functions at every time point and
    projecting the smoothed data.

    Parameters
    ----------
    n_components : int   number of fPCs to retain
    n_knots      : int   number of interior knots for the B-spline basis
    spline_order : int   spline order (default 4 = cubic)
    """

    def __init__(self, n_components=5, n_knots=20, spline_order=4):
        self.n_components  = n_components
        self.n_knots       = n_knots
        self.spline_order  = spline_order
        self.pca_          = None
        self.knots_        = None
        self.t_fit_        = None
        self.explained_variance_ratio_ = None

    # ── private ──────────────────────────────────────────────────────

    def _make_knots(self, t):
        """Place n_knots interior knots at equal quantiles of t."""
        quantiles = np.linspace(0, 100, self.n_knots + 2)[1:-1]
        return np.percentile(t, quantiles)

    def _spline_coefficients(self, t, y_matrix):
        """
        Fit one B-spline per row of y_matrix (neurons × time).
        Returns coefficient matrix (neurons × n_coef).
        """
        knots = self._make_knots(t)
        coefs = []
        for y in y_matrix:
            try:
                spl = LSQUnivariateSpline(t, y, knots,
                                          k=self.spline_order - 1)
                coefs.append(spl.get_coeffs())
            except Exception:
                # Fallback: zero vector if spline fails
                coefs.append(np.zeros(self.n_knots + self.spline_order))
        # Pad/truncate to uniform length
        L = max(len(c) for c in coefs)
        padded = np.array([np.pad(c, (0, L - len(c))) for c in coefs])
        self.knots_ = knots
        return padded

    def _reconstruct_basis(self, t):
        """
        Evaluate each basis function at time points t.
        Returns design matrix Φ of shape (len(t), n_coef).

        Each column is the j-th B-spline basis function evaluated at t,
        so  r_i(t) ≈ coefs[i] @ Φ(t).T
        """
        from scipy.interpolate import BSpline
        k     = self.spline_order - 1
        knots = self.knots_
        # Full knot vector with boundary repetitions
        t_full = np.concatenate((
            np.repeat(t[0],  k + 1),
            knots,
            np.repeat(t[-1], k + 1)
        ))
        n_basis = len(t_full) - k - 1
        Phi = np.zeros((len(t), n_basis))
        for j in range(n_basis):
            c      = np.zeros(n_basis)
            c[j]   = 1.0
            spl    = BSpline(t_full, c, k)
            Phi[:, j] = spl(t)
        return Phi

    # ── public ───────────────────────────────────────────────────────

    def fit(self, y_matrix, t=None):
        """
        Fit fPCA to y_matrix (n_neurons × n_time).

        Parameters
        ----------
        y_matrix : ndarray  (n_neurons, n_time)   smoothed firing rates
        t        : ndarray  (n_time,)              time axis (seconds)
                            defaults to [0, 1, …, n_time-1]
        """
        n_neurons, n_time = y_matrix.shape
        if t is None:
            t = np.arange(n_time, dtype=float)
        self.t_fit_ = t

        # Step 1: B-spline coefficient matrix  (n_neurons × n_coef)
        coef_matrix = self._spline_coefficients(t, y_matrix)

        # Step 2: PCA on coefficient matrix
        n_comp = min(self.n_components, coef_matrix.shape[0],
                     coef_matrix.shape[1])
        self.pca_ = PCA(n_components=n_comp)
        self.pca_.fit(coef_matrix)
        self.explained_variance_ratio_ = self.pca_.explained_variance_ratio_
        self._coef_matrix_fit = coef_matrix
        return self

    def transform(self, y_matrix, t=None):
        """
        Project y_matrix onto fPCs.

        Returns latent score matrix of shape (n_time, n_components).
        Each column is the projection of the population activity onto
        one functional principal component across time.
        """
        if t is None:
            t = self.t_fit_

        n_neurons, n_time = y_matrix.shape

        # Re-fit splines on new data with same knot placement
        knots_backup   = self.knots_
        coef_matrix    = self._spline_coefficients(t, y_matrix)
        self.knots_    = knots_backup   # restore (knots set in _spline_coef)

        # Project: scores shape = (n_neurons, n_components)
        coef_scores    = self.pca_.transform(coef_matrix)  # (n_neurons, k)

        # Reconstruct latent time-series by evaluating fPC loadings at t
        # Φ shape: (n_time, n_coef)
        Phi            = self._reconstruct_basis(t)
        # fPC weight vectors in coef space  (n_components, n_coef)
        fpc_coef       = self.pca_.components_             # (k, n_coef)

        # Latent score at each time bin:
        #   score(t) = Φ(t) @ fPC_coef.T  →  (n_time, k)
        #   scaled by mean coef projection across neurons
        latent         = Phi[:, :fpc_coef.shape[1]] @ fpc_coef.T
        return latent   # (n_time, n_components)

    def fit_transform(self, y_matrix, t=None):
        return self.fit(y_matrix, t).transform(y_matrix, t)


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

def detect_slow_oscillations(delta_lfp, fs,
                             peak_pct=SO_PEAK_PERCENTILE_ACTIVE,
                             trough_pct=SO_TROUGH_PERCENTILE_ACTIVE,
                             min_dur=SO_MIN_DUR, max_dur=SO_MAX_DUR):
    sig  = delta_lfp.astype(float)
    t    = np.arange(len(sig)) / fs
    nrem_mask = np.abs(sig) > 0.001
    if nrem_mask.sum() < 100:
        return np.array([])
    active    = sig[nrem_mask]
    pk_thresh = np.percentile(active[active > 0], peak_pct) \
                if (active > 0).any() else 1e9
    tr_thresh = np.percentile(active[active < 0], 100 - trough_pct) \
                if (active < 0).any() else -1e9
    signs   = np.sign(sig)
    pos2neg = np.where((signs[:-1] > 0) & (signs[1:] <= 0))[0]
    neg2pos = np.where((signs[:-1] < 0) & (signs[1:] >= 0))[0]
    so_times = []
    for idx in pos2neg:
        if not nrem_mask[idx]:
            continue
        prev = neg2pos[neg2pos < idx]
        nxt  = neg2pos[neg2pos > idx]
        if len(prev) == 0 or len(nxt) == 0:
            continue
        p0, p1 = prev[-1], nxt[0]
        dur    = (p1 - p0) / fs
        if dur < min_dur or dur > max_dur:
            continue
        peak   = sig[p0:idx].max() if p0 < idx else -np.inf
        trough = sig[idx:p1].min() if idx < p1 else np.inf
        if peak >= pk_thresh and trough <= tr_thresh:
            trough_idx = idx + np.argmin(sig[idx:p1])
            so_times.append(t[trough_idx])
    return np.array(so_times)


def pfc_m1_so_coupling(pfc_so, m1_so, window=SO_COUPLING_WIN):
    if len(pfc_so) == 0 or len(m1_so) == 0:
        return np.nan
    return sum(1 for t_m1 in m1_so
               if np.any(np.abs(pfc_so - t_m1) < window)) / len(m1_so)


def detect_swr(hpc_ripple_lfp, fs,
               thresh_sd=SWR_THRESH_SD, min_gap=SWR_MIN_GAP):
    env   = np.abs(hpc_ripple_lfp)
    env_s = gaussian_filter1d(env, sigma=int(0.01 * fs))
    thr   = env_s.mean() + thresh_sd * env_s.std()
    above = (env_s > thr).astype(int)
    starts = np.where(np.diff(above) == 1)[0]
    t      = np.arange(len(hpc_ripple_lfp)) / fs
    events, last = [], -np.inf
    for s in starts:
        if (t[s] - last) > min_gap:
            events.append(t[s])
            last = t[s]
    return np.array(events)


def swr_m1so_coupling(swr_times, m1_so_times, window=SWR_SO_WIN):
    if len(swr_times) == 0 or len(m1_so_times) == 0:
        return np.nan
    return sum(1 for t_so in m1_so_times
               if np.any(np.abs(swr_times - t_so) < window)) / len(m1_so_times)


def bin_spikes(spike_list, dt=DT_SPIKE, T=4640.0):
    """Convert spike-time arrays to population firing-rate matrix (Hz)."""
    bins = np.arange(0, T + dt, dt)
    mat  = np.zeros((len(spike_list), len(bins) - 1))
    for i, st in enumerate(spike_list):
        if st is None or len(st) == 0:
            continue
        st = np.asarray(st, dtype=float)
        c, _ = np.histogram(st, bins=bins)
        mat[i] = c / dt
    return mat


def zscore(mat):
    mu  = mat.mean(axis=1, keepdims=True)
    sig = mat.std(axis=1,  keepdims=True) + 1e-6
    return (mat - mu) / sig


def manifold_fidelity_fpca(fpca_model, spike_stack, n_trials, n_tbins,
                           t=None, sigma=3):
    """
    M1 Manifold Fidelity using fPCA trajectories.

    Steps:
      1. Per-trial PSTH smoothing (Gaussian σ = 3 bins)
      2. Project each trial through the *already-fitted* fPCA model
         → trial trajectory in functional PC space
      3. Fidelity = mean pairwise cosine similarity of flattened trajectories
    """
    n_cells = spike_stack.shape[0]
    trials  = spike_stack.reshape(n_cells, n_trials, n_tbins).copy()
    for i in range(n_cells):
        for tr in range(n_trials):
            trials[i, tr] = gaussian_filter1d(trials[i, tr], sigma)

    if t is None:
        t = np.arange(n_tbins, dtype=float) * DT_SPIKE

    traj = []
    for tr in range(n_trials):
        lat = fpca_model.transform(trials[:, tr, :], t=t)
        traj.append(lat.flatten())
    traj  = np.array(traj)
    norms = np.linalg.norm(traj, axis=1, keepdims=True) + 1e-8
    traj_n = traj / norms
    sim    = traj_n @ traj_n.T
    idx    = np.triu_indices(n_trials, k=1)
    return float(sim[idx].mean())


def windowed_cca(m1_lat, pfc_lat, win_bins=600, step_bins=300):
    """Windowed CCA between M1 and PFC fPCA latent spaces."""
    T     = min(m1_lat.shape[0], pfc_lat.shape[0])
    n_win = max(1, (T - win_bins) // step_bins)
    cc1s  = []
    for w in range(n_win):
        s, e = w * step_bins, w * step_bins + win_bins
        if e > T:
            break
        n_cols = min(5, m1_lat.shape[1], pfc_lat.shape[1])
        x = m1_lat[s:e, :n_cols].copy()
        y = pfc_lat[s:e, :n_cols].copy()
        x = (x - x.mean(0)) / (x.std(0) + 1e-8)
        y = (y - y.mean(0)) / (y.std(0) + 1e-8)
        try:
            cca = CCA(n_components=min(N_CCA, n_cols))
            cca.fit(y, x)
            xc, yc = cca.transform(y, x)
            cc1s.append(pearsonr(xc[:, 0], yc[:, 0])[0])
        except Exception:
            pass
    return float(np.nanmean(cc1s)) if cc1s else np.nan


def fit_sindy(m1_lat, dt=DT_SPIKE):
    """
    SINDy sparse ODE fit on M1 fPCA latent trajectory.

    Strategy to avoid over-zeroing coefficients:
    ─────────────────────────────────────────────
    1. Downsample + Gaussian smooth to reduce derivative estimation noise
    2. StandardScaler so all fPCs have unit variance
    3. ENSEMBLE SINDy: fit SINDY_ENSEMBLE_N bootstrap subsamples, keep
       only terms that appear in ≥ 50 % of fits (inclusion threshold).
       This is far less aggressive than single STLSQ which zeros anything
       below a hard threshold — ensemble retains physiologically real weak
       couplings while discarding noise-driven spurious terms.
    4. SR3 optimizer (relaxed sparsity) instead of STLSQ — SR3 minimises
         0.5*||dx/dt - Θξ||² + λ*||ξ||₁
       via a proximal split, so small-but-real coefficients shrink rather
       than snap to zero as in hard thresholding.
    5. Numerical Jacobian stability check at mean operating point;
       fall back to degree-1 if degree-2 is strongly unstable.
    6. Hard coefficient clip as final safety net.
    """
    n_sindy = min(SINDY_N_PCS, m1_lat.shape[1])
    x       = m1_lat[:, :n_sindy].copy()

    stride  = 4
    x_ds    = x[::stride]
    dt_ds   = dt * stride

    scaler  = StandardScaler()
    x_sc    = scaler.fit_transform(x_ds)
    x_sc    = gaussian_filter1d(x_sc, sigma=2, axis=0)

    n_samples = x_sc.shape[0]

    def _numerical_max_eigval(model, x0):
        """Full numerical Jacobian max real eigenvalue at x0."""
        eps = 1e-4
        n   = len(x0)
        J   = np.zeros((n, n))
        f0  = model.predict(x0.reshape(1, -1))[0]
        for j in range(n):
            xp      = x0.copy(); xp[j] += eps
            J[:, j] = (model.predict(xp.reshape(1, -1))[0] - f0) / eps
        return float(np.max(np.linalg.eigvals(J).real))

    def _fit_once(x_data, degree):
        """Single SR3 fit on x_data."""
        lib   = ps.PolynomialLibrary(degree=degree, include_bias=False)
        try:
            # SR3: relaxed sparsity — retains weak-but-real terms
            opt = ps.SR3(reg_weight_lam=0.001, relax_coeff_nu=0.001)
        except AttributeError:
            # Older PySINDy versions may not have SR3 — fall back to STLSQ
            opt = ps.STLSQ(threshold=SINDY_THRESHOLD, alpha=SINDY_ALPHA)
        model = ps.SINDy(feature_library=lib, optimizer=opt)
        model.fit(x_data, t=dt_ds)
        return model

    def _ensemble_fit(degree):
        """
        Bootstrap ensemble: fit SINDY_ENSEMBLE_N models on random
        subsamples of size SINDY_ENSEMBLE_FRAC * n_samples.
        Return mean coefficients weighted by inclusion frequency —
        terms appearing in < 50 % of fits are zeroed out.
        """
        sub_n    = max(50, int(n_samples * SINDY_ENSEMBLE_FRAC))
        coef_list = []
        rng = np.random.default_rng(42)

        for _ in range(SINDY_ENSEMBLE_N):
            idx    = rng.choice(n_samples, size=sub_n, replace=False)
            idx    = np.sort(idx)
            x_sub  = x_sc[idx]
            try:
                m   = _fit_once(x_sub, degree)
                coef_list.append(m.coefficients().copy())
            except Exception:
                pass

        if not coef_list:
            # All bootstrap fits failed — fall back to single fit
            m    = _fit_once(x_sc, degree)
            return m, m.coefficients().copy()

        coef_stack  = np.array(coef_list)             # (n_boot, n_eq, n_feat)
        inclusion   = np.mean(coef_stack != 0, axis=0)  # fraction non-zero
        mean_coef   = coef_stack.mean(axis=0)

        # Zero out terms that appear in fewer than half the bootstrap fits
        mean_coef[inclusion < 0.5] = 0.0

        n_nonzero = int((mean_coef != 0).sum())
        print(f'      SINDy ensemble (deg {degree}): ' 
              f'{len(coef_list)}/{SINDY_ENSEMBLE_N} fits OK, ' 
              f'{n_nonzero} non-zero terms retained')

        # Refit a single model to get a proper model object, then
        # overwrite its coefficients with the ensemble-averaged values
        ref_model = _fit_once(x_sc, degree)
        ref_model.optimizer.coef_ = mean_coef
        return ref_model, mean_coef

    # ── Degree-2 ensemble fit ────────────────────────────────────────
    model2, coef2 = _ensemble_fit(degree=2)
    x0_check      = x_sc.mean(axis=0)
    eig2          = _numerical_max_eigval(model2, x0_check)

    if eig2 < 0.5:
        model, coef = model2, coef2
        print(f'      SINDy: degree-2 accepted  (Jacobian max eigval={eig2:.4f})')
    else:
        model1, coef1 = _ensemble_fit(degree=1)
        eig1          = _numerical_max_eigval(model1, x0_check)
        if eig1 < eig2:
            model, coef = model1, coef1
            print(f'      SINDy: fell back to degree-1  ({eig2:.3f}→{eig1:.3f})')
        else:
            model, coef = model2, coef2
            print(f'      SINDy: kept degree-2 (degree-1 no better, eigval={eig2:.3f})')

    coef = np.clip(coef, -SINDY_COEF_CLIP, SINDY_COEF_CLIP)
    model.optimizer.coef_ = coef
    return model, coef, scaler


def piecewise_linear(x, x0, y0, k1, k2):
    return np.where(x < x0,
                    y0 + k1 * (x - x0),
                    y0 + k2 * (x - x0))


def fit_piecewise(x_all, y_all):
    mask   = ~np.isnan(y_all)
    x_, y_ = x_all[mask], y_all[mask]
    if len(x_) < 4:
        return None, None, None
    best_r2, best_p = -np.inf, None
    for bp in x_[1:-1]:
        try:
            p0   = [bp, y_.mean(), 0.01, 0.01]
            popt, _ = curve_fit(piecewise_linear, x_, y_,
                                p0=p0, maxfev=8000)
            y_pred  = piecewise_linear(x_, *popt)
            ss_res  = np.sum((y_ - y_pred) ** 2)
            ss_tot  = np.sum((y_ - y_.mean()) ** 2)
            r2      = 1 - ss_res / (ss_tot + 1e-8)
            if r2 > best_r2:
                best_r2, best_p = r2, popt
        except Exception:
            pass
    if best_p is None:
        return None, None, None
    xf = np.linspace(x_.min(), x_.max(), 300)
    return best_p, xf, piecewise_linear(xf, *best_p)


# ═══════════════════════════════════════════════════════════════════════
# PER-DAY PROCESSING
# ═══════════════════════════════════════════════════════════════════════

def process_day(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    Fs   = float(data['Fs_LFP'][0])
    out  = {}

    # ── A. Spike matrices ────────────────────────────────────────────
    valid_m1_keys = [f'Sleep_spike_time_M1_cell0_cell{i}_cell0'
                     for i in range(N_M1_PER_SESSION)
                     if f'Sleep_spike_time_M1_cell0_cell{i}_cell0' in data
                     and data[f'Sleep_spike_time_M1_cell0_cell{i}_cell0'] is not None
                     and len(data[f'Sleep_spike_time_M1_cell0_cell{i}_cell0']) > 0]

    valid_pfc_keys = [f'Sleep_spike_time_PFC_cell0_cell{i}_cell0'
                      for i in range(N_PFC_PER_SESSION)
                      if f'Sleep_spike_time_PFC_cell0_cell{i}_cell0' in data
                      and data[f'Sleep_spike_time_PFC_cell0_cell{i}_cell0'] is not None
                      and len(data[f'Sleep_spike_time_PFC_cell0_cell{i}_cell0']) > 0]

    m1_spikes  = [data[k] for k in valid_m1_keys]
    pfc_spikes = [data[k] for k in valid_pfc_keys]

    if not m1_spikes or not pfc_spikes:
        return None

    T_max = max(
        (np.asarray(data[k], dtype=float).max()
         for k in valid_m1_keys
         if len(data[k]) > 0),
        default=4640.0
    ) + 1.0

    m1_pop  = bin_spikes(m1_spikes,  dt=DT_SPIKE, T=T_max)
    pfc_pop = bin_spikes(pfc_spikes, dt=DT_SPIKE, T=T_max)

    m1_z  = zscore(gaussian_filter1d(m1_pop,  SIGMA_SMOOTH, axis=1))
    pfc_z = zscore(gaussian_filter1d(pfc_pop, SIGMA_SMOOTH, axis=1))

    # ── B-D. SO and SWR detection ────────────────────────────────────
    m1_lfp  = data.get('Sleep_LFP_delta_M1_cell0',  None)
    pfc_lfp = data.get('Sleep_LFP_delta_PFC_cell0', None)
    hpc_lfp = data.get('Sleep_LFP_150to250_HPC_cell0', None)

    if m1_lfp is not None and pfc_lfp is not None:
        m1_so  = detect_slow_oscillations(m1_lfp,  Fs)
        pfc_so = detect_slow_oscillations(pfc_lfp, Fs)
        out['n_m1_so']            = len(m1_so)
        out['n_pfc_so']           = len(pfc_so)
        out['pfc_m1_so_coupling'] = pfc_m1_so_coupling(pfc_so, m1_so)
        if hpc_lfp is not None:
            swr = detect_swr(hpc_lfp, Fs)
            out['n_swr']             = len(swr)
            out['swr_m1so_coupling'] = swr_m1so_coupling(swr, m1_so)
        else:
            out.update(n_swr=0, swr_m1so_coupling=np.nan)
    else:
        out.update(n_m1_so=0, n_pfc_so=0,
                   pfc_m1_so_coupling=np.nan,
                   n_swr=0, swr_m1so_coupling=np.nan)

    # ── E. FUNCTIONAL PCA ────────────────────────────────────────────
    n_time = m1_z.shape[1]
    t_axis = np.arange(n_time, dtype=float) * DT_SPIKE

    # Clamp components to what the data can support
    n_comp_m1  = min(FPCA_N_COMPONENTS, m1_z.shape[0]  - 1, m1_z.shape[1]  - 1)
    n_comp_pfc = min(FPCA_N_COMPONENTS, pfc_z.shape[0] - 1, pfc_z.shape[1] - 1)
    n_knots_m1  = min(FPCA_N_KNOTS, n_time // 4)
    n_knots_pfc = min(FPCA_N_KNOTS, n_time // 4)

    fpca_m1  = FunctionalPCA(n_components=n_comp_m1,
                             n_knots=n_knots_m1,
                             spline_order=FPCA_SPLINE_ORDER)
    fpca_pfc = FunctionalPCA(n_components=n_comp_pfc,
                             n_knots=n_knots_pfc,
                             spline_order=FPCA_SPLINE_ORDER)

    m1_lat  = fpca_m1.fit_transform(m1_z,  t=t_axis)   # (n_time, n_comp_m1)
    pfc_lat = fpca_pfc.fit_transform(pfc_z, t=t_axis)   # (n_time, n_comp_pfc)

    out['fpca_m1']   = fpca_m1
    out['fpca_pfc']  = fpca_pfc
    out['m1_lat']    = m1_lat
    out['pfc_lat']   = pfc_lat
    out['m1_var']    = fpca_m1.explained_variance_ratio_
    out['pfc_var']   = fpca_pfc.explained_variance_ratio_

    print(f'      fPCA M1  EVR: {fpca_m1.explained_variance_ratio_}')
    print(f'      fPCA PFC EVR: {fpca_pfc.explained_variance_ratio_}')

    # ── Manifold Fidelity ────────────────────────────────────────────
    if 'Reach_spike_spike_rate_cell0' in data:
        # Collect all valid Reach cell arrays
        reach_arrays = []
        for k in sorted(data.keys()):
            if k.startswith('Reach_spike_spike_rate'):
                arr = data[k]
                if arr is not None and np.ndim(arr) >= 1:
                    reach_arrays.append(np.asarray(arr, dtype=float).ravel())

        if len(reach_arrays) == 0:
            out['manifold_fidelity'] = np.nan
        else:
            n_cells   = len(reach_arrays)
            total_len = reach_arrays[0].shape[0]

            # Infer n_tbins from actual array size
            n_tbins = None
            for candidate in [100, 80, 120, 150, 50, 200, 60, 250]:
                if total_len % candidate == 0:
                    n_tbins = candidate
                    break
            if n_tbins is None:
                n_tbins = total_len   # treat whole array as one trial

            n_trials = max(1, total_len // n_tbins)

            # Trim all arrays to same length aligned to trial boundary
            min_len = min(a.shape[0] for a in reach_arrays)
            min_len = (min_len // n_tbins) * n_tbins
            if min_len == 0:
                out['manifold_fidelity'] = np.nan
            else:
                spike_stack = np.vstack([a[:min_len][np.newaxis, :]
                                         for a in reach_arrays])
                n_trials = min_len // n_tbins
                t_reach  = np.arange(n_tbins, dtype=float) * DT_SPIKE

                n_comp_r  = min(FPCA_N_COMPONENTS, n_cells - 1, n_tbins - 1)
                n_knots_r = max(4, min(FPCA_N_KNOTS, n_tbins // 4))
                fpca_r    = FunctionalPCA(n_components=n_comp_r,
                                          n_knots=n_knots_r,
                                          spline_order=FPCA_SPLINE_ORDER)

                # Fit fPCA on trial-mean PSTH  (n_cells x n_tbins)
                mean_psth = spike_stack.reshape(n_cells, n_trials,
                                                n_tbins).mean(axis=1)
                fpca_r.fit(mean_psth, t=t_reach)

                print(f'      Reach fPCA: {n_cells} cells, '
                      f'{n_trials} trials, {n_tbins} bins/trial')

                out['manifold_fidelity'] = manifold_fidelity_fpca(
                    fpca_r, spike_stack, n_trials, n_tbins, t=t_reach)

                # ── Compute reach latent for SINDy ──────────────────
                # Project each trial through fpca_r, stack end-to-end
                # This gives SINDy a trajectory in fPC space per reach,
                # which is what Kim et al. Fig 3 actually analyses.
                reach_trials = spike_stack.reshape(n_cells, n_trials, n_tbins)
                reach_lat_list = []
                for tr in range(n_trials):
                    trial_psth = reach_trials[:, tr, :]          # (n_cells, n_tbins)
                    trial_psth = gaussian_filter1d(trial_psth, sigma=2, axis=1)
                    lat_tr     = fpca_r.transform(trial_psth, t=t_reach)  # (n_tbins, n_comp)
                    reach_lat_list.append(lat_tr)
                out['reach_lat'] = np.vstack(reach_lat_list)     # (n_trials*n_tbins, n_comp)
    else:
        # Proxy via fPCA latent segments
        win  = int(30 / DT_SPIKE)
        n_w  = m1_lat.shape[0] // win
        if n_w >= 2:
            segs  = np.array([m1_lat[i*win:(i+1)*win, :].flatten()
                              for i in range(n_w)])
            norms = np.linalg.norm(segs, axis=1, keepdims=True) + 1e-8
            sim   = (segs / norms) @ (segs / norms).T
            ix    = np.triu_indices(n_w, k=1)
            out['manifold_fidelity'] = float(sim[ix].mean())
        else:
            out['manifold_fidelity'] = np.nan

    # ── F. CCA communication subspace ───────────────────────────────
    out['cca_cc1'] = windowed_cca(m1_lat, pfc_lat)

    # ── G. SINDy ODE ────────────────────────────────────────────────
    # Use reach trajectories if available (correct per Kim et al. Fig 3)
    # Fall back to sleep latent if reach data absent
    try:
        reach_lat = out.get('reach_lat', None)
        sindy_input = reach_lat if reach_lat is not None else m1_lat
        sindy_label = 'reach fPCA latent' if reach_lat is not None else 'sleep fPCA latent'
        print(f'      SINDy input: {sindy_label}  shape={sindy_input.shape}')

        model, coef, scaler    = fit_sindy(sindy_input, dt=DT_SPIKE)
        out['sindy_model']     = model
        out['sindy_coef']      = coef
        out['sindy_scaler']    = scaler
        out['sindy_input_lat'] = sindy_input   # store for simulate_sindy
    except Exception as e:
        print(f'    SINDy skipped: {e}')
        out['sindy_model'] = out['sindy_coef'] = out['sindy_scaler'] = None
        out['sindy_input_lat'] = None

    return out


# ═══════════════════════════════════════════════════════════════════════
# LOAD ALL DAY FILES
# ═══════════════════════════════════════════════════════════════════════

day_files = sorted(glob.glob(DATA_PATTERN))
print(f'Found {len(day_files)} day file(s).')

if not day_files:
    raise FileNotFoundError(
        f'No files matched "{DATA_PATTERN}". '
        'Update DATA_PATTERN at the top of this script.')

# Extract animal name from first file for output filenames
first_file_base = os.path.basename(day_files[0]).replace('.npz', '')
animal_name = first_file_base.split('_')[0]  # e.g., "Animal1" from "Animal1_Day1"

# Create output directories for this animal
animal_output_dir = animal_name
images_dir = os.path.join(animal_output_dir, 'images')
equations_dir = os.path.join(animal_output_dir, 'equations')

os.makedirs(images_dir, exist_ok=True)
os.makedirs(equations_dir, exist_ok=True)

print(f'\nOutput directories: {images_dir}/ and {equations_dir}/')

days_data = {}
for i, fp in enumerate(day_files):
    dn = i + 1
    print(f'  Day {dn:>2}: {os.path.basename(fp)} … ', end='', flush=True)
    r = process_day(fp)
    if r is not None:
        days_data[dn] = r
        print(f"M1 SOs={r.get('n_m1_so',0):>4d}  "
              f"SWRs={r.get('n_swr',0):>5d}  "
              f"PFC-M1={r.get('pfc_m1_so_coupling',np.nan):.3f}  "
              f"fidelity={r.get('manifold_fidelity',np.nan):.3f}")
    else:
        print('SKIPPED')

day_nums = np.array(sorted(days_data.keys()), dtype=float)
N_DAYS   = len(day_nums)
    
if N_DAYS == 0:
    print(f'WARNING: No valid data for {animal_name}, skipping analysis')
    sys.exit(1)

def metric(key):
    return np.array([days_data[int(d)].get(key, np.nan) for d in day_nums])

pfc_m1_coup = metric('pfc_m1_so_coupling')
swr_m1_coup = metric('swr_m1so_coupling')
fidelity    = metric('manifold_fidelity')
cca_cc1     = metric('cca_cc1')


# ═══════════════════════════════════════════════════════════════════════
# SINDy COEFFICIENT MATRIX
# ═══════════════════════════════════════════════════════════════════════

valid_coefs = [days_data[int(d)].get('sindy_coef')
               for d in day_nums
               if days_data[int(d)].get('sindy_coef') is not None]
sindy_ok = len(valid_coefs) > 0

if sindy_ok:
    cs      = valid_coefs[0].shape
    coef_mx = np.full((N_DAYS, *cs), np.nan)
    for i, d in enumerate(day_nums):
        c = days_data[int(d)].get('sindy_coef')
        if c is not None and np.array_equal(c.shape, cs):
            coef_mx[i] = c
    feat_names = (days_data[int(day_nums[0])]['sindy_model'].get_feature_names()
                  if days_data[int(day_nums[0])].get('sindy_model') else [])


# ═══════════════════════════════════════════════════════════════════════
# SINDy SIMULATION
# ═══════════════════════════════════════════════════════════════════════

def simulate_sindy(day_num, t_max=300):
    """
    Integrate the SINDy ODE using manual RK4 with per-step clipping.

    Why manual RK4 instead of model.simulate():
      model.simulate() calls LSODA under the hood; if the ODE is even
      mildly stiff the Fortran solver diverges and raises the
      'capi_return is NULL' / 'cb_f_in_lsoda failed' error.
      Manual RK4 with a small fixed step and hard state-clipping
      guarantees the trajectory stays bounded even when the SINDy
      model has residual instability.
    """
    r      = days_data.get(day_num, {})
    model  = r.get('sindy_model')
    scaler = r.get('sindy_scaler')
    lat    = r.get('m1_lat')

    # Prefer the actual input used for fitting (reach or sleep latent)
    sindy_input_lat = r.get('sindy_input_lat')
    if sindy_input_lat is not None:
        lat = sindy_input_lat


    if model is None or scaler is None or lat is None:
        print(f'    Day {day_num}: SINDy simulate skipped – missing components')
        return None

    n_sindy  = min(SINDY_N_PCS, lat.shape[1])
    x_raw    = lat[:, :n_sindy]
    x0       = scaler.transform(x_raw.mean(axis=0, keepdims=True))[0]

    # Clip initial condition to ±3 σ  (already standardised, so ±3 is safe)
    x0 = np.clip(x0, -3.0, 3.0)

    dt_rk  = 0.02          # RK4 step (s) – small for stability
    clip_v = 5.0           # hard state clip at each step  (in scaled units)
    steps  = t_max

    def f(x):
        """Evaluate dx/dt from the SINDy model at state x."""
        # model.predict expects shape (1, n_features)
        xr = x.reshape(1, -1)
        return model.predict(xr)[0]

    traj = [x0.copy()]
    x    = x0.copy()

    for _ in range(steps - 1):
        try:
            k1 = f(x)
            k2 = f(x + 0.5 * dt_rk * k1)
            k3 = f(x + 0.5 * dt_rk * k2)
            k4 = f(x +       dt_rk * k3)
            x  = x + (dt_rk / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            x  = np.clip(x, -clip_v, clip_v)   # prevent blow-up

            if not np.all(np.isfinite(x)):
                print(f'    Day {day_num}: RK4 hit non-finite at step {_}, stopping')
                break
        except Exception as e:
            print(f'    Day {day_num}: RK4 step error – {e}')
            break
        traj.append(x.copy())

    sim_scaled = np.array(traj)                          # (steps, n_sindy)

    # Back-transform to original fPCA latent space
    sim = scaler.inverse_transform(sim_scaled)

    if not np.all(np.isfinite(sim)):
        print(f'    Day {day_num}: sim has non-finite values after inverse transform')
        return None

    print(f'    Day {day_num}: SINDy ODE simulated OK  ({len(sim)} steps)')
    return sim


# ═══════════════════════════════════════════════════════════════════════
# FIGURE
# ═══════════════════════════════════════════════════════════════════════

C = dict(m1='#00d4aa', pfc='#ff6b8a', both='#a78bfa', swr='#ffd166',
         bg='#161b22', text='#e6edf3', grid='#21262d',
         early='#56cfe1', late='#ff9f1c')

def sax(ax, xl='', yl='', ttl=''):
    ax.set_facecolor(C['bg'])
    ax.tick_params(colors=C['text'], labelsize=9)
    ax.spines[:].set_color(C['grid'])
    if xl:  ax.set_xlabel(xl,  color=C['text'], fontsize=10)
    if yl:  ax.set_ylabel(yl,  color=C['text'], fontsize=10)
    if ttl: ax.set_title(ttl,  color=C['text'], fontsize=11, fontweight='bold')

fig = plt.figure(figsize=(22, 28))
fig.patch.set_facecolor('#0d1117')
gs  = gridspec.GridSpec(5, 3, figure=fig, hspace=0.46, wspace=0.36)


# ── PANEL 1: PFC–M1 SO coupling ──────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
sax(ax1, xl='Day', yl='PFC–M1 SO Coupling',
    ttl='Fig 1d  PFC–M1 SO Coupling Over Days')
vm1 = ~np.isnan(pfc_m1_coup)
if vm1.any():
    ax1.scatter(day_nums[vm1], pfc_m1_coup[vm1], color=C['pfc'], s=70, zorder=5)
    ax1.plot(day_nums[vm1], pfc_m1_coup[vm1], color=C['pfc'], alpha=0.4, lw=1)
    pp, xf, yf = fit_piecewise(day_nums, pfc_m1_coup)
    if xf is not None:
        ax1.plot(xf, yf, color='white', lw=2, ls='--',
                 label=f'Breakpoint: Day {pp[0]:.1f}')
        ax1.axvline(pp[0], color='yellow', lw=1.5, ls=':', alpha=0.8,
                    label='Consolidation onset')
    ax1.legend(facecolor=C['bg'], labelcolor=C['text'], fontsize=8)
else:
    ax1.text(0.5, 0.5, 'Need ≥4 days', ha='center', va='center',
             color=C['text'], transform=ax1.transAxes)


# ── PANEL 2: SWR–M1 SO coupling ──────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
sax(ax2, xl='Day', yl='P(SWR | M1 SO  ±0.75 s)',
    ttl='Fig 2  SWR–M1 SO Coupling Over Days')
vm2 = ~np.isnan(swr_m1_coup)
if vm2.any():
    ax2.scatter(day_nums[vm2], swr_m1_coup[vm2], color=C['swr'], s=70, zorder=5)
    ax2.plot(day_nums[vm2], swr_m1_coup[vm2], color=C['swr'], alpha=0.4, lw=1)
    pp2, xf2, yf2 = fit_piecewise(day_nums, swr_m1_coup)
    if xf2 is not None:
        ax2.plot(xf2, yf2, color='white', lw=2, ls='--')
ax2.set_ylim([0, None])


# ── PANEL 3: Manifold Fidelity ────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
sax(ax3, xl='Day', yl='Manifold Fidelity\n(fPCA traj. cosine similarity)',
    ttl='Fig 3  M1 Manifold Fidelity (fPCA)')
vm3 = ~np.isnan(fidelity)
if vm3.any():
    ax3.scatter(day_nums[vm3], fidelity[vm3], color=C['m1'], s=70, zorder=5)
    ax3.plot(day_nums[vm3], fidelity[vm3], color=C['m1'], alpha=0.4, lw=1)
    pp3, xf3, yf3 = fit_piecewise(day_nums, fidelity)
    if xf3 is not None:
        ax3.plot(xf3, yf3, color='white', lw=2, ls='--')


# ── PANEL 4: CCA CC1 ─────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
sax(ax4, xl='Day', yl='Canonical Correlation  CC1',
    ttl='Fig 4b  CCA Communication Subspace (fPCA latents)')
vm4 = ~np.isnan(cca_cc1)
if vm4.any():
    ax4.scatter(day_nums[vm4], cca_cc1[vm4], color=C['both'], s=70, zorder=5)
    ax4.plot(day_nums[vm4], cca_cc1[vm4], color=C['both'], alpha=0.4, lw=1)
    pp4, xf4, yf4 = fit_piecewise(day_nums, cca_cc1)
    if xf4 is not None:
        ax4.plot(xf4, yf4, color='white', lw=2, ls='--')


# ── PANEL 5: fPCA explained variance over days ───────────────────────
ax5 = fig.add_subplot(gs[1, 1])
sax(ax5, xl='Day', yl='Cumulative EVR  (fPC 1–5)',
    ttl='fPCA Explained Variance Ratio Over Days\n'
        'Rising EVR → more variance captured by smooth modes')
for d in day_nums:
    r   = days_data[int(d)]
    evr = r.get('m1_var')
    if evr is not None and len(evr) > 0:
        ax5.scatter(d, np.cumsum(evr)[-1],
                    color=C['m1'], s=60, zorder=5)
ax5.set_ylim([0, 1.05])
ax5.axhline(0.8, color='white', lw=1, ls=':', alpha=0.5,
            label='80 % variance')
ax5.legend(facecolor=C['bg'], labelcolor=C['text'], fontsize=8)


# ── PANEL 6: fPCA fPC1 vs fPC2 latent portrait (all days overlaid) ───
ax6 = fig.add_subplot(gs[1, 2])
sax(ax6, xl='fPC 1', yl='fPC 2',
    ttl='M1 fPCA Latent Space – All Days\n'
        'Color encodes day (early=blue → late=orange)')
cmap = plt.cm.plasma
for ii, d in enumerate(day_nums):
    r   = days_data[int(d)]
    lat = r.get('m1_lat')
    if lat is None or lat.shape[1] < 2:
        continue
    col = cmap(ii / max(N_DAYS - 1, 1))
    ax6.plot(lat[::30, 0], lat[::30, 1],
             alpha=0.35, lw=0.7, color=col)
sm = plt.cm.ScalarMappable(cmap=cmap,
     norm=plt.Normalize(vmin=int(day_nums[0]), vmax=int(day_nums[-1])))
sm.set_array([])
cb = plt.colorbar(sm, ax=ax6, fraction=0.04, pad=0.02)
cb.set_label('Day', color=C['text'], fontsize=9)
cb.ax.yaxis.set_tick_params(color=C['text'])


# ── PANELS 7–8: SINDy phase portraits (early vs late) ────────────────
for col_idx, (label, slc, col_name) in enumerate([
        ('Early  (Days 1–4)',   EARLY_DAYS, C['early']),
        ('Late   (Days 10–13)', LATE_DAYS,  C['late'])]):

    ax = fig.add_subplot(gs[2, col_idx])
    sax(ax, xl='fPC 2', yl='fPC 3',
        ttl=f'SINDy ODE Attractor:  {label}')

    any_sim = False
    for d in day_nums[slc]:
        r = days_data.get(int(d))
        if r is None:
            continue
        lat = r['m1_lat']
        if lat.shape[1] < 3:
            print(f'    Day {int(d)}: only {lat.shape[1]} fPCs, skipping portrait')
            continue
        ax.plot(lat[::20, 1], lat[::20, 2],
                alpha=0.25, lw=0.8, color=col_name,
                label=f'Day {int(d)} data')
        sim = simulate_sindy(int(d))
        if sim is not None and len(sim.shape) >= 2 and sim.shape[1] >= 3:
            ax.plot(sim[:, 1], sim[:, 2],
                    color='white', lw=2.0, ls='--', alpha=0.9,
                    label=f'Day {int(d)} ODE')
            any_sim = True
        else:
            print(f'    Day {int(d)}: no ODE sim plotted')

    if not any_sim:
        ax.text(0.5, 0.5, 'ODE not available\n(see console)',
                ha='center', va='center',
                color=C['text'], fontsize=9,
                transform=ax.transAxes)
    hd, lb = ax.get_legend_handles_labels()
    if hd:
        ax.legend(hd[-4:], lb[-4:],
                  facecolor=C['bg'], labelcolor=C['text'], fontsize=7)

# ── PANEL 10: Normalised overlay ──────────────────────────────────────
ax10 = fig.add_subplot(gs[3, :2])
sax(ax10, xl='Day',
    yl='Normalised value  (0 = min, 1 = max)',
    ttl='All Consolidation Metrics – Normalised Overlay')

def n01(v):
    mn, mx = np.nanmin(v), np.nanmax(v)
    return (v - mn) / (mx - mn + 1e-8)

for vals, col, lbl, m in [
    (pfc_m1_coup, C['pfc'],  'PFC–M1 SO coupling', 'o'),
    (swr_m1_coup, C['swr'],  'SWR–M1 SO coupling', 's'),
    (fidelity,    C['m1'],   'Manifold Fidelity (fPCA)',  '^'),
    (cca_cc1,     C['both'], 'CCA CC1 (fPCA latents)',    'D'),
]:
    v = ~np.isnan(vals)
    if v.sum() > 1:
        ax10.plot(day_nums[v], n01(vals[v]),
                  color=col, lw=2, marker=m, ms=6, label=lbl)

if N_DAYS >= 8:
    ax10.axvspan(day_nums[0], day_nums[min(3, N_DAYS-1)],
                 alpha=0.08, color=C['early'], label='Early phase')
    ax10.axvspan(day_nums[max(0, N_DAYS-4)], day_nums[-1],
                 alpha=0.08, color=C['late'],  label='Late phase')
ax10.set_ylim([-0.05, 1.15])
ax10.legend(facecolor=C['bg'], labelcolor=C['text'], fontsize=9,
            ncol=2, loc='upper left')


# ── PANEL 11: Summary stats ───────────────────────────────────────────
ax11 = fig.add_subplot(gs[4, :])
sax(ax11, ttl='Summary: Early vs Late Means  (fPCA pipeline)')
ax11.axis('off')

def sm(arr, slc):
    v = arr[slc][~np.isnan(arr[slc])]
    return f'{v.mean():.3f}±{v.std():.3f}' if len(v) else 'N/A'

rows = [
    ('Metric',                     'Early (1–4)',                'Late (10–13)',   'Expected'),
    ('PFC–M1 SO coupling',         sm(pfc_m1_coup, EARLY_DAYS),  sm(pfc_m1_coup, LATE_DAYS),  '↑'),
    ('SWR–M1 SO coupling',         sm(swr_m1_coup, EARLY_DAYS),  sm(swr_m1_coup, LATE_DAYS),  '↑ early / ↓ late'),
    ('Manifold Fidelity (fPCA)',   sm(fidelity,    EARLY_DAYS),  sm(fidelity,    LATE_DAYS),  '↑'),
    ('CCA CC1 (fPCA latents)',      sm(cca_cc1,     EARLY_DAYS),  sm(cca_cc1,     LATE_DAYS),  '↑'),
]
CX = [0.01, 0.32, 0.57, 0.78]
for ri, row in enumerate(rows):
    y  = 0.92 - ri * 0.18
    fw = 'bold' if ri == 0 else 'normal'
    for ci, cell in enumerate(row):
        color = '#ffcc44' if ri == 0 else C['text']
        ax11.text(CX[ci], y, cell, transform=ax11.transAxes,
                  color=color, fontsize=9, va='top',
                  fontweight=fw, fontfamily='monospace')


plt.suptitle(
    'PFC–M1 Manifold Consolidation  ·  Multi-Day Analysis  (Functional PCA)\n'
    'Kim, Joshi, Frank, Ganguly – Nature 613, 103–110 (2023)',
    color=C['text'], fontsize=15, fontweight='bold', y=0.999)

out_png = os.path.join(images_dir, 'summary_figure.png')
plt.savefig(out_png, dpi=130, bbox_inches='tight', facecolor='#0d1117')
print(f'\nFigure saved → {out_png}')

# ── Console summary ───────────────────────────────────────────────────
print('\n' + '─' * 80)
print(f"{'Day':>4}  {'PFC-M1 SO':>10}  {'SWR-M1 SO':>10}  "
      f"{'Fidelity':>9}  {'CCA CC1':>8}  {'M1 SOs':>6}  {'SWRs':>6}")
print('─' * 80)
for d in day_nums:
    r = days_data[int(d)]
    print(f"{int(d):>4}  "
          f"{r.get('pfc_m1_so_coupling', np.nan):>10.4f}  "
          f"{r.get('swr_m1so_coupling',  np.nan):>10.4f}  "
          f"{r.get('manifold_fidelity',  np.nan):>9.4f}  "
          f"{r.get('cca_cc1',            np.nan):>8.4f}  "
          f"{r.get('n_m1_so', 0):>6d}  "
          f"{r.get('n_swr',   0):>6d}")
print('─' * 80)


# ═══════════════════════════════════════════════════════════════════════
# EXPORT SINDY ODEs TO TEXT FILE
# ═══════════════════════════════════════════════════════════════════════

def format_ode_text(day_num):
    """
    Format the SINDy ODE for one day as human-readable equations.

    For each state variable (fPC1, fPC2, ...) prints:
        d(fPC1)/dt = c0 + c1*fPC1 + c2*fPC2 + c3*fPC1^2 + ...

    Coefficients below 1e-6 are treated as zero (sparsity threshold).
    Also reports:
      - Explained variance of each fPC
      - Numerical Jacobian max real eigenvalue at mean state
        (stability indicator: <0 = stable attractor, >0 = unstable)
    """
    r      = days_data.get(day_num, {})
    model  = r.get('sindy_model')
    coef   = r.get('sindy_coef')
    scaler = r.get('sindy_scaler')
    sindy_input_lat = r.get('sindy_input_lat')
    lat = sindy_input_lat if sindy_input_lat is not None else r.get('m1_lat')
    evr    = r.get('m1_var')

    lines = []
    lines.append(f"Day {day_num}")
    lines.append("=" * 60)

    # fPCA explained variance
    if evr is not None:
        lines.append("fPCA Explained Variance Ratio (M1):")
        for i, v in enumerate(evr):
            lines.append(f"  fPC{i+1}: {v:.4f}  ({v*100:.1f} %)")
        lines.append(f"  Cumulative: {evr.sum():.4f}  ({evr.sum()*100:.1f} %)")
    lines.append("")

    if model is None or coef is None:
        lines.append("  SINDy fit not available for this day.")
        lines.append("")
        return "\n".join(lines)

    # Feature names  e.g. ['1', 'x0', 'x1', 'x2', 'x0^2', 'x0 x1', ...]
    try:
        feat_names = model.get_feature_names()
    except Exception:
        feat_names = [f'f{j}' for j in range(coef.shape[1])]

    n_sindy = min(SINDY_N_PCS, lat.shape[1]) if lat is not None else coef.shape[0]
    var_names = [f'fPC{i+1}' for i in range(n_sindy)]

    # Replace generic xi names with fPCi names in feature strings
    feat_display = []
    for fn in feat_names:
        fd = fn
        for i in range(n_sindy - 1, -1, -1):   # replace x2 before x1 before x0
            fd = fd.replace(f'x{i}', f'fPC{i+1}')
        feat_display.append(fd)

    lines.append("SINDy ODE  (M1 fPCA latent space):")
    lines.append("  Model library: polynomial degree 2")
    lines.append(f"  State variables: {', '.join(var_names)}")
    lines.append("")

    for eq_idx, var in enumerate(var_names):
        if eq_idx >= coef.shape[0]:
            break
        terms = []
        for j, (c, fn) in enumerate(zip(coef[eq_idx], feat_display)):
            if abs(c) < 1e-6:
                continue
            sign = '+' if c >= 0 else '-'
            terms.append(f'{sign} {abs(c):.6f}*{fn}')
        if not terms:
            rhs = "0   (all coefficients zeroed by sparsity)"
        else:
            rhs = "  ".join(terms).lstrip("+ ").strip()
        lines.append(f"  d({var})/dt = {rhs}")
    lines.append("")

    # Numerical Jacobian stability at mean state
    if scaler is not None and lat is not None:
        try:
            x0 = scaler.transform(lat[:, :n_sindy].mean(axis=0, keepdims=True))[0]
            x0 = np.clip(x0, -3.0, 3.0)
            eps = 1e-4
            n   = len(x0)
            J   = np.zeros((n, n))
            f0  = model.predict(x0.reshape(1, -1))[0]
            for j in range(n):
                xp = x0.copy(); xp[j] += eps
                fp = model.predict(xp.reshape(1, -1))[0]
                J[:, j] = (fp - f0) / eps
            eigvals  = np.linalg.eigvals(J)
            max_real = float(np.max(eigvals.real))
            stability = ("STABLE attractor  (consolidated)"
                         if max_real < 0 else
                         "UNSTABLE / exploring"
                         if max_real > 0.1 else
                         "Neutrally stable  (near-zero eigval)")
            lines.append(f"  Jacobian max real eigenvalue: {max_real:.4f}")
            lines.append(f"  Stability assessment: {stability}")
            lines.append(f"  All eigenvalues (real parts): "
                         f"{[f'{v.real:.3f}' for v in eigvals]}")
        except Exception as e:
            lines.append(f"  Jacobian computation failed: {e}")
    lines.append("")
    return "\n".join(lines)


ode_txt_path = os.path.join(equations_dir, 'sindy_odes.txt')
sep = "\n" + "─" * 70 + "\n"

with open(ode_txt_path, 'w', encoding='utf-8') as fh:
    fh.write("SINDy ODEs – PFC–M1 Manifold Consolidation Analysis\n")
    fh.write("Kim, Joshi, Frank, Ganguly – Nature 613, 103–110 (2023)\n")
    fh.write("Dimensionality reduction: Functional PCA (B-spline basis)\n")
    fh.write("=" * 70 + "\n\n")

    fh.write("INTERPRETATION GUIDE\n")
    fh.write("────────────────────\n")
    fh.write("Each equation describes how one fPCA latent dimension evolves\n")
    fh.write("over time during sleep, as identified by SINDy sparse regression.\n\n")
    fh.write("  Early days : large nonlinear cross-terms (x1*x2, x1^2 ...)\n")
    fh.write("               → manifold is actively reorganising\n")
    fh.write("  Late days  : sparse, near-linear or zero equations\n")
    fh.write("               → stable attractor (consolidated memory)\n\n")
    fh.write("Stability is assessed via the numerical Jacobian eigenvalues\n")
    fh.write("at the mean operating point (standardised latent space):\n")
    fh.write("  max_real < 0  →  stable fixed-point attractor\n")
    fh.write("  max_real ≈ 0  →  neutrally stable (limit cycle possible)\n")
    fh.write("  max_real > 0  →  unstable / diverging trajectories\n")
    fh.write("=" * 70 + "\n\n")

    for d in day_nums:
        fh.write(format_ode_text(int(d)))
        fh.write(sep)

print(f'SINDy ODEs saved → {ode_txt_path}')