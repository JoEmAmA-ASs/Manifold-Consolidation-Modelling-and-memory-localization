"""
FPCA_Population.py  —  Population-level FPCA on Kim et al. 2022 data
=========================================================================
"""

import os, re, glob, warnings, logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.linalg import svd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning

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
    
    # Changed output directory as requested
    OUT_FOLDER      = 'FPCA_scores_allanimals',

    BIN_SIZE_MS     = 50,           # spike-count bin width in ms

    # Which LFP to use for NREM detection: 'M1' | 'PFC' | 'HPC'
    NREM_LFP_REGION = 'M1',

    # Population aggregation method: 'pca_ensemble' | 'activity_weighted'
    #                                 | 'kmeans' | 'mean'
    POP_METHOD      = 'pca_ensemble',
    N_PCA_COMPS     = 5,            # per region, for pca_ensemble
    N_CLUSTERS      = 4,            # per region, for kmeans
    SMOOTH_SIGMA    = 2.0,          # Gaussian smooth sigma (bins)
    MIN_RATE_HZ     = 0.05,         # exclude neurons below this firing rate

    N_FPCA_COMPS    = 6,            # functional PCs kept across days

    SAVE_PLOTS      = True,
    PLOT_DPI        = 120,
)
# =============================================================================


# ---------------------------------------------------------------------------
#  STEP 1 — LOAD & PARSE
# ---------------------------------------------------------------------------

def load_npz(filepath):
    raw = np.load(filepath, allow_pickle=True)
    return {k: raw[k] for k in raw.files}


def get_fs(data, epoch_idx):
    """Sampling rate for epoch_idx (0=pre, 1=post)."""
    arr = np.ravel(data['Fs_LFP'])
    return float(arr[min(epoch_idx, len(arr) - 1)])


def get_lfp(data, region, epoch_idx):
    """Return 1-D LFP array.  region: 'M1' | 'PFC' | 'HPC'"""
    suffix = {
        'M1':  f'Sleep_LFP_delta_M1_cell{epoch_idx}',
        'PFC': f'Sleep_LFP_delta_PFC_cell{epoch_idx}',
        'HPC': f'Sleep_LFP_150to250_HPC_cell{epoch_idx}',
    }[region]
    if suffix not in data:
        raise KeyError(f"LFP key {suffix!r} missing")
    return data[suffix].astype(float)


def collect_spike_times(data, region, epoch_idx):
    """
    Return list[np.ndarray] of spike times in seconds — one array per neuron.
    Keys: Sleep_spike_time_{region}_cell{epoch}_cell{n}_cell0
    """
    pat = re.compile(
        rf'Sleep_spike_time_{region}_cell{epoch_idx}_cell(\d+)_cell0$')
    neuron_ids = sorted(
        int(m.group(1))
        for k in data
        if (m := pat.match(k))
    )
    if not neuron_ids:
        log.warning(f"  No spike keys for {region} epoch{epoch_idx}")
        return []
    log.info(f"  [{region} epoch{epoch_idx}] {len(neuron_ids)} neurons")
    return [
        data[f'Sleep_spike_time_{region}_cell{epoch_idx}_cell{n}_cell0'
             ].astype(float)
        for n in neuron_ids
    ]


def bin_spike_times(spike_times, duration_s, bin_size_ms):
    """Spike times -> (N x T_bins) integer count matrix."""
    bin_s  = bin_size_ms / 1000.0
    n_bins = int(duration_s / bin_s)
    mat    = np.zeros((len(spike_times), n_bins), dtype=np.float32)
    for i, t in enumerate(spike_times):
        if t.size:
            counts, _ = np.histogram(t, bins=n_bins, range=(0.0, duration_s))
            mat[i]    = counts
    return mat


def make_nrem_mask(lfp, fs, bin_size_ms):
    """Bool mask (T_bins,): True where any LFP sample in the bin != 0."""
    bin_samples = int(fs * bin_size_ms / 1000.0)
    n_bins      = len(lfp) // bin_samples
    mask = np.array([
        np.any(lfp[b * bin_samples:(b + 1) * bin_samples] != 0.0)
        for b in range(n_bins)
    ])
    log.info(f"  NREM mask: {mask.sum()}/{n_bins} bins "
             f"({100 * mask.mean():.1f}%)")
    return mask


# ---------------------------------------------------------------------------
#  STEP 2 — POPULATION AGGREGATION
# ---------------------------------------------------------------------------

def _smooth(arr, sigma):
    return gaussian_filter1d(arr.astype(float), sigma=sigma, axis=1) \
           if sigma > 0 else arr.astype(float)


def _active(spike_mat, bin_size_ms, min_rate_hz):
    bin_dur = bin_size_ms / 1000.0
    rate    = spike_mat.mean(axis=1) / bin_dur
    mask    = rate >= min_rate_hz
    log.info(f"    Active neurons >=({min_rate_hz} Hz): {mask.sum()}/{len(mask)}")
    return mask, rate


def build_population_signals(spike_mat, cfg, label=''):
    """(N x T) -> (K x T) z-scored population signals."""
    if spike_mat.shape[1] == 0:
        return np.zeros((1, 0))

    method  = cfg['POP_METHOD']
    active, rate = _active(spike_mat, cfg['BIN_SIZE_MS'], cfg['MIN_RATE_HZ'])

    if not active.any():
        log.warning(f"  [{label}] No active neurons")
        return np.zeros((1, spike_mat.shape[1]))

    sub  = spike_mat[active]
    r    = rate[active]
    sm   = _smooth(sub, cfg['SMOOTH_SIGMA'])

    if method == 'mean':
        pop = sm.mean(axis=0, keepdims=True)

    elif method == 'activity_weighted':
        w   = r / r.sum()
        pop = (w[:, None] * sm).sum(axis=0, keepdims=True)

    elif method == 'pca_ensemble':
        nc  = min(cfg['N_PCA_COMPS'], active.sum(), sm.shape[1])
        pca = PCA(n_components=nc, random_state=42)
        pop = pca.fit_transform(sm.T).T            # (nc, T)
        var = pca.explained_variance_ratio_
        log.info(f"  [{label}] PCA {nc} comps "
                 f"var={var.sum()*100:.1f}%  "
                 f"[{', '.join(f'{v*100:.1f}' for v in var)}]%")

    elif method == 'kmeans':
        nc  = min(cfg['N_CLUSTERS'], active.sum())
        km  = KMeans(n_clusters=nc, random_state=42, n_init=10)
        lbl = km.fit_predict(sm)
        pop = np.zeros((nc, sm.shape[1]))
        for k in range(nc):
            m = lbl == k
            if m.any():
                w = r[m] / r[m].sum()
                pop[k] = (w[:, None] * sm[m]).sum(axis=0)
        log.info(f"  [{label}] KMeans {nc} ensembles "
                 f"sizes={[(lbl==k).sum() for k in range(nc)]}")

    else:
        raise ValueError(f"Unknown POP_METHOD: {method!r}")

    # Z-score each signal
    mu  = pop.mean(axis=1, keepdims=True)
    std = pop.std(axis=1,  keepdims=True)
    std[std == 0] = 1.0
    pop = (pop - mu) / std

    log.info(f"  [{label}] Population signals shape: {pop.shape}")
    return pop


def build_state_matrix(data, cfg, epoch_idx, epoch_label):
    """
    Full (K_total x T_nrem) state matrix for one day/epoch.
    Combines M1 + PFC population signals.
    Returns None if data is insufficient.
    """
    fs = get_fs(data, epoch_idx)
    try:
        lfp = get_lfp(data, cfg['NREM_LFP_REGION'], epoch_idx)
    except KeyError as e:
        log.warning(f"  {e} — skipping epoch {epoch_label}")
        return None

    duration_s = len(lfp) / fs
    log.info(f"  [{epoch_label}] duration={duration_s:.1f}s  fs={fs:.2f}Hz")

    nrem = make_nrem_mask(lfp, fs, cfg['BIN_SIZE_MS'])

    all_pop = []
    for region in ['M1', 'PFC']:
        times = collect_spike_times(data, region, epoch_idx)
        if not times:
            continue
        mat   = bin_spike_times(times, duration_s, cfg['BIN_SIZE_MS'])
        T     = mat.shape[1]
        mask  = nrem[:T]                        # guard against off-by-one
        pop   = build_population_signals(mat[:, mask], cfg,
                                         f"{region}_{epoch_label}")
        all_pop.append(pop)

    if not all_pop:
        return None

    state = np.vstack(all_pop)
    log.info(f"  [{epoch_label}] State matrix: {state.shape}")
    return state


# ---------------------------------------------------------------------------
#  STEP 3 — FPCA across days
# ---------------------------------------------------------------------------

def run_fpca(matrices, n_components):
    """
    Functional PCA across days.
    matrices: list of (K, T_i) arrays — T_i may differ.
    Returns dict with scores, components, var_explained.
    """
    if len(matrices) < 2:
        raise ValueError(f"Need >=2 days, got {len(matrices)}")

    K     = min(m.shape[0] for m in matrices)   # safest common K
    max_T = max(m.shape[1] for m in matrices)
    nc    = min(n_components, len(matrices) - 1)

    rows = []
    for m in matrices:
        pad = np.zeros((K, max_T - m.shape[1]))
        rows.append(np.hstack([m[:K], pad]).ravel())

    X    = np.vstack(rows)
    mu   = X.mean(axis=0)
    U, S, Vt = svd(X - mu, full_matrices=False)

    var  = (S[:nc] ** 2) / (S ** 2).sum()
    log.info(f"  FPCA {nc} comps  total_var={var.sum()*100:.1f}%  "
             f"[{', '.join(f'{v*100:.1f}' for v in var)}]%")

    return dict(scores        = U[:, :nc] * S[:nc],
                components    = Vt[:nc],
                var_explained = var,
                mean          = mu,
                K=K, max_T=max_T)


# ---------------------------------------------------------------------------
#  PLOTTING & SAVING
# ---------------------------------------------------------------------------

def plot_pop_signals(state, label, dt_s, out_dir, dpi):
    K, T = state.shape
    t = np.arange(T) * dt_s / 60.0
    fig, axes = plt.subplots(K, 1, figsize=(14, 2.2 * K), sharex=True)
    if K == 1:
        axes = [axes]
    fig.suptitle(f"{label} — population signals", fontsize=11)
    for k, ax in enumerate(axes):
        ax.plot(t, state[k], lw=0.6, color=f'C{k}')
        ax.set_ylabel(f'Pop {k}', fontsize=8)
        ax.axhline(0, color='k', lw=0.3, ls='--', alpha=0.4)
    axes[-1].set_xlabel('Time (min)')
    plt.tight_layout()
    out = os.path.join(out_dir, f'{label}_pop.png')
    fig.savefig(out, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_fpca_scores(fpca, label, out_dir, dpi):
    sc, var = fpca['scores'], fpca['var_explained']
    n_days, nc = sc.shape
    days = np.arange(1, n_days + 1)
    fig, axes = plt.subplots(1, nc, figsize=(3 * nc, 3.5))
    if nc == 1:
        axes = [axes]
    fig.suptitle(f"{label} — FPCA scores across days", fontsize=11)
    for i, ax in enumerate(axes):
        ax.plot(days, sc[:, i], 'o-', color=f'C{i}', lw=1.5, ms=5)
        ax.set_title(f'fPC{i+1}  ({var[i]*100:.1f}%)', fontsize=9)
        ax.set_xlabel('Day')
        if i == 0:
            ax.set_ylabel('Score')
    plt.tight_layout()
    out = os.path.join(out_dir, f'{label}_fpca.png')
    fig.savefig(out, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def save_results(animal, epoch_label, fpca, out_dir):
    payload = dict(fpca_scores       = fpca['scores'],
                   fpca_var_explained= fpca['var_explained'],
                   fpca_components   = fpca['components'],
                   fpca_mean         = fpca['mean'])
                   
    fname = os.path.join(out_dir, f'{animal}_{epoch_label}_fpca_results.npz')
    np.savez_compressed(fname, **payload)
    log.info(f"  Saved -> {fname}")


# ---------------------------------------------------------------------------
#  MAIN
# ---------------------------------------------------------------------------

EPOCHS = {'pre': 0, 'post': 1}

def process_animal(animal_name, cfg):
    folder  = cfg['DATA_FOLDER']
    pattern = os.path.join(
        folder, cfg['DATA_PATTERN'].replace('Animal*', animal_name))
    files   = sorted(glob.glob(pattern))

    if not files:
        log.error(f"No files: {animal_name}  pattern={pattern}")
        return

    out_dir = os.path.join(folder, cfg['OUT_FOLDER'], animal_name)
    os.makedirs(out_dir, exist_ok=True)

    log.info(f"\n{'='*68}")
    log.info(f"  {animal_name}  |  {len(files)} days")
    log.info(f"{'='*68}")

    day_states = {'pre': [], 'post': []}
    day_tags   = []

    for fpath in files:
        day = os.path.splitext(os.path.basename(fpath))[0]
        day_tags.append(day)
        log.info(f"\n  -- {day} --")
        data = load_npz(fpath)

        for epoch, idx in EPOCHS.items():
            state = build_state_matrix(data, cfg, idx, epoch)
            if state is not None and state.size > 0:
                day_states[epoch].append(state)
                if cfg['SAVE_PLOTS']:
                    plot_pop_signals(
                        state,
                        f"{animal_name}_{day}_{epoch}",
                        cfg['BIN_SIZE_MS'] / 1000.0,
                        out_dir, cfg['PLOT_DPI'])

    for epoch, mats in day_states.items():
        if len(mats) < 2:
            log.warning(f"  [{epoch}] only {len(mats)} day(s) — FPCA skipped")
            continue

        log.info(f"\n  -- FPCA [{epoch}]  {len(mats)} days --")
        fpca  = run_fpca(mats, cfg['N_FPCA_COMPS'])

        if cfg['SAVE_PLOTS']:
            plot_fpca_scores(fpca, f"{animal_name}_{epoch}", out_dir, cfg['PLOT_DPI'])

        save_results(animal_name, epoch, fpca, out_dir)

    log.info(f"\n  Done: {animal_name}")


if __name__ == '__main__':
    import sys
    cfg = CONFIG.copy()
    if len(sys.argv) > 1:
        animal = sys.argv[1]
        cfg['DATA_PATTERN'] = f'{animal}_Day*.npz'
        process_animal(animal, cfg)
    else:
        all_files = sorted(glob.glob(
            os.path.join(cfg['DATA_FOLDER'], cfg['DATA_PATTERN'])))
        animals = sorted(set(
            os.path.basename(f).split('_')[0] for f in all_files))
        log.info(f"Found animals: {animals}")
        for a in animals:
            process_animal(a, cfg)