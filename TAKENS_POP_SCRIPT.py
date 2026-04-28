"""
Takens_Population.py  —  Takens' Delay Embedding on Kim et al. 2022 data
=========================================================================
"""

import os, re, glob, warnings, logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
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
    
    # Output directory for the delay-embedded manifolds
    OUT_FOLDER      = 'Takens_Embedding_allanimals',

    BIN_SIZE_MS     = 50,           # spike-count bin width in ms

    # Which LFP to use for NREM detection: 'M1' | 'PFC' | 'HPC'
    NREM_LFP_REGION = 'M1',

    # Population aggregation method
    POP_METHOD      = 'pca_ensemble',
    N_PCA_COMPS     = 5,            
    N_CLUSTERS      = 4,            
    SMOOTH_SIGMA    = 2.0,          
    MIN_RATE_HZ     = 0.05,         

    # --- TAKENS EMBEDDING PARAMETERS ---
    TARGET_SIGNAL   = 0,            # Which population signal to embed (0 = PC1)
    TAU_BINS        = 5,            # Time delay (tau) in number of bins
    EMBED_DIM       = 3,            # Embedding dimension (d). 3 is good for visualization

    SAVE_PLOTS      = True,
    PLOT_DPI        = 120,
)
# =============================================================================


# ---------------------------------------------------------------------------
#  STEP 1 — LOAD & PARSE (Unchanged)
# ---------------------------------------------------------------------------

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
    if not neuron_ids: return []
    return [data[f'Sleep_spike_time_{region}_cell{epoch_idx}_cell{n}_cell0'].astype(float) for n in neuron_ids]

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
    return np.array([np.any(lfp[b * bin_samples:(b + 1) * bin_samples] != 0.0) for b in range(n_bins)])


# ---------------------------------------------------------------------------
#  STEP 2 — POPULATION AGGREGATION (Unchanged)
# ---------------------------------------------------------------------------

def _smooth(arr, sigma):
    return gaussian_filter1d(arr.astype(float), sigma=sigma, axis=1) if sigma > 0 else arr.astype(float)

def _active(spike_mat, bin_size_ms, min_rate_hz):
    bin_dur = bin_size_ms / 1000.0
    rate    = spike_mat.mean(axis=1) / bin_dur
    mask    = rate >= min_rate_hz
    return mask, rate

def build_population_signals(spike_mat, cfg, label=''):
    if spike_mat.shape[1] == 0: return np.zeros((1, 0))
    method  = cfg['POP_METHOD']
    active, rate = _active(spike_mat, cfg['BIN_SIZE_MS'], cfg['MIN_RATE_HZ'])
    if not active.any(): return np.zeros((1, spike_mat.shape[1]))

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
            if m.any(): pop[k] = ((r[m] / r[m].sum())[:, None] * sm[m]).sum(axis=0)

    mu, std = pop.mean(axis=1, keepdims=True), pop.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (pop - mu) / std

def build_state_matrix(data, cfg, epoch_idx, epoch_label):
    fs = get_fs(data, epoch_idx)
    try: lfp = get_lfp(data, cfg['NREM_LFP_REGION'], epoch_idx)
    except KeyError: return None

    duration_s = len(lfp) / fs
    nrem = make_nrem_mask(lfp, fs, cfg['BIN_SIZE_MS'])

    all_pop = []
    for region in ['M1', 'PFC']:
        times = collect_spike_times(data, region, epoch_idx)
        if not times: continue
        mat   = bin_spike_times(times, duration_s, cfg['BIN_SIZE_MS'])
        mask  = nrem[:mat.shape[1]]
        pop   = build_population_signals(mat[:, mask], cfg, f"{region}_{epoch_label}")
        all_pop.append(pop)

    return np.vstack(all_pop) if all_pop else None


# ---------------------------------------------------------------------------
#  STEP 3 — TAKENS DELAY EMBEDDING
# ---------------------------------------------------------------------------

def run_takens_embedding(state_matrix, cfg):
    """
    Constructs a delay-embedded manifold from a single 1D signal.
    """
    target_idx = cfg['TARGET_SIGNAL']
    tau = cfg['TAU_BINS']
    d = cfg['EMBED_DIM']

    if state_matrix.shape[0] <= target_idx:
        log.warning(f"  Target signal index {target_idx} out of bounds. Using 0.")
        target_idx = 0

    # Extract the 1D signal to embed (e.g., PC1 of M1)
    signal = state_matrix[target_idx, :]
    T = len(signal)
    
    # Calculate the number of valid vectors we can form
    max_idx = T - (d - 1) * tau
    
    if max_idx <= 0:
        log.warning("  Signal too short for the chosen Tau and Dimension.")
        return None

    # Construct the embedding matrix (d x max_idx)
    embedded = np.zeros((d, max_idx))
    
    for i in range(d):
        start = i * tau
        end = start + max_idx
        # Row 0: x(t)
        # Row 1: x(t + tau)
        # Row 2: x(t + 2*tau) ...
        embedded[i, :] = signal[start:end]

    log.info(f"  Created Delay Embedding: tau={tau}, d={d}. Shape: {embedded.shape}")
    return embedded


# ---------------------------------------------------------------------------
#  PLOTTING & SAVING
# ---------------------------------------------------------------------------

def plot_takens_attractor(embedded, label, out_dir, dpi):
    """Plots the reconstructed 3D manifold."""
    if embedded is None or embedded.shape[0] < 3:
        return

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # We plot the first 3000 points to avoid visual clutter
    plot_points = min(embedded.shape[1], 3000)
    
    x = embedded[0, :plot_points]
    y = embedded[1, :plot_points]
    z = embedded[2, :plot_points]
    
    # Color by time to show the trajectory evolution
    time_color = np.linspace(0, 1, plot_points)
    
    # Use a scatter plot with connected lines for visual continuity
    ax.plot(x, y, z, color='gray', lw=0.5, alpha=0.5)
    scatter = ax.scatter(x, y, z, c=time_color, cmap='viridis', s=2, alpha=0.8)
    
    ax.set_title(f"{label} — Reconstructed Takens Attractor", fontsize=11)
    ax.set_xlabel('x(t)')
    ax.set_ylabel(r'$x(t + \tau)$')
    ax.set_zlabel(r'$x(t + 2\tau)$')
    
    plt.colorbar(scatter, label='Normalized Time', ax=ax, pad=0.1)
    plt.tight_layout()
    out = os.path.join(out_dir, f'{label}_takens_3d.png')
    fig.savefig(out, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def save_results(animal, epoch_label, day, embedded, out_dir):
    if embedded is None: return
    fname = os.path.join(out_dir, f'{animal}_{day}_{epoch_label}_takens.npz')
    np.savez_compressed(fname, delay_embedded_manifold=embedded)


# ---------------------------------------------------------------------------
#  MAIN
# ---------------------------------------------------------------------------

EPOCHS = {'pre': 0, 'post': 1}

def process_animal(animal_name, cfg):
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
        data = load_npz(fpath)

        for epoch, idx in EPOCHS.items():
            state = build_state_matrix(data, cfg, idx, epoch)
            if state is not None and state.size > 0:
                
                # 1. Embed the manifold for this specific day/epoch
                embedded = run_takens_embedding(state, cfg)
                
                # 2. Visualize and Save
                if cfg['SAVE_PLOTS'] and embedded is not None:
                    plot_takens_attractor(embedded, f"{animal_name}_{day}_{epoch}", out_dir, cfg['PLOT_DPI'])
                save_results(animal_name, epoch, day, embedded, out_dir)

    log.info(f"\n  Done: {animal_name}")


if __name__ == '__main__':
    import sys
    cfg = CONFIG.copy()
    if len(sys.argv) > 1:
        animal = sys.argv[1]
        cfg['DATA_PATTERN'] = f'{animal}_Day*.npz'
        process_animal(animal, cfg)
    else:
        all_files = sorted(glob.glob(os.path.join(cfg['DATA_FOLDER'], cfg['DATA_PATTERN'])))
        animals = sorted(set(os.path.basename(f).split('_')[0] for f in all_files))
        for a in animals:
            process_animal(a, cfg)