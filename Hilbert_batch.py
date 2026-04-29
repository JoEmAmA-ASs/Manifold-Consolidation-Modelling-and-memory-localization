import os
import glob
import zipfile
import numpy as np
import pysindy as ps
from pysindy.feature_library import FourierLibrary, PolynomialLibrary, ConcatLibrary
from sklearn.decomposition import PCA
from scipy.signal import hilbert

# ==========================================
# 1. SETUP AND FILE DISCOVERY
# ==========================================
data_folder = r"D:\College\SEM 8\Sem_project"
output_folder = os.path.join(data_folder, "pySINDY_KA")
os.makedirs(output_folder, exist_ok=True)

# Find all .npz files in the target directory
npz_files = glob.glob(os.path.join(data_folder, "*.npz"))

if not npz_files:
    print(f"No .npz files found in {data_folder}. Please check the path.")

# ==========================================
# 2. BATCH PROCESSING LOOP
# ==========================================
for file in npz_files:
    base_name = os.path.basename(file).replace('.npz', '')
    
    try:
        animal_id, day_id = base_name.split('_')
    except ValueError:
        animal_id, day_id = base_name, "Unknown"

    print(f"\n==========================================")
    print(f"=== Processing {animal_id} | {day_id} ===")
    print(f"==========================================")
    
    # ---------------------------------------------------------
    # A. LOAD AND VALIDATE FILE (SKIPS TRUNCATED/NOT FOUND)
    # ---------------------------------------------------------
    try:
        # Load the file
        data = np.load(file, allow_pickle=True)
        # Test reading the files attribute to catch truncation immediately
        _ = data.files 
    except FileNotFoundError:
        print(f"⚠️ SKIPPING {base_name}: File not found.")
        continue
    except (zipfile.BadZipFile, EOFError, ValueError) as e:
        print(f"⚠️ SKIPPING {base_name}: File is truncated or corrupted ({e}).")
        continue
    except Exception as e:
        print(f"⚠️ SKIPPING {base_name}: Unexpected read error ({e}).")
        continue

    # ---------------------------------------------------------
    # B. EXTRACT AND CLEAN SPIKE DATA
    # ---------------------------------------------------------
    # Extract keys
    m1_keys = [k for k in data.files if k.startswith('Sleep_spike_time_M1')]
    pfc_keys = [k for k in data.files if k.startswith('Sleep_spike_time_PFC')]
    
    # Check if either region is completely missing
    if len(m1_keys) == 0 or len(pfc_keys) == 0:
        print(f"⚠️ SKIPPING {base_name}: Missing data. Found {len(m1_keys)} M1 and {len(pfc_keys)} PFC neurons.")
        data.close()
        continue 
        
    # Helper function to sanitize the data arrays
    def clean_spikes(raw_array):
        flat_arr = np.ravel(raw_array)
        no_nones = [x for x in flat_arr if x is not None]
        float_arr = np.array(no_nones, dtype=np.float64)
        return float_arr[~np.isnan(float_arr)]

    # Clean all arrays once and store them
    clean_m1_data = [clean_spikes(data[k]) for k in m1_keys]
    clean_pfc_data = [clean_spikes(data[k]) for k in pfc_keys]
    
    # Find global maximum time
    global_max_time = 0
    for spike_arr in clean_m1_data + clean_pfc_data:
        if len(spike_arr) > 0:
            global_max_time = max(global_max_time, np.max(spike_arr))

    if global_max_time == 0:
        print(f"⚠️ SKIPPING {base_name}: No valid spike timestamps found.")
        data.close()
        continue

    # Memory safe 10ms binning
    dt = 0.01 
    global_time_edges = np.arange(0, global_max_time + dt, dt)
    t_sindy = global_time_edges[:-1]

    def get_firing_rate(spike_timestamps):
        # If the neuron had zero valid spikes, return an array of zeros
        if len(spike_timestamps) == 0:
            return np.zeros_like(t_sindy, dtype=np.float32)
            
        counts, _ = np.histogram(spike_timestamps, bins=global_time_edges)
        rate = np.convolve(counts, np.ones(5)/5, mode='same')
        return (rate / dt).astype(np.float32) 

    print(f"Smoothing {len(m1_keys)} M1 neurons and {len(pfc_keys)} PFC neurons...")
    X_M1 = np.column_stack([get_firing_rate(arr) for arr in clean_m1_data])
    X_PFC = np.column_stack([get_firing_rate(arr) for arr in clean_pfc_data])
    
    data.close() # Close the npz file manually to free memory

    # ---------------------------------------------------------
    # C. DYNAMIC PCA (MANIFOLD EXTRACTION)
    # ---------------------------------------------------------
    print("Extracting Manifolds...")
    target_variance = 0.80

    def get_optimal_manifold(X_data, region_name):
        # Fallback if too few neurons or no variance
        if X_data.shape[1] == 0 or np.all(X_data == 0):
            return np.zeros((X_data.shape[0], 1)), 1
            
        pca_full = PCA().fit(X_data)
        cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
        optimal_components = np.argmax(cumulative_variance >= target_variance) + 1
        print(f"  {region_name}: Needs {optimal_components} PCs to hit {target_variance*100}% variance.")
        return PCA(n_components=optimal_components).fit_transform(X_data), optimal_components

    manifold_M1, num_m1 = get_optimal_manifold(X_M1, "M1")
    manifold_PFC, num_pfc = get_optimal_manifold(X_PFC, "PFC")

    feature_names = [f"M1_{i+1}" for i in range(num_m1)] + [f"PFC_{i+1}" for i in range(num_pfc)]
    
    # ---------------------------------------------------------
    # D. HILBERT TRANSFORM (AMPLITUDE -> PHASE)
    # ---------------------------------------------------------
    print("Applying Hilbert Transform to extract unwrapped phase...")

    def extract_unwrapped_phase(manifold_data):
        # Apply Hilbert transform along the time axis (axis=0)
        analytic_signal = hilbert(manifold_data, axis=0)
        # Extract the instantaneous phase (-pi to pi)
        instantaneous_phase = np.angle(analytic_signal)
        # Unwrap phase to prevent massive derivative spikes
        unwrapped_phase = np.unwrap(instantaneous_phase, axis=0)
        return unwrapped_phase

    phase_M1 = extract_unwrapped_phase(manifold_M1)
    phase_PFC = extract_unwrapped_phase(manifold_PFC)

    # Stack the phases together for SINDy
    X_latent_phase = np.column_stack((phase_M1, phase_PFC))

    # ---------------------------------------------------------
    # E. PYSINDY MODEL FITTING (KURAMOTO-ADLER)
    # ---------------------------------------------------------
    print("Running PySINDy with Phase Library...")

    # Library: Constant (intrinsic freq) + Sines/Cosines (phase coupling)
    const_lib = PolynomialLibrary(degree=0) 
    fourier_lib = FourierLibrary(n_frequencies=1) 
    kuramoto_library = ConcatLibrary([const_lib, fourier_lib])

    # Optimizer (Adjust threshold here to control sparsity)
    optimizer = ps.STLSQ(threshold=0.05, alpha=0.01)

    # Initialize the model without feature_names
    model = ps.SINDy(
        feature_library=kuramoto_library,
        optimizer=optimizer
    )

    try:
        # Fit the model and pass feature_names HERE
        model.fit(
            X_latent_phase, 
            t=dt, 
            feature_names=feature_names
        )
    except Exception as e:
        print(f"⚠️ SKIPPING {base_name}: SINDy fitting failed ({e}).")
        continue

    # ---------------------------------------------------------
    # F. SAVE EQUATIONS TO TEXT FILE
    # ---------------------------------------------------------
    print(f"Saving equations to {output_folder}...")
    eq_filename = os.path.join(output_folder, f"{base_name}_Kuramoto_equations.txt")
    
    with open(eq_filename, 'w') as f:
        f.write(f"--- {base_name} Discovered Phase Equations (Kuramoto-Adler) ---\n")
        
        # PySINDy's print() function prints to stdout, we need to capture it or use .equations()
        equations = model.equations(precision=3)
        for i, eq in enumerate(equations):
            f.write(f"({feature_names[i]})' = {eq}\n")
            
    print(f"✅ Successfully processed {base_name}!")

print("\n🎉 BATCH PROCESSING COMPLETE!")