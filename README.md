# Manifold-Consolidation-Modelling-and-memory-localization
Kim, J., Joshi, A., Frank, L. et al. Cortical–hippocampal coupling during manifold exploration in motor cortex. Nature 613, 103–110 (2023). https://doi.org/10.1038/s41586-022-05533-z

Systems consolidation, a process crucial for long-term memory stabilization, is believed to occur in two stages. Initially, new memories rely on the hippocampus; however, over time, these memories integrate into cortical networks, making them independent of the hippocampus. Our findings indicate that each animal exhibits a noticeable increase in coupling between slow oscillations in the prefrontal cortex and area M1 during sleep, which coincides with stabilised performance. This sharp increase predicts a subsequent decline in hippocampal sharp-wave ripple (SWR) and M1 slow oscillation coupling, suggesting a feedback mechanism that informs hippocampal disengagement and facilitates the transition to the second stage. In the first stage, there are significant increases in hippocampal SWR and M1 slow oscillation coupling during post-training sleep. This stage is closely associated with rapid learning and the variability observed in the low-dimensional manifold of M1. 
I attempted to model this phenomena in two ways : 1) Trying to capture the evolution of each neuron with time 
                                                  2) Trying to capture the evolution of each subpopulation of neuron with time.

But before moving onto computational methods lets look at the theoretical constructs that shape this:
1) Wilson - Cowan Model: The Wilson–Cowan model describes the dynamics of interactions between populations of very simple excitatory and inhibitory model neurons. It was developed by Hugh R. Wilson and Jack D. Cowan.Extensions of the model have been widely used in modeling neuronal populations.The model is important historically because it uses phase plane methods and numerical solutions to describe the responses of neuronal populations to stimuli. However since the model is pretty simple only elementary limit cycle behaviour (oscillations) and stimulus-dependent evoked responses are predicted.
2) Kuramoto - Adler Models : The KA models are utilized to showcase synchroniztion between two or more oscillator systems. Its formulation was motivated by the behavior of systems of chemical and biological oscillators, and it has found widespread applications in areas such as neuroscience. We have to show the initial synchronization and phase locking between Hippocampal neurons and Motor cortex neurons which fades away along the trials and bolsters the synchronization between Motor cortex and the Prefrontal Cortex. We assume these three neuronal subpopulations to be osciallators having different intrinsic frequencies which evolve to a common value as the regions are phase locked.
3) Hidden State Models: Final theoretical assumption is assuming the three subpopulations to be states in the model. These states would be intricately linked to each other and the transition probablities would be substituted with phase nearing that of the other state, the self probablities would be substituted by inhibition that would reject phase coupling. Lastly, the goals of the modelling i.e manifold consolidation and manifold exploration will be termed as hidden states in this model which will be affected by the coupling among the states.


Once the theoretical basis was clear I moved onto the computational part. The equations I obatined through theoretical means do capture some essence but did not fully encapsulate the dynamics of manifold consolidation. My next approach was using SINDY [Sparse Idetification of Non-Linear Dynamics] to capture the essence of manifold exploration.

Preparation of Data: 

1) Description about dataset

The number of animals = 6 rats (Animal 1 - Animal 6).

Single mat file is for a single-day data including three sequential recording blocks:

[pre-training sleep] - [reach task] - [post-training sleep]

In each mat file, you will see below data:

Fs_LFP: sampling rate of LPFs. Index 1 and 2 represents pre- and post-training sleep, respectively.
Sleep_LFP_delta_M1: delta-band (0.1-4 Hz) LFP in M1 during NREM sleep. Index 1 and 2 represents pre- and post-training sleep, respectively.
Sleep_LFP_delta_PFC: delta-band (0.1-4 Hz) LFP in PFC during NREM sleep. Index 1 and 2 represents pre- and post-training sleep, respectively.
Sleep_LFP_150to250_HPC: LFP filtered in 150-250 Hz in hippocampus CA1 during NREM sleep. Index 1 and 2 represents pre- and post-training sleep, respectively.
Sleep_spike_time_M1: Spike timing (in sec) in a single unit in M1 during NREM sleep. Cell index 1 and 2 represents pre- and post-training sleep, respectively. In each cell, there are N sub-cells representing N single units.
Sleep_spike_time_PFC: Spike timing (in sec) in a single unit in PFC during NREM sleep. Cell index 1 and 2 represents pre- and post-training sleep, respectively. In each cell, there are N sub-cells representing N single units.

Reach_spike:     
The distribution of spikes from reach onset. It is structure array in MATLAB. Each structure are N-by-4  (N trials and 4 fields). 
The field of 'spike_rate' contains the number of spike in each bin (bin size = 1 ms) from reach onset timing (the bin index indicating reach onset is informed in the other field of 'reach_onset_index').
The field of 'condition' shows the condition of success or fail in the reach-to-grasp task.
The field of 'bin_size' shows the bin size of the data in 'spike_rate.' It is 1 ms in this dataset.
The field of 'reach_onset_index' shows the bin index indicating reach onset. It is 201 in this dataset.
e.g., If spike_rate is 31-by-600 array, there are 31 units and 600 bins containing the number of spike in each bin. 
      If the bin (row 3, column 301) has number 2, it means there were two spikes during 1-ms window at 100 ms after reach onset.
      If the bin (row 7, column 1) has number 1, it means there were one spike during 1-ms window at -200 ms before reach onset.
      
2) Conversion to .npz : The data is stored in .mat form with multiple embedded cell arrays which made it almost impossible to work. However a conversion to .npz [NumPy-specific binary file format] proved to be fruitful in working with the data in python. The script for conversion of the data is mentioned (mat_to_csv.py).


Pipeline:

1) Spike Binning : The raw spike times for each neuron are converted into a matrix [bin_size x N_neurons]
2) Sleep Mask: The script filters out wakefulness and REM sleep by identifying periods of NREM sleep.It uses the specified NREM_LFP_REGION (default 'M1') to create a boolean mask. A 50 ms time bin is marked True (NREM) if the LFP signal in that bin contains non-zero values
3) Dimensionality Reduction: As the data is of very high dimensions [1 x 6666666], using dimensionality reduction techniques was a no brainer. I have used multiple Dimensionality reduction techniques devised for low-dimensional latent trajectories from high-dimensional, noisy time-series data. [GPFA, FPCA, PCA, Taken's embedding theorem]. The scripts of which are given.
The script filters out "silent" neurons by excluding any neuron whose average firing rate falls below MIN_RATE_HZ (default 0.05 Hz).The surviving spike trains are smoothed using a Gaussian filter (SMOOTH_SIGMA) to create continuous firing rate estimates. These smoothed rates are then aggregated into a set of population-soecific signals using the above mentioned techniques.The signals are then z-scored.
4) Matrix Construction: For a given epoch (e.g., 'post'), the population signals from the regions are vertically stacked (concatenated) to create a single, unified "State Matrix" representing the neocortical network's coordinated state during that epoch.
5) SINDy : The continuous latent trajectories from the FPCA step are fed into the PySINDy optimizer. Majority of scripts use STLSQ regression, some use SR3 regression on a polynomial library to discover the underlying differential equations at govern how these latent variables interact and evolve over time.
6) The final results are saved in an output directory, and the results are plotted.

Methodology: 
The above methodology is followed while implementing the two approaches: population dynamics and singular neuron dynamics.

Results:
The equations output across all methodologies and techniques implemented showcase the trend given in the paper, the equations showcase clear evidence of manifold consolidation.
The metric used to check the fidelty of the output was R2.
  


The data availability is : https://zenodo.org/records/7226711
