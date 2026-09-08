# Roman-Numerals

## Certifying Hidden Dissipation from Observed Current Fluctuations

Reproducible numerical code and figure data for the manuscript by **Ahmed Roman**.

Repository: [AhmedHRoman/Roman-Numerals](https://github.com/AhmedHRoman/Roman-Numerals).

Affiliations: Dana-Farber Cancer Institute; Broad Institute of MIT and Harvard;
Harvard Medical School. The author is a Damon Runyon Quantitative Biology Fellow.

This package contains synthetic Markov-network calculations and simulated
trajectories. It contains no patient data, experimental participant data,
manuscript drafts, referee correspondence, or third-party paper PDFs.

## Reproduce the figures

Tested with Python 3.12.7. From this directory:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python make_caricature_figure.py
python make_main_figure.py
python make_kinesin_figure.py
python make_finite_sample_figure.py
python submission_audit/validate.py
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.
For a headless machine, set `MPLBACKEND=Agg`. A LaTeX installation is not required.
Run scripts from this directory because data paths are relative to it.
The finite-sample script can take several minutes. Each script overwrites its
own generated figure/data files, not the source code.

## Contents

| File | Purpose |
| --- | --- |
| `kinesin_step1.py` | Markov-network engine, kinesin rates, and thermodynamic checks |
| `make_caricature_figure.py` | Figure 1 schematic |
| `make_main_figure.py` | Figure 2 exact-theory curves and CSV data |
| `make_kinesin_figure.py` | Figure 3 kinesin curves, finite-sample inset, and NPZ data |
| `make_finite_sample_figure.py` | Supplement Figure S1 and NPZ data |
| `build_finite_sample.py` | Additional synthetic-network covariance diagnostic |
| `submission_audit/validate.py` | Independent covariance, bound, and consistency checks |
| `submission_audit/post_cleanup_results.json` | Saved independent validation output |
| `fig_*.pdf`, `fig_*.png` | Generated figures, not manuscript PDFs |
| `fig_main_panel*.csv` | Figure 2 statistics and network parameters |
| `fig_kinesin_data.npz` | Force curves, means/SDs, parameters, and sampling metadata |
| `fig_finite_sample_data.npz` | Synthetic-network parameters and means/SDs for both panels |
| `fig_kinesin_sim_stall_exact.npz` | Validated finite-sample kinesin cache |
| `SHA256SUMS.txt` | Checksums of the distributed files |

NPZ files contain numeric arrays and non-executable metadata. Inspect them with:

```python
import numpy as np
with np.load("fig_finite_sample_data.npz", allow_pickle=False) as data:
    print(data.files)
    print(data["total_times"], data["panel_a_mean"], data["panel_a_std"])
```

The first line of `fig_main_panelA.csv` stores exact sigma and gamma; the second
line is the column header. Other CSV files begin directly with a header.
`ceiling_over_sigma` in the panel-B CSV means the **hybrid** full-observation
level, not the operational maximum, which equals sigma at full observation.

## Sampling and interpretation

- The synthetic network uses the five oriented edges and forward/reverse rates
  stored in its NPZ. Observed edges are indices 2 and 4 (zero-based).
- Figure S1(a) uses 28 replicates, block length 40, and seeds
  `1000 + 10*total_time + replicate_index`. Panel (b) uses total time 9600 and
  seeds `7000 + 7*block_length + replicate_index`.
- Kinesin uses 1000 micromolar ATP, 1 micromolar ADP, 1 micromolar phosphate,
  14 replicates, 50-second blocks, and seeds `100 + total_time + replicate_index`.
  The observed edges are the mechanical edge and chemical edge index 1.
- Error bars are standard deviations across replicates, not standard errors
  or guaranteed confidence bounds. Raw jump histories are not retained;
  fixed seeds, rates and scripts regenerate the sampled statistics.
- Fixed blocks retain a finite-window bias. In the synthetic example, the
  population plug-in hybrid level at block length 40 is 0.2698335, versus the
  long-time bound 0.2757955. At kinesin stall the corresponding covariance
  levels are 45.6793 and 45.8168 per second. Increasing only the number of
  fixed-length blocks does not remove this bias.
- Cycle observability recovers the full density-contracted quadratic cost,
  not generally the total entropy-production rate. The total rate at kinesin
  stall is approximately 331.9165 per second in units of Boltzmann's constant.
- Both statistical estimators use a relative pseudoinverse cutoff of 0.001.
  Finite-time plug-in estimates are not guaranteed thermodynamic lower bounds.

## Kinesin model provenance

The six-state model follows the Carter-Cross parameter row of Liepelt and
Lipowsky (2007), with the load factors discussed in both source papers.
To enforce exact balance with rounded tabulated constants, the implementation
keeps kappa_65=0.02 and derives kappa_16=0.0255102041 (both in inverse
micromolar seconds). Table I instead supplies rounded kappa_16=0.02 and marks
kappa_65 as balance-derived. This convention is documented in the supplement.

- S. Liepelt and R. Lipowsky, *Kinesin's Network of Chemomechanical Motor Cycles*,
  [Phys. Rev. Lett. 98, 258102 (2007)](https://doi.org/10.1103/PhysRevLett.98.258102).
- S. Liepelt and R. Lipowsky, *Operation modes of the molecular motor kinesin*,
  [Phys. Rev. E 79, 011917 (2009)](https://doi.org/10.1103/PhysRevE.79.011917).

## Checks

The independent audit uses a Poisson-equation/martingale covariance calculation
rather than reusing the density-contraction formula. It tests 100 random
networks, all their nonempty observation subsets, 41 kinesin forces, cycle
affinities, energy balance, and an independent finite-window covariance formula.
It also performs a fresh stochastic check; this is not an exhaustive proof or
a guarantee for arbitrary ill-conditioned numerical inputs.

See `CITATION.cff` for citation metadata.

This repository uses the MIT license selected by the author. See `LICENSE`.
