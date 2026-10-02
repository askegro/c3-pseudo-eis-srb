# Simulation and post-processing code for 'Pseudo-Random Switching of Cell Groups for On-Board Impedance Identification in Self-Reconfigurable Batteries'

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23111635.svg)](https://doi.org/10.5281/zenodo.23111635)

Code accompanying the paper "Pseudo-Random Switching of Cell Groups for
On-Board Impedance Identification in Self-Reconfigurable Batteries"
(A. Škegro, A. Laurin, V. Heiries, R. Thomas, C. Zou), submitted to the
2027 American Control Conference (ACC). Python implementation of
pseudo-EIS: a cell group of a self-reconfigurable battery is switched in
and out of the current path with a maximum-length sequence, the impedance
is estimated with a period-synchronous cross-spectral estimator, and the
resistances of an equivalent circuit are fitted by non-negative least
squares. The truth model is the Doyle-Fuller-Newman model in
[PyBaMM](https://pybamm.org/) with the Chen2020 parameter set, scaled to
60 Ah.

The stored results in [`reference/`](reference) are the ones used in the
manuscript. `python run.py report reference` regenerates every reported
number, table and figure from them in a few seconds, without running a
simulation.

## Repository structure

```
run.py                 # entry point: quick | full | report | test
config.json            # settings of the manuscript runs (DFN truth model)
config_quick.json      # same study on a fast linear surrogate
prbsid/
├── sequences.py       # maximum-length sequences, line spectrum, cascading rule
├── acquisition.py     # sampling and anti-aliasing filter
├── estimator.py       # cross-spectral estimate and coherence-based variance
├── fit.py             # non-negative least-squares fit, baselines
├── truth_pybamm.py    # DFN truth model (PyBaMM)
├── truth_ecm.py       # linear surrogate used by the quick run
├── experiments.py     # main study: maps, ageing, amplitude, drive cycle, pack
├── extra.py           # extension study: Monte Carlo at every map point, pulse tests
├── tasks.py  execute.py            # task list and runner
└── summarize.py  summarize_extra.py  plots.py   # reports, tables, figures
tests/test_theory.py   # numerical checks of the lemmas, propositions and the code
data/                  # WLTC power profile (shape of the drive-cycle load)
reference/             # stored results of the manuscript runs and their reports
slurm/                 # SLURM scripts for running the tasks on a cluster
start.sh  start.bat    # one-click setup, test and quick run
```

## Quick start

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt   # Windows: .venv\Scripts\python

python run.py test               # 12 numerical checks of the theory (seconds)
python run.py report reference   # manuscript numbers, tables and figures from the stored results
python run.py quick              # whole pipeline on the linear surrogate (about 2 minutes)
python run.py full --jobs 4      # all DFN simulations of the manuscript, then the report
```

`start.sh` (Linux, macOS) and `start.bat` (Windows) build the environment
and run `test` and `quick` in one step.

`report reference` writes

- `reference/main/RESULTS.md`, `summary.json`, and the folders `tables/`
  and `figures/` (generated locally, not stored in the repository): the
  nominal point, the resistance maps, ageing, load amplitude, drive cycle
  and pack-level results;
- `reference/extra/RESULTS.md`, `summary.json`: the Monte Carlo study at
  all 24 operating points and the pulse-test and drive-cycle comparisons.
  Its last section, "Statistics quoted in the paper", lists the statistics
  that the manuscript quotes from this study.

`quick` and `full` write to `results/quick` and `results/full` and leave
`reference/` untouched. The quick run uses a linear equivalent-circuit
surrogate in place of the DFN model, so it checks the pipeline but does
not reproduce the manuscript numbers. The stored DFN runs took about
8.6 CPU-hours in total (65 tasks of 5 to 20 minutes each); see
[`slurm/`](slurm) for running them as a job array: `slurm/submit.sh`,
`slurm/status.sh`, then `slurm/collect.sh`.

## Requirements

| Component | Requirement |
| --- | --- |
| Python | 3.12 or 3.13 (stored results: 3.12.3) |
| Packages | pinned in [`requirements.txt`](requirements.txt): NumPy 2.5.3, SciPy 1.18.1, PyBaMM 26.9.0.0, CasADi 3.8.1, Matplotlib 3.11.2 |
| Cluster runs | a SLURM cluster (optional) |

Noise realizations are seeded from `config.json`. The DFN solver results
depend slightly on platform and package versions, so a rerun reproduces
the stored values closely but not bit for bit.

`data/wltc_power_pybamm.csv` is the file `WLTC.csv` of the PyBaMM data
registry; only its shape is used, as the envelope of the load current.

## Citation

If you use this code, please cite both the paper and this software.

Paper:

```
A. Škegro, A. Laurin, V. Heiries, R. Thomas, C. Zou, "Pseudo-Random Switching
of Cell Groups for On-Board Impedance Identification in Self-Reconfigurable
Batteries," submitted to the 2027 American Control Conference (ACC), 2026.
```

Software:

```
A. Škegro, "Simulation and post-processing code for 'Pseudo-Random Switching
of Cell Groups for On-Board Impedance Identification in Self-Reconfigurable
Batteries'," Zenodo, 2026.
https://doi.org/10.5281/zenodo.23111635
```

This DOI covers all versions and resolves to the latest one; version 1.0.0
is https://doi.org/10.5281/zenodo.23111636.

Machine-readable metadata: [`CITATION.cff`](CITATION.cff).

## Licence

MIT: see [`LICENSE`](LICENSE). If you use this code, please cite the
associated paper and software above.
