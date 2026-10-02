# Results

Backend: pybamm. Versions: {"numpy": "2.5.3", "python": "3.12.3", "scipy": "1.18.1", "pybamm": "26.9.0.0"}

## Nominal point

| Fit | R_tot (mOhm) | R after 10 s (mOhm) | sd | RMSE, pulse at the load current (mV) | RMSE, WLTC window (mV) | RMSE, 2C pulse (mV) |
|---|---|---|---|---|---|---|
| Proposed, half-decade grid | 4.003 | 2.788 | 0.0027 | 1.96 | 3.81 | 83.58 |
| Proposed, decade grid | 4.949 | 2.759 | 0.0026 | 5.38 | 6.66 | 106.72 |
| Unweighted, half-decade grid | 4.021 | 2.779 | 0.0052 | 2.00 | 3.65 | 81.60 |
| Nonlinear routine, decade grid | 4.803 | 2.769 | -- | 4.07 | 5.22 | 100.61 |
| Fit of the reference EIS, half-decade grid | 4.204 | 2.988 | 0.0051 | 3.24 | 5.13 | 104.53 |
| Fit of the reference EIS, decade grid | 5.196 | 2.968 | 0.0131 | 7.57 | 9.95 | 128.00 |
| Pulse test, half-decade grid | 3.491 | 2.445 | -- | 7.76 | 9.32 | 46.07 |
| Pulse test, decade grid | 4.029 | 2.468 | -- | 6.09 | 5.72 | 53.67 |
| Nonlinear routine, free time constants, 5 random starts | 3.982 to 4.413 | 2.791 to 2.792 | -- | 0.44 to 1.54 | 2.25 to 3.58 | 84.08 to 87.62 |
| Recursive least squares, 1 RC, no excitation | 3.364 | -- | -- | 8.23 | 10.19 | 69.18 |
| Recursive least squares, 2 RC, no excitation | 3.533 | -- | -- | 6.00 | 8.18 | 74.65 |
| Batch fit, decade grid, no excitation | 4.528 | 2.797 | -- | 1.63 | 2.74 | 92.62 |
| Batch fit, half-decade grid, no excitation | 4.511 | 2.796 | -- | 1.79 | 2.80 | 94.99 |

Validation records: pulse at the load current: rms current 16.3 A, rms overpotential 57.9 mV; WLTC window: rms current 19.0 A, rms overpotential 62.6 mV; 2C pulse: rms current 97.7 A, rms overpotential 258.5 mV.
Impedance error against the reference EIS: mean 4.27 % with noise, 3.16 % noise-free (per band, noise-free: slow 7.15 %, mid 2.25 %, fast 0.05 %); mean coherence 0.99793.
Misfit of the model to the reference EIS (mean relative): decade grid 10.6 %, half-decade grid 1.9 %.
Monte Carlo, 500 runs. Predicted over empirical variance of the impedance estimate: slow 1.102, mid 1.030, fast 0.998.
Two-sigma coverage: 95.3 % against the noise-free estimate (per-line variance 91.9 %), 60.4 % against the reference EIS.
Fitted resistances, empirical / predicted standard deviation (mOhm): R_tot, decade grid 0.0436 / 0.0429 (coverage 94.0 %); R_10, decade grid 0.0067 / 0.0028 (coverage 59.6 %); R_10, half-decade grid 0.0030 / 0.0029 (coverage 94.8 %).
Nonlinear routine with fixed time constants: largest parameter difference to the non-negative least-squares solution 3.77e-04 mOhm, cost ratio 1.000099, 8 function evaluations.

## Maps

24 operating points. Resistance after 10 s (half-decade grid), relative difference to the fit of the reference EIS: proposed mean 9.92 % (max 24.60 %), proposed with the BMS open-circuit arm 9.89 %, pulse test mean 23.47 % (max 47.15 %).
R_tot on the decade grid, as in the preliminary paper: proposed mean 8.28 % (max 25.58 %), pulse test mean 25.66 %.
Impedance error against the reference EIS, mean over points: 5.74 % (noise-free 4.64 %).
Median voltage error over the points (mV): pulse at the load current: proposed 5.45, proposed_dense 1.73, eis_reference_dense 3.34, pulse_test_dense 8.72; WLTC window: proposed 7.47, proposed_dense 3.89, eis_reference_dense 5.72, pulse_test_dense 9.88; 2C pulse: proposed 81.68, proposed_dense 76.90, eis_reference_dense 95.61, pulse_test_dense 49.75.
All validation records ran to the end.

## Ageing

| SOH | capacity (Ah) | R after 10 s (mOhm) | 2 sd | same, EIS fit | RMSE pulse at the load current: BOL model / re-identified (mV) | RMSE WLTC window: BOL model / re-identified (mV) | RMSE 2C pulse: BOL model / re-identified (mV) |
|---|---|---|---|---|---|---|---|
| 1.00 | 61.8 | 2.041 | 0.005 | 2.092 | 1.00 / 1.00 | 1.22 / 1.22 | 24.82 / 24.82 |
| 0.95 | 58.7 | 2.185 | 0.005 | 2.250 | 2.24 / 0.21 | 2.24 / 0.96 | 11.40 / 23.62 |
| 0.90 | 55.7 | 2.352 | 0.006 | 2.424 | 5.34 / 0.42 | 5.70 / 0.95 | 14.22 / 25.76 |
| 0.85 | 52.6 | 2.537 | 0.005 | 2.620 | 8.85 / 0.30 | 9.59 / 1.21 | 31.46 / 28.10 |
| 0.80 | 49.5 | 2.749 | 0.006 | 2.839 | 12.79 / 0.41 | 13.96 / 1.43 | 52.05 / 30.70 |

## Load amplitude

| Load (A) | impedance error | noise-free | SOC span, slow band | RMSE at the load current: proposed / EIS fit (mV) |
|---|---|---|---|---|
| 5 | 6.07 % | 0.43 % | 2.3 % | 0.40 / 0.52 |
| 10 | 3.63 % | 1.14 % | 4.6 % | 1.16 / 1.01 |
| 20 | 4.27 % | 3.16 % | 9.2 % | 1.96 / 3.24 |
| 40 | 8.29 % | 7.78 % | 18.4 % | 6.17 / 15.09 |
| 60 | 12.12 % | 11.82 % | 27.6 % | 12.20 / 33.89 |

## Drive cycle

Load: rms 20.0 A, mean 9.6 A, from -43.1 to 70.5 A.

| Band | load rms in the record (A) | error, WLTC | noise-free | error, constant load | noise-free | coherence, WLTC |
|---|---|---|---|---|---|---|
| slow | 19.0 | 15.01 % | 15.00 % | 7.15 % | 7.15 % | 0.99430 |
| mid | 11.3 | 2.91 % | 2.15 % | 2.55 % | 2.25 % | 0.99811 |
| fast | 11.5 | 5.54 % | 0.04 % | 3.09 % | 0.05 % | 0.98172 |

Voltage error of the fit identified under the WLTC load / under the constant load (half-decade grid, mV): pulse at the load current: 4.38 / 1.96; WLTC window: 4.71 / 3.81; 2C pulse: 94.32 / 83.58.

## Pack level

| Case | Band | overpotential of the inserted group, min to max (mV) | bound of Proposition 3 (mV) | inside | OCV difference (mV) | voltage p-p (mV) | step if one group is toggled alone (V) |
|---|---|---|---|---|---|---|---|
| matched | slow | 19.6 to 74.4 | 4.2 to 78.5 | yes | 0.6 | 137.1 | 3.706 |
| matched | mid | 6.6 to 59.3 | 4.2 to 78.5 | yes | 0.0 | 55.0 | 3.718 |
| matched | fast | 4.3 to 33.0 | 4.2 to 78.5 | yes | 0.0 | 28.8 | 3.726 |
| soc_mismatch | slow | 19.6 to 74.7 | 4.2 to 80.8 | yes | 18.2 | 154.7 | 3.706 |
| soc_mismatch | mid | 6.6 to 59.3 | 4.2 to 80.8 | yes | 19.4 | 74.3 | 3.718 |
| soc_mismatch | fast | 4.3 to 33.0 | 4.2 to 80.8 | yes | 19.4 | 48.1 | 3.726 |
| resistance_mismatch | slow | 19.6 to 81.6 | 4.3 to 85.3 | yes | 0.5 | 147.9 | 3.706 |
| resistance_mismatch | mid | 6.6 to 64.8 | 4.3 to 85.3 | yes | 0.0 | 60.3 | 3.718 |
| resistance_mismatch | fast | 4.3 to 35.7 | 4.3 to 85.3 | yes | 0.0 | 31.4 | 3.726 |
| both | slow | 19.6 to 82.4 | 4.2 to 86.9 | yes | 18.1 | 156.9 | 3.706 |
| both | mid | 6.6 to 64.8 | 4.2 to 86.9 | yes | 19.4 | 73.7 | 3.718 |
| both | fast | 4.3 to 35.6 | 4.2 to 86.9 | yes | 19.4 | 47.6 | 3.726 |

Both groups identified in the same test, mean impedance error: matched: p 4.28 %, q 4.26 %; soc_mismatch: p 4.25 %, q 4.26 %; resistance_mismatch: p 4.29 %, q 4.69 %; both: p 4.24 %, q 4.65 %.
