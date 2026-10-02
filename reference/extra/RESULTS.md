# Extension study
map points complete: 24 / 24, drive points complete: 3 / 3

## PRBS test at every map point (half-decade grid, R10, Monte Carlo)
| quantity | median | min | max |
|---|---|---|---|
| bias vs reference (%) | -6.965 | -24.534 | -1.499 |
| sd empirical (% of R10) | 0.101 | 0.052 | 0.125 |
| sd predicted / empirical | 1.244 | 0.672 | 2.999 |
| coverage of 2 sd vs reference | 0.000 | 0.000 | 0.000 |
| R_tot (decade) sd predicted / empirical | 0.729 | 0.087 | 3.170 |
| R10 (decade) sd predicted / empirical | 0.472 | 0.087 | 2.906 |

PRBS test duration: 2100 s

## Pulse tests at every map point (half-decade grid, R10)
| protocol | duration (s) | median bias (%) | median sd (%) | median sd pred/emp | median coverage 2 sd | median RMSE to ref (%) |
|---|---|---|---|---|---|---|
| p20_single (24 pts) | 101 | -4.71 | 0.025 | 1.20 | 0.00 | 4.71 |
| p60_single (24 pts) | 101 | -19.94 | 0.008 | 4.16 | 0.00 | 19.94 |
| p20_x21 (24 pts) | 2101 | -4.25 | 0.012 | 1.17 | 0.00 | 4.25 |
| p60_x21 (24 pts) | 2101 | -19.54 | 0.004 | 4.02 | 0.00 | 19.54 |

## Validation RMSE (mV), median over map points, half-decade grid
- pulse_load: prbs 1.72, p20_single 1.72, p60_single 8.72, p20_x21 2.38, p60_x21 7.43
- pulse_two_c: prbs 77.19, p20_single 78.73, p60_single 49.83, p20_x21 83.22, p60_x21 51.40
- wltc: prbs 4.40, p20_single 3.87, p60_single 9.88, p20_x21 3.95, p60_x21 8.59

## Drive cycle: batch fit of the load alone against the PRBS test (R10, half-decade grid)
| T (C) | method | duration (s) | bias (%) | sd (%) | sd pred/emp | coverage 2 sd | val WLTC RMSE (mV) |
|---|---|---|---|---|---|---|---|
| 10 | batch forward_1000s | 1000 | -18.44 | 0.013 | 10.03 | 0.00 | 7.13 |
| 10 | batch reversed_2100s | 2100 | -32.84 | 0.004 | 24.32 | 0.00 | 10.92 |
| 10 | PRBS under WLTC | 2100 | -23.16 | 0.092 | 11.14 | 0.00 | 7.40 |
| 25 | batch forward_1000s | 1000 | -6.45 | 0.020 | 3.42 | 0.00 | 2.79 |
| 25 | batch reversed_2100s | 2100 | -15.54 | 0.006 | 13.21 | 0.00 | 4.13 |
| 25 | PRBS under WLTC | 2100 | -11.10 | 0.081 | 8.18 | 0.00 | 4.27 |
| 40 | batch forward_1000s | 1000 | -2.49 | 0.032 | 1.38 | 0.00 | 1.90 |
| 40 | batch reversed_2100s | 2100 | -6.78 | 0.009 | 9.97 | 0.00 | 1.50 |
| 40 | PRBS under WLTC | 2100 | -5.33 | 0.093 | 3.83 | 0.00 | 1.94 |

## Mean absolute R10 error against the reference EIS by temperature (%), half-decade grid
| T (C) | pseudo-EIS | p20_single | p60_single | p20_x21 | p60_x21 |
|---|---|---|---|---|---|
| 40 | 2.6 | 1.6 | 8.5 | 1.1 | 7.9 |
| 25 | 6.9 | 5.0 | 20.8 | 4.6 | 20.4 |
| 10 | 20.2 | 17.1 | 41.1 | 16.9 | 40.9 |

## Statistics quoted in the paper
- predicted / empirical standard deviation of R10 (half-decade grid): min 0.67, median 1.24, max 3.00
- predicted / empirical standard deviation of R_tot (decade grid): min 0.09, median 0.73, max 3.17
- empirical standard deviation of R10: 0.05 to 0.12 % of its value; bias over standard deviation: at least 12.0
- map points at which a pulse test has the smaller R10 error than pseudo-EIS: p20_single 24 of 24, p60_single 0 of 24, p20_x21 24 of 24, p60_x21 0 of 24
- median predicted / empirical standard deviation of the pulse fit: p20_single 1.20, p60_single 4.16, p20_x21 1.17, p60_x21 4.02
- largest difference between the single and the repeated 20 A pulse: 0.75 percentage points
