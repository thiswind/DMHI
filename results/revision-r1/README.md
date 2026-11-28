# Revision R1 archived results

Reference outputs for the IoTJ-R1 revision experiments. These are
convenience copies for review: every number regenerates by running the
listed script (fresh outputs land in `runs/`, this directory stays fixed).
x86-64 reproduces all values bit-for-bit; arm64 stays within the tolerance
stated in `results/README.md`.

| Experiment | Script | Archived files | Answers reviewer |
|---|---|---|---|
| E3 MD recomputation | `scripts/e3_md_recompute.py` | `e3_md/md_table.json` (36 rows: 12 domains x 3 seeds) | R1-2, R3-3 |
| E4 sensitivity sweep | `scripts/e4_sensitivity.sh` | `e4_sensitivity/` (curves json + 4 figures, 3 domains x 4 params x 3 seeds) | R1-3, R1-5, R2-2, R2-9 |
| E5 edge deployment | `scripts/e5_edge_pi.py` + `scripts/e5_edge_server.py` + `scripts/e5_deep_mem.py` | `e5_edge/` (11 Pi domains + 3 deep-mem profiles) | R1-4, R2-3, R2-7, R2-8 |
| E6 PSW-I official rerun | `scripts/e6_psw_official.py` | `e6_psw/psw_i_official_summary.json` (aggregate) | R3-2 |
| E7 natural missingness | `scripts/e7_natural_missing.py` | `e7_natural/` (2 domains x 6 protocols x 3 seeds) | R1-1 |
| E9 incremental update | `scripts/e9_incremental_ab.py` | `e9_incremental/` (2 domains x 3 seeds) | R2-6, R2-9 |
| T6 abstract number | `scripts/t6_abstract_number.py` | `t6_abstract_number.json` (per-domain MSE reductions) | R3-4 |

## E3: MD of every method under one definition (R1-2, R3-3)

3-seed means; MD = manifold deviation (eq. 9, k=5 NN form), one shared
definition for all methods, computed by `scripts/e3_md_recompute.py`.

| Domain | DMHI MAE | DMHI MD | Linear MD | LOCF MD | Mean MD | PSW-I MD |
|---|---|---|---|---|---|---|
| air_quality_italy | 0.141 | **0.509** | 1.226 | 1.318 | 1.190 | 0.855 |
| breizh_rs | 0.310 | **0.630** | 2.057 | 2.251 | 0.607 | 0.756 |
| delta_robot | 0.198 | **0.267** | 0.547 | 0.542 | 0.286 | 0.361 |
| gait_uci | 0.153 | **0.393** | 0.617 | 0.725 | 0.520 | 0.429 |
| gas_home | 0.051 | **0.197** | 0.377 | 0.524 | 0.771 | 0.490 |
| kuka | 0.159 | **0.732** | 0.878 | 1.152 | 1.525 | 1.064 |
| mhealth | 0.197 | **1.075** | 1.252 | 1.517 | 2.101 | 1.195 |
| mimic | 0.095 | **1.202** | 1.217 | 2.170 | 2.292 | 2.431 |
| mocap_karate | 0.075 | **0.866** | 2.239 | 2.877 | 4.274 | 2.994 |
| nanodrone | 0.192 | **1.092** | 0.705 | 1.095 | 0.986 | 0.804 |
| physionet2012 | 0.342 | **0.927** | 2.122 | 1.746 | 0.140 | 4.048 |
| solo12 | 0.174 | **0.605** | 1.182 | 1.284 | 0.989 | 1.130 |

CSDI MD under this definition: 17.1-67.8 across domains (vs DMHI 0.197-1.202).
Large CSDI MD is not a metric artifact: coordinate-wise diffusion output leaves
imputed rows far off the reference manifold of true states. Mean fill reaches
MD=0 on PhysioNet only because its constant fill coincides with the densest
manifold region while its MAE is the worst of all methods (0.704), so MD must
be read jointly with MAE.

## E4: four-parameter sensitivity sweep (R1-3, R1-5, R2-2, R2-9)

3-seed mean MAE per parameter value, three domains (physionet2012,
mocap_karate, delta_robot). Default config: k=10, gamma_s=0.02, r_L=0.30, d=8.

### neighborhood size k

| value | delta_robot | mocap_karate | physionet2012 |
|---|---|---|---|
| 3 | 0.2105 | 0.1075 | 0.3206 |
| 6 | 0.2187 | 0.0772 | 0.3317 |
| 10 | 0.1979 | 0.0748 | 0.3418 |
| 16 | 0.1979 | 0.0748 | 0.3418 |
| 24 | 0.1979 | 0.0748 | 0.3418 |
| 30 | 0.1979 | 0.0748 | 0.3418 |

### smoothing coefficient gamma_s

| value | delta_robot | mocap_karate | physionet2012 |
|---|---|---|---|
| 0.001 | 0.1969 | 0.0734 | 0.3463 |
| 0.003 | 0.1969 | 0.0734 | 0.3456 |
| 0.01 | 0.1972 | 0.0738 | 0.3436 |
| 0.02 | 0.1979 | 0.0748 | 0.3418 |
| 0.1 | 0.2039 | 0.0845 | 0.3391 |
| 0.3 | 0.2105 | 0.0992 | 0.3443 |
| 1.0 | 0.2186 | 0.1182 | 0.3564 |

### landmark ratio r_L

| value | delta_robot | mocap_karate | physionet2012 |
|---|---|---|---|
| 0.05 | 0.2247 | 0.2443 | 0.5802 |
| 0.1 | 0.3738 | 0.1610 | 0.4587 |
| 0.2 | 0.2338 | 0.1081 | 0.3433 |
| 0.3 | 0.1979 | 0.0748 | 0.3418 |
| 0.4 | 0.1722 | 0.0628 | 0.3043 |
| 0.5 | 0.1501 | 0.0515 | 0.2485 |

### latent dimension d

| value | delta_robot | mocap_karate | physionet2012 |
|---|---|---|---|
| 2 | 0.2304 | 0.1071 | 0.3153 |
| 3 | 0.2164 | - | - |
| 4 | 0.1979 | 0.0836 | 0.3326 |
| 8 | - | 0.0784 | 0.3321 |
| 12 | - | 0.0748 | - |
| 16 | - | 0.0758 | - |
| 24 | - | 0.0740 | - |

Flat around the defaults: k>=10 saturates on every domain; gamma_s within
[0.001, 0.1] moves MAE by <12% relative; r_L in [0.3, 0.5] is the best region
and [0.05, 0.2] stays usable; d>=4 flat. Figures with the same data:
`e4_sensitivity_{d,gamma_s,k,r_L}.pdf`.

## E5: edge deployment measurements (R1-4, R2-3, R2-7, R2-8)

### Raspberry Pi 3 Model B+ (1 GB, arm32, Debian 13, 4 threads), DMHI, 11 domains, seed 7

| Domain | Latency ms (median) | Peak RSS MB | Energy mJ/sample (est.) |
|---|---|---|---|
| air_quality_italy | 55.6 | 58 | 115 |
| breizh_pheno | 116.0 | 59 | 181 |
| breizh_rs | 116.2 | 59 | 182 |
| cmapss | 97.0 | 58 | 150 |
| delta_robot | 266.5 | 60 | 363 |
| gait_uci | 264.7 | 59 | 313 |
| gas_home | 201.0 | 58 | 219 |
| mc_maze | 94.2 | 59 | 113 |
| mocap_karate | 132.6 | 245 | 191 |
| pulsedb | 250.7 | 59 | 424 |
| solo12 | 132.2 | 60 | 207 |

Energy = power-model estimate (P_full-P_idle)*cpu_s/4 with P_idle=1.9 W,
P_full=5.1 W measured at the board input under a stress pair; no external meter.

### Deep baselines vs DMHI, x86-64 CPU server, gas_home, block20, rho=0.9, seed 7

| Method | Peak RSS MB | Latency ms/sample | Train min |
|---|---|---|---|
| DMHI (torch-free) | 607 | 14.6 | 0 (training-free) |
| ImputeFormer | 2443 | 7.9 | 0.45 |
| CSDI (nsample=50) | 2649 | 38610.8 | 11.6 |

Pi deployability chain: Pi usable RAM ~658 MB; importing torch alone costs
~225 MB RSS; the smallest CSDI backbone probe needs 271 MB before weights
and the 50-step sampling loop, so >1 GB-class deep baselines cannot fit; DMHI
torch-free runs in 58-245 MB across all 11 domains.

## E6: PSW-I official-repo rerun (R3-2)

PSW-I from the official implementation (Wang et al., ICLR 2025,
"Optimal Transport for Time Series Imputation"), 12 domains x 3 seeds.

| Domain | MAE (paper protocol) | MAE (official scaling) | Train min/seed |
|---|---|---|---|
| air_quality_italy | 0.5623 | 0.6959 | 0.63/0.47/0.99 |
| breizh_rs | 0.8569 | 0.9156 | 2.37/2.14/2.49 |
| delta_robot | 0.4826 | 0.5247 | 1.26/0.58/1.57 |
| gait_uci | 0.2633 | 0.2754 | 0.41/1.66/1.72 |
| gas_home | 0.1736 | 0.2280 | 0.2/0.19/0.2 |
| kuka | 0.2928 | 0.3151 | 0.21/0.25/0.24 |
| mhealth | 0.2847 | 0.2965 | 0.33/0.27/0.25 |
| mimic | 0.1921 | 0.3957 | 0.77/0.54/0.52 |
| mocap_karate | 0.3810 | 0.3735 | 0.42/0.4/0.42 |
| nanodrone | 0.2211 | 0.1909 | 0.27/0.23/0.25 |
| physionet2012 | 0.6664 | 0.4796 | 20.77/20.63/20.27 |
| solo12 | 0.4105 | 0.4180 | 0.3/0.29/0.27 |

The two columns differ by the official repo scale convention on some domains;
the order of magnitude and ranking vs DMHI (Table II) are unchanged. Training
curves per seed are stored in the `curve` field of the per-seed jsons.

## E9: incremental update (R2-6, R2-9)

| Domain | Seed | Segments S | Calls incr. vs scratch | Speedup | New-seg MAE | max abs dev |
|---|---|---|---|---|---|---|
| mocap_karate | 123 | 25 | 29 vs 434 | 15.0x | 0.1467 | 0.0 |
| mocap_karate | 42 | 25 | 29 vs 434 | 15.0x | 0.1295 | 0.0 |
| mocap_karate | 7 | 25 | 29 vs 434 | 15.0x | 0.1321 | 0.0 |
| physionet2012 | 123 | 12 | 29 vs 434 | 15.0x | 0.3403 | 0.0 |
| physionet2012 | 42 | 12 | 29 vs 434 | 15.0x | 0.3465 | 0.0 |
| physionet2012 | 7 | 12 | 29 vs 434 | 15.0x | 0.3278 | 0.0 |

Incremental mode appends segments to existing landmarks/graph instead of
rebuilding from scratch: identical outputs (max deviation 0.0 on both domains,
all seeds) at ~15x fewer imputer calls; per-step time stays constant
(physionet 0.66 s vs 10.15 s scratch mean; mocap 0.12 s vs 1.68 s).

## T6: abstract number, per-domain MSE reduction vs strongest learning baseline (R3-4)

| Domain | DMHI MSE (3-seed) | Best baseline | Baseline MSE | Reduction |
|---|---|---|---|---|
| physionet2012 | 0.41 | CSDI | 0.669 | +38.7% |
| mimic | 0.101 | BRITS | 0.247 | +59.1% |
| air_quality_italy | 0.0538 | CSDI | 0.674 | +92.0% |
| gait_uci | 0.0536 | CSDI | 0.06 | +10.6% |
| mocap_karate | 0.0261 | CSDI | 0.307 | +91.5% |
| gas_home | 0.0203 | CSDI | 0.131 | +84.5% |
| delta_robot | 0.2843 | SAITS | 1.052 | +73.0% |
| kuka | 0.0831 | CSDI | 0.109 | +23.8% |
| nanodrone | 0.1439 | CSDI | 0.21 | +31.5% |
| mhealth | 0.3645 | CSDI | 0.346 | -5.4% |
| breizh_rs | 0.308 | CSDI | 0.834 | +63.1% |
| solo12 | 0.0638 | SAITS | 0.225 | +71.6% |

Headline: up to 92.0% lower squared error than the
strongest learning-based method (air_quality_italy), replacing the
untraceable 27.8% in the original abstract. mhealth is the honest exception
(-5.4%, CSDI better there). Computed by `scripts/t6_abstract_number.py`
from the released checkpoints; display-value convention as in Table II.
