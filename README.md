# SR-AttELM-H: two-time-scale online refinement for causal reconstruction under changing arrival-delay regimes

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ALAXAY/sr-attelm-h-adaptation/blob/main/notebooks/SR_AttELM_H_adaptation_Colab.ipynb) [![DOI](https://zenodo.org/badge/1430261396.svg)](https://doi.org/10.5281/zenodo.23579401)

Code for the article "Causal reconstruction of asynchronously delivered sensor data under changing arrival-delay regimes: availability-aware attention with two-time-scale online refinement of a sector-structured output layer" (O. Fesenko, manuscript, 2026). Ukrainian description: [README_UA.md](README_UA.md).

## Purpose

The code reproduces the three experiments of the article. At every decision time the state estimate is formed only
from the messages received by that time, each message carrying the generation time of the measurement and the
arrival time of each of its components. Availability-aware spatiotemporal attention weights the neighbouring sources
by the received fraction, the age of information and the mean arrival delay; a recurrent extreme learning machine
with a sector-structured output layer forms the correction. When the verified value arrives, the output layer is
refined by a two-time-scale procedure: a per-source offset with a fast forgetting factor (λ_b = 0.99) follows the
shift of the error mean after a change of the delay regime, and sector-wise recursive least squares (RLS) with a slow
forgetting factor refine the output matrix of the active sector. The attention block and the hidden layer stay fixed.

## Relation to ALAXAY/sr-attelm-h

The evaluation protocol, the stages of the technique, the delay models and the AGG adapter are taken without changes
from [ALAXAY/sr-attelm-h](https://github.com/ALAXAY/sr-attelm-h) v1.0.0 (https://doi.org/10.5281/zenodo.22975926),
which accompanied an earlier version of the manuscript. This repository adds the two-time-scale refinement
(`srattelm/adapt_experiment2.py`, function `stream2`), the experiment on adaptation of the output layer, adapted
reproductions of GRU-D, BRITS, SAITS and GRIN under the same causal protocol, the summary scripts of the article
tables and a Lean 4 file with a formal check of expressions (1)–(36).

## Contents

| Folder | Contents |
|---|---|
| `srattelm/` | evaluation protocol with two time coordinates per message, stages 1–7 and the preliminary stage of the technique, two-time-scale refinement, adaptation experiment, adapted reproductions of GRU-D, BRITS, SAITS and GRIN, LOCF, result tables |
| `srattelm/agg/` | adapter of the official Asynchronous Graph Generator (AGG) core to the same protocol (GPL-3.0) |
| `netsim/` | packet-level model of an intermittently connected monitoring network, network delay realizations, delay statistics and Fig. 3 of the article |
| `notebooks/` | Google Colab notebook that reruns the experiments |
| `supplementary/` | Lean 4 / Mathlib file with the formal check of expressions (1)–(36) |
| `data/` | location of the Beijing Multi-Site Air Quality archive; the data set is not redistributed |

## Experiments and scripts

| Article | Scripts (in order) | Output | Time, 8 CPU threads |
|---|---|---|---|
| Quick check | `srattelm/check_reproduction.py` | console | about 2 min |
| Experiment 3: adaptation of the output layer (Tables 9–12, Fig. 5) | `adapt_experiment.py`, `adapt_experiment2.py`, `adapt_experiment3.py`, `summarize_adaptation.py` | `results_adaptation.csv`, `results_adaptation2.csv`, `results_adaptation3.csv`, `summary_adaptation.csv`, `paired_adaptation.csv` | about 20 min |
| Experiment 1: comparison with existing methods (Tables 6, 7, Fig. 4) | `baselines.py`, `agg/run_agg.py`, `summarize_methods.py` | `results_baselines.csv`, `results_official_agg.csv`, `summary_methods.csv`, `paired_methods.csv` | 26 min; AGG 20–25 min per split and delay model (12 threads) |
| Experiment 2: contribution of the components (Table 8) | `run_experiments.py`, `run_component_check.py`, `summarize.py`, `summarize_components.py` | `results_geometric.csv`, `results_network.csv`, `results_component_check.csv`, `summary_network.csv`, `availability_check_summary.csv` | about 6 min for `run_experiments.py` |
| Delay statistics (Table 4, Section 7.1, Fig. 3) | `netsim/delay_statistics.py`, `netsim/make_figure.py` | `netsim/out/` | below 1 min |

`summarize_methods.py` reads the rows of the technique from the result tables of experiment 3, so experiment 3 is
run first. The scripts `adapt_experiment*.py`, `baselines.py` and `agg/run_agg.py` resume after an interruption from
their CSV output; to recompute an experiment from the beginning, rename its shipped CSV files first (the notebook
does this automatically). The number of PyTorch threads is set by
the environment variable `SRATTELM_THREADS` (default 8); the run uses the CPU (`CUDA_VISIBLE_DEVICES=`).

## Running

In Google Colab, open `notebooks/SR_AttELM_H_adaptation_Colab.ipynb` with the badge above and run all cells. The notebook clones the repository, downloads
the data set from the UCI repository, checks the SHA-256 of the data array and the evaluation protocol, runs the quick
check and, depending on the flags of its first cell, the experiments.

Locally (Python 3.10 or later):

```
pip install -r requirements.txt
curl -L -o data/beijing_multi_site_air_quality.zip https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip
cd srattelm
python check_reproduction.py     # quick check against the shipped result tables
python adapt_experiment.py       # experiment 3: frozen layer, global RLS, sector RLS, per-source offset
python adapt_experiment2.py      # experiment 3: two-time-scale refinement, output layer with intercept
python adapt_experiment3.py      # experiment 3: sensitivity to lambda_b
python summarize_adaptation.py   # Tables 9-12
python baselines.py              # experiment 1: GRU-D, BRITS, SAITS, GRIN
python agg/run_agg.py            # experiment 1: official AGG core (optional)
python summarize_methods.py      # Tables 6, 7
python run_experiments.py        # experiment 2: technique, variants, LOCF
python run_component_check.py    # experiment 2: component exclusion
python summarize.py
python summarize_components.py   # Table 8
cd ../netsim
python delay_statistics.py       # Table 4, delay autocorrelation
python make_figure.py            # Fig. 3 (file out/Fig8_network_delays.png)
```

The data path can be set with the environment variable `SRATTELM_DATA` (an archive at any nesting level or a folder
with the `PRSA_Data_*.csv` files). The network delay realizations are stored in `netsim/`; `generate_delays.py` and
`generate_training_delays.py` regenerate them bit for bit.

## Results

Table 1. Pooled RMSE of reconstructing delayed components over six paired runs (preliminary stage on the delay
levels 1, 2, 4 of the delay model of the test regime; `summary_methods.csv`)

| Method | Geometric ×4 | Geometric ×8 | Network-derived |
|---|---|---|---|
| Technique, two-time-scale refinement | 0.2866 | 0.3531 | 0.3884 |
| Technique, sector RLS alone | 0.2941 | 0.3637 | 0.3894 |
| GRU-D (adapted reproduction) | 0.4650 | 0.6051 | 0.6589 |
| BRITS (adapted reproduction) | 0.4732 | 0.6350 | 0.7576 |
| SAITS (adapted reproduction) | 0.3039 | 0.4048 | 0.4525 |
| GRIN (adapted reproduction) | 0.4173 | 0.5302 | 0.5347 |
| Official AGG core | 0.3451 | – | 0.4974 |
| Last observation carried forward (LOCF) | 0.5430 | 0.7300 | 0.7968 |

The technique with the two-time-scale refinement had a lower RMSE than every existing method in all six paired runs of every regime
(`paired_methods.csv`); relative to SAITS, the closest method, the pooled RMSE was lower by 5.69%, 12.79%
and 14.17%. The official AGG core was not evaluated in regime ×8.

Table 2. Pooled RMSE for models trained on geometric delays, by coverage of the preliminary stage, test regime and
refinement procedure (`summary_adaptation.csv`); κ is the share of the accuracy loss of the single-level model
recovered by the refinement, expression (36)

| Preliminary stage | Test regime | LOCF | Frozen output layer | Sector RLS alone | Per-source offset alone | Two-time-scale refinement | κ of the two-time-scale refinement, % |
|---|---|---|---|---|---|---|---|
| delay level 1 | geometric ×4 | 0.5430 | 0.3305 | 0.3192 | 0.3185 | 0.3118 | 52 |
| delay level 1 | geometric ×8 | 0.7300 | 0.4310 | 0.4033 | 0.4103 | 0.3938 | 57 |
| delay level 1 | network-derived | 0.7968 | 0.4782 | 0.4560 | 0.4675 | 0.4486 | 58 |
| delay levels 1, 2, 4 | geometric ×4 | 0.5430 | 0.2945 | 0.2941 | 0.2872 | 0.2866 | – |
| delay levels 1, 2, 4 | geometric ×8 | 0.7300 | 0.3653 | 0.3637 | 0.3545 | 0.3531 | – |
| delay levels 1, 2, 4 | network-derived | 0.7968 | 0.4267 | 0.4214 | 0.4128 | 0.4106 | – |

Coverage of the delay regimes at the preliminary stage reduced the RMSE of the frozen output layer by
10.8–15.3%. Under a regime not covered by the preliminary stage, the two-time-scale refinement
recovered 52–58% of the accuracy loss, sector RLS alone 31–43% and the per-source offset alone
21–33%.

## Formal check of the expressions

`supplementary/SRAttELMH_verification.lean` states and proves 45 theorems on expressions (1)–(36) of the article in
Lean 4 with Mathlib: causality of the available set, properties of RMSE and MAE, bounds of the availability
parameters, properties of the attention weights, the range of the sector index (16), the convergence of the offset
recursion (22), the scalar form of the RLS recursion (23)–(25), the cost and memory relations (28)–(31), the entropy
bounds (35) and the recovered share κ (36), including numerical checks of Table 9. See `supplementary/README.md`.

## Limitations

1. The measured concentrations are real, whereas the arrival delays come from the geometric delay model and from
   the network delay model with assumed link, traffic and UAV parameters; field message logs were not used.
2. One data set of twelve air-quality monitoring stations is used; the six paired runs are formed by two temporal
   splits and three delay realizations with a single test interval within each split.
3. GRU-D, BRITS, SAITS and GRIN are adapted reproductions of the main architectural principles of the published
   methods with compact hyperparameters and without a hyperparameter search; the official AGG core is trained with
   the shortened schedule of the adapter and evaluated in regimes ×4 and network-derived only. The values do not
   replicate the figures of the original publications.
4. Results were obtained on a CPU with Python 3.12.14, PyTorch 2.13.0, NumPy 2.5.3, pandas 3.0.6 and SciPy 1.18.1.
   On other hardware the RMSE may differ in the fourth decimal; the set of delayed components does not change. GPU
   execution was not tested.

## License

The code is licensed under the Apache License 2.0 (`LICENSE`, `NOTICE`). The folder `srattelm/agg/` includes model
classes transcribed from ChristopherLey/AsyncGraphGenerator, which is distributed under GNU GPL 3.0; this folder is
therefore licensed under GPL-3.0 (`srattelm/agg/LICENSE`, `THIRD_PARTY_NOTICE.md`). The data set is used under the
terms of the UCI Machine Learning Repository (https://doi.org/10.24432/C5RK5G).

## Citation

Version 1.0.0 is archived in Zenodo: https://doi.org/10.5281/zenodo.23579402 (all versions: https://doi.org/10.5281/zenodo.23579401).
Citation metadata are given in `CITATION.cff`.
