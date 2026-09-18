# Dynamic Geographic Ratemaking under Tail Risk

### A Distributional Reinforcement Learning Framework for Dynamic Insurance Zoning

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Distributional_RL-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-Supported-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Reproducible Research](https://img.shields.io/badge/Reproducible-Research-2EA44F)](#reproducibility)
[![Zenodo](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.22826154-1682D4?logo=zenodo&logoColor=white)](https://doi.org/10.5281/zenodo.22826154)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22826154.svg)](https://doi.org/10.5281/zenodo.22826154)

---

## Overview

This repository contains the computational and reproducibility materials for the study:

> **Dynamic Geographic Ratemaking under Tail Risk: A Distributional Reinforcement Learning Approach**

The project studies dynamic geographic ratemaking as a finite-horizon sequential actuarial decision problem. Spatial risk information is converted into admissible geographic classifications and territorial pricing decisions, while policies are evaluated through their induced cumulative actuarial return distributions.

The computational framework distinguishes three objects that should not be conflated:

1. **statistical risk representation**, describing predecision spatial insurance risk;
2. **geographic and pricing decisions**, determining admissible territorial classifications and relativities;
3. **policy-level return distributions**, used to evaluate complete sequential policies.

The implementation combines geographic insurance zoning, tail-sensitive policy evaluation, and distributional reinforcement learning in a reviewer-auditable computational pipeline.

---

## Research Design

The pipeline implements a commune-level French calibrated actuarial environment and supports four empirical policy architectures:

| Policy | Return representation | Local decision rule | Policy-level selection |
|---|---|---|---|
| **Static Zoning** | Fixed geographic benchmark | Adaptive pricing only | Benchmark |
| **Mean DQN** | Scalar expected return | Mean based | Mean based |
| **QR Mean** | Quantile return distribution | Mean based | Mean based |
| **Tail-QR Zoning** | Quantile return distribution | Tail sensitive | Tail sensitive |

This design separates **distributional representation** from **decision and selection rules**. In particular, moving from Mean DQN to QR Mean changes the representation of policy-conditional returns while retaining mean-based ranking and selection. Moving from QR Mean to Tail-QR Zoning retains the quantile representation while introducing tail-sensitive ranking and policy selection.

The distributional recursion is used for policy-conditional return-law evaluation. It is **not** interpreted as a Bellman decomposition of the policy-level precommitment tail criterion.

---

## Actuarial Criterion

For a policy $\pi$, the computational analysis evaluates cumulative actuarial return over the finite horizon. Policy comparison combines expected cumulative return with the Tail Value at Risk of adverse cumulative return.

Conceptually,

$$
J_{\alpha,\lambda}^{\pi} = \mathbb{E}[G_0^\pi] - \lambda \, \mathrm{TVaR}_{\alpha}(-G_0^\pi)
$$

The baseline implementation uses:

* tail probability level: $\alpha = 0.95$;
* tail loading: $\lambda = 0.75$;
* discount factor: $\gamma = 0.97$;
* horizon: 5 years.

The tail functional is applied to the distribution of the **complete cumulative return**. Local action scores are used for candidate-policy construction and are not recursively aggregated as policy-level values of $J_{\alpha,\lambda}$.
The baseline implementation uses

- tail probability level: $\alpha = 0.95$;
- tail loading: $\lambda = 0.75$;
- discount factor: $\gamma = 0.97$;
- horizon: $5$ years.

The tail functional is applied to the distribution of the **complete cumulative return**. Local action scores are used for candidate-policy construction and are not recursively aggregated as policy-level values of $J_{\alpha,\lambda}$.

---

## Geographic Decision Space

The geographic support consists of metropolitan French communes linked through an adopted adjacency graph.

The finite computational action approximation includes explicit geographic edits:

- **Keep** — retain the current partition;
- **Split** — divide an admissible geographic zone;
- **Merge** — combine admissible neighboring zones;
- **Shift** — move admissible boundary communes;
- **Pricing adjustments** — modify territorial relativities around current zonal indications.

Geographic feasibility incorporates connectedness, a maximum number of zones, and a minimum exposure-proxy threshold.

The finite action set is a computational approximation to the admissible control problem and should not be interpreted as exhaustive optimization over every possible geographic partition.

---

## Data and Scientific Scope

The baseline experiment combines public French geographic, demographic, and hazard information.

The pipeline uses, among other inputs:

- commune geometries and administrative identifiers;
- commune population as an **exposure proxy**;
- GASPAR / CatNat administrative hazard histories;
- BRGM / Géorisques information related to shrink-swell clay risk when available;
- optional climate and auxiliary public-data modules.

### Important scientific boundary

> **This repository does not treat public hazard records as observed insurance claims.**

The baseline application is a **calibrated actuarial loss environment**, not an observed-claims estimation exercise.

Population is used as an exposure proxy and is not interpreted as observed insured exposure. CatNat administrative recognition is used as hazard information and is not an insurer claims database. Monetary loss levels, loss dispersion, tail thickness, and spatial dependence are calibrated components of the experimental environment.

Accordingly, the repository should not be used as evidence of observed French insurer pricing behavior, causal climate effects on insurance losses, or empirically identified French territorial ratemaking practices.

---

## Temporal Design

The baseline analysis uses a strictly separated temporal protocol:

| Block | Years | Role |
|---|---:|---|
| Training | 2000–2015 | Learning and candidate construction |
| Selection | 2016–2020 | Independent candidate-policy selection |
| Test | 2021–2025 | Final out-of-sample evaluation |

Selected policies are frozen before final test and stress evaluation.

The baseline experiment uses **10 independent random seeds**:

```text
101, 202, 303, 404, 505, 606, 707, 808, 909, 1010
```

---

## Repository Structure

A standard execution creates the following structure:

```text
.
├── ime_tail_qr_pipeline_v13_final.py
├── Data_IME/
│   ├── raw/
│   ├── processed/
│   ├── cache/
│   └── metadata/
│
└── Results_IME/
    ├── figures/
    ├── tables/
    ├── csv_figures/
    ├── csv_tables/
    ├── models/
    ├── logs/
    ├── manifests/
    └── ime_parameters.json
```

The exact inventory depends on the execution mode and on the availability of optional public-data sources.

---

## Requirements

### Core software

- Python **3.10+**
- NumPy
- pandas
- SciPy
- scikit-learn
- requests
- Matplotlib
- GeoPandas
- Shapely
- PyArrow
- NetworkX
- PyTorch

`tqdm` is optional.

Install the Python dependencies with:

```bash
pip install numpy pandas scipy scikit-learn requests matplotlib \
    geopandas shapely pyarrow networkx torch tqdm
```

For GPU execution, install a PyTorch build compatible with the CUDA configuration of your system. See the official [PyTorch installation instructions](https://pytorch.org/get-started/locally/).

---


### 1. Validate the software installation

Before running the complete experiment, execute the lightweight offline smoke test:

```bash
python ime_tail_qr_pipeline_v13_final.py --smoke-test
```

The smoke test uses a small synthetic spatial environment and is intended to validate the software pipeline. Its outputs are **not manuscript evidence**.

### 2. Download public inputs only

```bash
python ime_tail_qr_pipeline_v13_final.py --download-data
```

### 3. Run the complete pipeline with CUDA

```bash
python ime_tail_qr_pipeline_v13_final.py --run-all --use-cuda
```

### 4. Run on CPU

```bash
python ime_tail_qr_pipeline_v13_final.py --run-all --cpu
```

A complete CPU execution can be substantially slower than CUDA execution.

---

## Reproducing the Analysis

### Full reproduction from public inputs

For a fresh end-to-end run:

```bash
python ime_tail_qr_pipeline_v13_final.py ^
    --run-all ^
    --use-cuda ^
    --force-download ^
    --force-compute
```

This workflow performs the complete sequence:

```text
public-data acquisition
        ↓
spatial preprocessing
        ↓
predecision risk representation
        ↓
admissible geographic action construction
        ↓
policy training
        ↓
independent candidate selection
        ↓
frozen-policy test evaluation
        ↓
stress and sensitivity analysis
        ↓
figures + tables + CSV outputs + manifests
```

### Rebuild outputs without retraining

When the processed results are already available:

```bash
python ime_tail_qr_pipeline_v13_final.py --rebuild-outputs
```

This mode reconstructs supported manuscript figures from saved processed outputs without repeating RL training.

### Post-training V13 analysis

When the baseline trained models and required processed files already exist:

```bash
python ime_tail_qr_pipeline_v13_final.py ^
    --v13-analysis-only ^
    --use-cuda
```

The V13 preference analysis re-ranks the **frozen candidate family**. It does not retrain the reinforcement-learning models separately for every $(\alpha,\lambda)$ configuration.

---

## Configuration

The default configuration is defined directly in the `Config` dataclass.

A JSON file can override configuration parameters:

```bash
python ime_tail_qr_pipeline_v13_final.py ^
    --run-all ^
    --config ime_parameters.json ^
    --use-cuda
```

Custom data and result directories can also be supplied:

```bash
python ime_tail_qr_pipeline_v13_final.py ^
    --run-all ^
    --data-dir Data_IME ^
    --results-dir Results_IME ^
    --use-cuda
```

Unknown JSON configuration keys are rejected rather than silently ignored.

---

## Reproducibility

The pipeline was designed to make the computational experiment auditable.

### Randomness

Random seeds are propagated to:

- Python's `random`;
- NumPy;
- PyTorch;
- CUDA generators when CUDA is available.

### Run metadata

Each production run writes a machine-readable manifest containing, among other information:

- pipeline version;
- creation timestamp;
- Python version;
- PyTorch version;
- CUDA availability;
- CUDA device name;
- random seeds;
- generated output inventory;
- scientific-status metadata.

The configuration used for the run is separately exported to:

```text
Results_IME/ime_parameters.json
```

The principal run manifest is stored under:

```text
Results_IME/manifests/run_manifest.json
```

### Data provenance

Downloaded public resources are recorded in a data manifest containing source information, local paths, scientific roles, status information, and SHA-256 hashes when applicable.

### Determinism

The pipeline controls random seeds, but exact bitwise identity across different GPUs, CUDA versions, PyTorch releases, operating systems, and hardware configurations is not guaranteed unless deterministic PyTorch execution is explicitly enabled and supported by all operations.

Reproducibility should therefore be assessed primarily through the documented experimental design, configuration, seeds, saved outputs, manifests, and reported numerical tolerance rather than by assuming cross-platform bitwise identity.

---

## CUDA Support

GPU acceleration is implemented through PyTorch.

Request CUDA execution with:

```bash
python ime_tail_qr_pipeline_v13_final.py --run-all --use-cuda
```

When CUDA is available, the pipeline records the detected GPU and PyTorch version in the execution log and run manifest.

To force CPU execution:

```bash
python ime_tail_qr_pipeline_v13_final.py --run-all --cpu
```

CUDA is recommended for the complete multi-seed reinforcement-learning experiment but is not required for the software smoke test.

---

## Outputs

The pipeline separates publication artifacts from intermediate computational objects.

### Figures

```text
Results_IME/figures/
```

### Tables

```text
Results_IME/tables/
```

### Figure-level numerical data

```text
Results_IME/csv_figures/
```

### Table-level numerical data

```text
Results_IME/csv_tables/
```

### Processed analysis objects

```text
Data_IME/processed/
```

### Trained models

```text
Results_IME/models/
```

### Execution logs

```text
Results_IME/logs/
```

### Reproducibility manifests

```text
Results_IME/manifests/
```

CSV outputs are retained wherever possible so that reported numerical results and supported figures can be inspected independently of the complete RL training process.

---

## Sensitivity and Stress Analysis

The baseline implementation includes stress and post-training preference analyses.

The default preference grid is

```text
alpha  ∈ {0.90, 0.95, 0.975}
lambda ∈ {0.25, 0.50, 0.75, 1.00, 1.50}
```

and the tail-multiplier grid is

```text
{1.00, 1.45, 1.80}.
```

This produces **45 evaluated preference/stress configurations**.

These analyses operate on the frozen post-training candidate family. They must therefore be interpreted as finite-grid sensitivity analyses, not as evidence of global optimization or as full retraining under every preference configuration.

---

## Interpretation of the Computational Results

Several distinctions are essential when using the repository:

- a larger number of geographic zones does not by itself imply superior actuarial performance;
- an increase in the number of zones is not necessarily a set-theoretic refinement of the previous partition;
- distributional return representation and tail-sensitive policy selection are distinct computational components;
- recursive distributional evaluation does not imply recursive optimization of the precommitment tail criterion;
- the finite candidate-policy family is not the unrestricted admissible policy space;
- stress and sensitivity results apply to the evaluated experimental configurations.

These boundaries are part of the scientific design rather than implementation caveats.

---

## Data Availability

The public-data acquisition layer is implemented directly in the pipeline. Source metadata and downloaded-file provenance are recorded during execution.

The archived research materials associated with this project are available on Zenodo:

**DOI:** [10.5281/zenodo.22826154](https://doi.org/10.5281/zenodo.22826154)

> For exact reproducibility, cite the Zenodo version corresponding to the files used in your analysis.

Zenodo distinguishes version-specific DOIs from the concept DOI representing all versions. When reproducing a particular published computational artifact, the version-specific DOI is preferable.

---


## Reproducibility Checklist

- [x] Public-data acquisition documented
- [x] Fixed temporal train / selection / test separation
- [x] Independent random seeds
- [x] CPU execution supported
- [x] CUDA acceleration supported
- [x] Synthetic smoke test available
- [x] Configuration exported
- [x] Execution logs retained
- [x] Machine-readable run manifest
- [x] Data provenance recorded
- [x] SHA-256 hashing of downloaded files where applicable
- [x] CSV data exported for manuscript outputs
- [x] Selected policies frozen before final test evaluation
- [x] Stress and preference sensitivity distinguished from baseline training
- [x] Zenodo archival DOI

---

## Scientific Disclaimer

This repository implements a calibrated actuarial experiment for methodological research.

Unless an explicitly schema-audited claims extension is supplied, generated insured losses are simulated within the calibrated actuarial environment. Public hazard records, population data, and geographic information must not be interpreted as observed insurance claims, insured exposure, insurer tariffs, or identified insurer behavior.

The computational comparisons are controlled algorithmic contrasts within the specified experimental environment. They do not constitute causal estimates of the effects of geographic zoning, reinforcement learning, or climate hazards on observed insurer outcomes.

---


## Archival Record

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22826154.svg)](https://doi.org/10.5281/zenodo.22826154)

**Zenodo DOI:** [10.5281/zenodo.22826154](https://doi.org/10.5281/zenodo.22826154)

The Zenodo archive provides a persistent scholarly record of the reproducibility materials associated with this project.

---

## Contact

For questions concerning the methodology, computational implementation, or reproducibility package, please use the GitHub issue tracker or contact the corresponding author.

---

<p align="center">
  <b>Actuarial Science · Geographic Ratemaking · Tail Risk · Distributional Reinforcement Learning</b>
</p>
