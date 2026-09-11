# Attention Is All You Need (to Avoid Spurious Oscillations)

Research code, trained models, datasets, and numerical results for the accompanying
paper. The method learns a **CFL-conditioned attention flux** for conservative,
large-time-step shock capturing.

**Start here:** [Quick verification](#quick-verification) ·
[Reported results](results/INDEX.md) ·
[Data and trained models](https://github.com/Jinyoung-Jeong/attention-shock-capturing/releases/tag/v0.1.0) ·
[Full reproduction guide](docs/REPRODUCIBILITY.md)

The repository is named `attention-shock-capturing`; the Python package remains
`transport_attention_fv`.

The method keeps the finite-volume flux-difference update fixed and learns one
shared numerical flux at each cell face. Attention acts as a state-dependent
information stencil: the local state and CFL-based transport scale determine
which upstream candidate cells contribute to the face flux.

## Scope

The released experiments cover:

1. one-dimensional inviscid Burgers transport, the central mechanism test;
2. directional two-dimensional scalar Burgers transport on a periodic Cartesian grid;
3. the one-dimensional shallow-water system on a periodic, flat, wet bed.

The primary models use the same four-head, CFL-conditioned design principle.
Each face flux is shared by its adjacent cells, so conservation follows from the
finite-volume assembly rather than from a learned penalty.

## Repository map

```text
src/transport_attention_fv/  Model, equations, numerics, data, training, evaluation
configs/                     Exact production and smoke configurations
scripts/                     Data generation, training, evaluation, and diagnostics
tests/                       Conservation and protocol regression tests
data/samples/                One small trajectory per problem
checkpoints/samples/         Three primary samples + frozen paper 1D checkpoint
results/                     Machine-readable main-text and appendix results
docs/                        Terminology, protocol, and reproducibility notes
environment/                 Reported software and hardware environment
```

Manuscript files, tracked revisions, internal review material, plotting code,
and figure-layout scripts are intentionally not part of this repository.

## Installation

Clone the repository and enter it:

```bash
git clone https://github.com/Jinyoung-Jeong/attention-shock-capturing.git
cd attention-shock-capturing
```

Run all commands from this folder, which contains `pyproject.toml`.
Use a separate environment; the public loader requires PyTorch
2.14 or later for restricted tensor loading.

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, replace the activation line with:

```powershell
.\.venv\Scripts\Activate.ps1
```

Then install the package and tests:

```text
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Alternatively, use the supplied conda environment:

```bash
conda env create -f environment/environment.yml
conda activate transport-attention-fv
python -m pip install -e ".[dev]"
```

For GPU runs, use a compatible PyTorch CUDA build. The reported historical
environment and the tested public environment are listed in
[`environment/EXPERIMENT_ENVIRONMENT.md`](environment/EXPERIMENT_ENVIRONMENT.md).

## Choose your starting point

| Goal | What you need | Next step |
|---|---|---|
| Check installation and sample predictions | Included samples; CPU is enough | Quick verification below |
| Evaluate released models | Data and checkpoint release archives; no training | Reproduce an evaluation below |
| Regenerate data and retrain | Production configurations and compute time | [Full reproduction guide](docs/REPRODUCIBILITY.md) |

## Quick verification

The smoke test loads the three sample checkpoints, performs one conservative
update on the packaged trajectories, and checks finite outputs and conservation:

```bash
python scripts/smoke_test.py
python -m pytest
```

The smoke test runs on CPU and does not retrain a model.
Successful checks report finite predictions, conservation checks, and passing
tests. They verify installation, not full production accuracy.

## Reproduce an evaluation

After extracting the full datasets and checkpoints described in
[`docs/RELEASE_ASSETS.md`](docs/RELEASE_ASSETS.md), evaluate one trained seed:

```text
python scripts/evaluate.py --config configs/production/burgers_1d.yaml --seed 2027 --device cuda:0
```

Results are written under
`artifacts/production/runs/burgers_1d/seed_2027/evaluation/`.
This command evaluates the learned model and classical baselines; it does not
train. Use `--device cpu` if needed, but full evaluation will be slower.

The same interface applies to `burgers_2d.yaml` and `shallow_water_1d.yaml`.
For other seeds, diagnostics, or retraining, see
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).
Wall-clock timings depend on the hardware and runtime used for evaluation.

## Reported protocol

- Training seeds: 2027, 2028, and 2029.
- Training CFL numbers: 0.4, 0.8, 1.2, and 1.6.
- Training loss: multi-step rollouts with a curriculum from 2 to 20 updates.
- Evaluation CFL numbers: 0.4, 0.8, 1.2, 1.6, 2.4, and 2.8.
- Common-final-time rollout lengths: 84, 42, 28, 21, 14, and 12 updates.

CFL 2.4 and 2.8 are evaluation-only stress tests outside the training range.
The complete problem and reference-solution definitions are recorded in
[`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md) and the versioned
YAML configurations.

## Public terminology

The public API follows the manuscript. In particular, it uses
`CFL-conditioned attention flux`, `transport reach`, `transport alignment`,
`candidate cells`, `attention weights`, and `shared face flux`. Historical
project labels are not used in public commands. See
[`docs/TERMINOLOGY.md`](docs/TERMINOLOGY.md).

## Data and checkpoints

Small samples are committed for verification. Full datasets and selected
checkpoints are kept outside normal Git history as release assets, with SHA-256
checksums. The datasets are deterministic given the supplied configurations and
split seeds.

The sample weights include three primary models (seed 2027) and a separate
frozen original-paper 1D model. See [checkpoints/README.md](checkpoints/README.md)
to choose the appropriate checkpoint.

## Citation and license

Please cite **“Attention Is All You Need (to Avoid Spurious Oscillations)”** by
Jinyoung Jeong, Joseph B. Choi, Xinlun Cheng, H. S. Udaykumar, Sanghun Choi,
and Stephen S. Baek when using this work. Citation metadata are provided in
[`CITATION.cff`](CITATION.cff); the paper link will be added when the preprint
becomes available.

The source code is released under the [MIT License](LICENSE), retaining the
original code's copyright notice. This software license is separate from the
publication license of the paper.
