# Reproduction guide

Choose the path that matches your goal. Evaluating released models does not
require data generation or training.

Run commands from the repository root, after following
[installation](../README.md#installation). Commands below work in Bash and
PowerShell. GPU examples use `cuda:0`; replace it with `cpu` if needed.

## 1. Check the included samples

No extra downloads or training are needed:

```text
python scripts/smoke_test.py
python -m pytest
```

This checks sample predictions, finite outputs, conservation, and regression
tests. For reported numerical results without running experiments, start with
[results/INDEX.md](../results/INDEX.md).

## 2. Evaluate released models — no retraining

Download and extract the data for your chosen problem and
`reported_checkpoints.zip` using [Release assets](RELEASE_ASSETS.md).
For example, evaluate the primary 1D model:

```text
python scripts/evaluate.py --config configs/production/burgers_1d.yaml --seed 2027 --device cuda:0
```

Results are saved in
`artifacts/production/runs/burgers_1d/seed_2027/evaluation/`.
Repeat with seeds 2028 and 2029 for the three-seed comparison. Replace the
configuration with `burgers_2d.yaml` or `shallow_water_1d.yaml` for extensions.
Each evaluation includes classical baselines unless `--attention-only` is set.

### Attention and physical diagnostics

Use the same extracted data and checkpoints:

```text
python scripts/diagnose.py --config configs/production/burgers_1d.yaml --seed 2027 --device cuda:0
python scripts/diagnose_heads.py --config configs/production/burgers_1d.yaml --seed 2027 --device cuda:0
```

To inspect the original paper's fixed 1D checkpoint instead:

```text
python scripts/diagnose.py --config configs/production/burgers_1d.yaml --checkpoint checkpoints/samples/burgers_1d_frozen_paper.pt --output outputs/frozen_paper_diagnostics --device cuda:0
```

This uses the public diagnostic implementation and production test data; it is
not a rerun of the original paper's entire evaluation protocol.

### Reference and transfer checks

These require the corresponding datasets and, for model transfer, the trained
1D checkpoint. They do not retrain a model, but reference refinement can be costly.

```text
python scripts/audit_reference.py --config configs/production/burgers_2d.yaml --device cuda:0 --samples 8
python scripts/audit_reference.py --config configs/production/shallow_water_1d.yaml --device cuda:0 --samples 8
python scripts/evaluate_burgers_generalization.py --config configs/production/burgers_1d.yaml --seed 2027 --device cuda:0
```

## 3. Regenerate and retrain — optional full reproduction

Use a separate working copy to keep downloaded checkpoints and evaluation
outputs unchanged. This path is substantially more expensive than sample checks
or evaluation. Keep the supplied configurations unchanged for protocol reproduction.

### Generate data

Skip this step if you have extracted the released datasets.

```text
python scripts/generate_data.py --config configs/production/burgers_1d.yaml --device cuda:0
python scripts/generate_data.py --config configs/production/burgers_2d.yaml --device cuda:0
python scripts/generate_data.py --config configs/production/shallow_water_1d.yaml --device cuda:0
```

Datasets are written below `artifacts/production/data/`.

### Train and evaluate

```text
python scripts/train.py --config configs/production/burgers_1d.yaml --seed 2027 --device cuda:0
```

Repeat for seeds 2028 and 2029 and the two extension configurations. Then use
the evaluation and diagnostic commands in section 2 on the new checkpoints.

### Retrain mechanism controls

These compare CFL conditioning, dynamic attention, and a non-attention
ordered-stencil multilayer perceptron (MLP):

```text
python scripts/train.py --config configs/production/burgers_1d.yaml --seed 2027 --device cuda:0 --variant cfl_blind
python scripts/train.py --config configs/production/burgers_1d.yaml --seed 2027 --device cuda:0 --variant uniform
python scripts/train.py --config configs/production/burgers_1d_stencil_mlp.yaml --seed 2027 --device cuda:0
```

Evaluate each control with the matching configuration and variant:

```text
python scripts/evaluate.py --config configs/production/burgers_1d.yaml --seed 2027 --device cuda:0 --variant cfl_blind
python scripts/evaluate.py --config configs/production/burgers_1d.yaml --seed 2027 --device cuda:0 --variant uniform
python scripts/evaluate.py --config configs/production/burgers_1d_stencil_mlp.yaml --seed 2027 --device cuda:0
```

Repeat with the other two seeds for the reported three-seed protocol.
See [Experiment protocol](EXPERIMENT_PROTOCOL.md) for model selection and controls.

## Classical-solver check

This independent CPU check needs neither release downloads nor neural training:

```text
python scripts/validate_classical_convergence.py --check
```

It compares against
`results/appendix/classical_convergence_reported.csv`, including WENO-5's
finest-pair observed order of 5.107 (reported as 5.11).

The test uses pre-shock Burgers data `u0=1+0.5*sin(2*pi*x)` at `T=0.15`,
grids 32/64/128/256, exact initial cell averages, 128-point midpoint reference
quadrature, float64, and SSP-RK3 at a CFL bound of 0.05. It checks spatial
accuracy on these grids, not fifth-order time integration or convergence of
the learned model.

The ENO implementations serve different purposes:

- This appendix check uses standard recursive undivided-difference ENO-3.
- The extension production function `burgers_eno3_flux_1d` uses a
  minimum-Jiang–Shu-indicator ENO-type variant. Its historical `ENO-3` label
  remains for file compatibility.

Do not treat the appendix table as validation of the extension variant.
Archived result values are unchanged.
