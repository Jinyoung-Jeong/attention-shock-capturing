# Checkpoints

`samples/` contains four inference checkpoints:

| File | Purpose |
|---|---|
| `burgers_1d_seed_2027.pt` | Primary 1D Burgers model, seed 2027 |
| `burgers_2d_seed_2027.pt` | Primary directional 2D Burgers model, seed 2027 |
| `shallow_water_1d_seed_2027.pt` | Primary shallow-water model, seed 2027 |
| `burgers_1d_frozen_paper.pt` | Original paper's fixed 1D model, converted to the public format |

The smoke test uses the first three. The frozen paper checkpoint is separate
from the three-seed reproduction; do not interchange their reported results.
Its diagnostic command is in [the reproduction guide](../docs/REPRODUCIBILITY.md).

Checkpoints include weights and public metadata, but no optimizer state or
machine-specific paths. They support inference; train from scratch to reproduce
training rather than resume the original optimizer state.

The separate release archive contains all selected three-seed primary weights
and the mechanism-control weights reported in the paper.
