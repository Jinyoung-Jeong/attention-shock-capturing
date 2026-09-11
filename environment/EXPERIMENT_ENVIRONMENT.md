# Reported experiment environment

The reported production runs used:

- Python 3.10.20
- PyTorch 2.5.1
- CUDA 12.1
- NVIDIA RTX A6000 with 48 GB memory
- Intel Xeon Silver 4410Y, 24 physical cores and 48 logical CPUs
- 125 GiB system memory

The production pipeline used physical GPUs 0, 1, and 5 on the original server,
with one independent training seed per GPU. Those indices are not required by
the public code; any compatible CUDA device can be selected with `--device`.
CPU execution is supported for tests and the packaged smoke test.

## Public execution environment

`environment.yml` and `requirements-lock.txt` specify the public dependency
set, with PyTorch 2.14.0 in place of 2.5.1. This set was tested on Python 3.12
and CPU; the historical CUDA timing results were not remeasured here.
The original model definitions, training rules, datasets and weights are
unchanged. Use a separate environment; do not upgrade an active experiment
environment in place. For CPU-only installation, install PyTorch from its
official CPU wheel index before installing the remaining requirements.

The former dependency pins were numpy 2.2.6, pandas 2.3.3, PyYAML 6.0.3,
scipy 1.15.2, torch 2.5.1 and pytest 9.1.1. They are historical provenance,
not the recommended environment for opening downloaded tensor files.
