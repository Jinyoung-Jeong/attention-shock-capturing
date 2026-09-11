# Frozen paper-checkpoint conversion

The frozen one-dimensional paper checkpoint is distributed with public module
names. Conversion changes state-dictionary keys only:

- `W_q`, `W_k`, and `W_v` map to `attention.query`, `attention.key`, and
  `attention.value`;
- `decoder` maps to `attention.decoder`;
- every parameter tensor is copied without numerical modification.

The original source SHA-256 is stored inside the converted checkpoint. On four
random 256-cell states spanning four `dt/dx` values, the maximum absolute flux
difference between the historical class and the public class was
`4.76837158203125e-07`, consistent with floating-point operation ordering.
