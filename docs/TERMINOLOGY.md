# Manuscript terminology

Public names follow the accompanying manuscript.

| Public term or identifier | Meaning |
|---|---|
| CFL-conditioned attention flux | Learned shared face-flux model used by the primary method |
| candidate cell | A cell whose features may contribute to a face flux |
| relative offset | Candidate-cell center measured from the queried face, in cell widths |
| transport reach | Travel distance during one update, normalized by cell width |
| transport alignment | Candidate offset shifted by the signed local transport reach |
| attention weights | State- and CFL-dependent candidate-selection weights |
| shared face flux | One numerical flux applied with opposite signs to adjacent cells |
| ordered-stencil MLP | Fixed left-to-right concatenation control without attention heads |

The Python package is named `transport_attention_fv`. Historical project names,
experiment-arm letters, personal directory labels, and manuscript revision names
are intentionally omitted. No compatibility alias changes the numerical model.
