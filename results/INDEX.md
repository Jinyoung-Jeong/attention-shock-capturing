# Result-file index

## Main-text evidence

| File | Reported quantity |
|---|---|
| `main/performance_three_seed.csv` | Accuracy, completion, conservation, work, and wall-clock summaries |
| `main/performance_by_seed.csv` | Seed-level values underlying the aggregate |
| `main/attention_reach_aggregate.csv` | CFL-resolved physical-reach/attention-reach relation |
| `main/attention_region_stats_aggregate.csv` | Shock-versus-smooth attention statistics |
| `main/fixed_checkpoint_interventions.csv` | Inference-time mechanism interventions |
| `main/retrained_mechanism_controls.csv` | CFL-blind and uniform-attention retraining controls |

## Appendix evidence

| File group | Reported quantity |
|---|---|
| `head_*` | Functional head alignment and single-head interventions |
| `ordered_stencil_mlp_*` | Parameter-matched non-attention control |
| `generalization_*` | Grid, initial-condition-family, and rollout-horizon transfer |
| `smooth_convergence_*` | Smooth-solution resolution study |
| `classical_convergence_reported.csv` | Separate ENO-3/WENO-5 pre-shock reconstruction check |
| `paper_*` | Frozen 1D checkpoint consistency, shock-speed, rarefaction, and attention diagnostics |
| `reference_audit_*` | Twofold reference-grid self-convergence audits |

Errors named `median_l2_successful` are conditioned on completed rollouts and
must be interpreted together with `success_rate`. CFL 2.4 and 2.8 are
evaluation-only stress tests rather than trained operating points.
