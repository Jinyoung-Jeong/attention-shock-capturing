# Machine-readable results

`main/` contains the performance, conservation, attention-reach, and mechanism
summaries used in the main text. `appendix/` contains seed-level head analyses,
transfer tests, reference-grid audits, and the parameter-matched ordered-stencil
MLP comparison.

These files preserve the reported numerical outputs. Plotting and figure-layout
code are intentionally excluded. Method labels were normalized to the
manuscript term `CFL-conditioned attention`. Accuracy, completion and timing
columns are unchanged. Invalid mixed-component conservation entries for shallow
water in the two performance summaries are blanked, not replaced by zeros or
estimated component errors. New evaluations report h and q conservation separately.

`classical_convergence_reported.csv` contains the ENO-3/WENO-5 subset of the
paper's archived smooth-solution verification. It is distinct from the learned
model's `smooth_convergence_*` transfer tests. See `docs/REPRODUCIBILITY.md`
for the difference between standard ENO-3 and the extension's historical
minimum-indicator `ENO-3` label.
