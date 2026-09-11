# Locked experiment protocol

## One-dimensional inviscid Burgers

- Equation: `u_t + (u^2/2)_x = 0` on the periodic interval `[0,1]`.
- Learning grid: 256 cells.
- Initial conditions: 2--6 periodic top hats with exact cell averages.
- Data: 2,048 training, 256 validation, and 256 test trajectories.
- Reference: periodic Lax--Hopf entropy solution converted to cell averages.

## Directional two-dimensional scalar Burgers transport

- Equation: `u_t + (beta_x u^2/2)_x + (beta_y u^2/2)_y = 0` on a periodic square.
- Direction: one random direction per trajectory with `|beta_x|+|beta_y|=1`.
- Learning grid: `128 x 128`.
- Initial conditions: 2--6 periodic rectangles represented by cell averages.
- Data: 512 training, 64 validation, and 256 test trajectories.
- Reference: WENO-5/Rusanov with SSP-RK3 at CFL 0.1 on `256 x 256`
  training grids and `512 x 512` validation/test grids, conservatively averaged.

## One-dimensional shallow water

- Equations: `h_t+q_x=0` and `q_t+(q^2/h+g h^2/2)_x=0`, with `g=1`.
- Domain: periodic, flat bottom, wet bed.
- Learning grid: 256 cells.
- Initial conditions: 2--6 piecewise-constant depth and velocity segments.
- Data: 2,048 training, 256 validation, and 256 test trajectories.
- Reference: WENO-5/HLL with SSP-RK3 at CFL 0.1 on 1,024-cell training
  grids and 2,048-cell validation/test grids, conservatively averaged.

## Training and rollout

All problems use base CFL 0.4 and 84 stored time intervals. Integer temporal
strides 1, 2, 3, and 4 provide training transitions at CFL 0.4, 0.8, 1.2, and
1.6. The rollout-loss curriculum increases from 2 to 20 learned updates.
Independent seeds 2027, 2028, and 2029 are trained.

Evaluation reaches a common final time at CFL 0.4, 0.8, 1.2, 1.6, 2.4, and
2.8 using 84, 42, 28, 21, 14, and 12 learned updates, respectively. CFL 2.4
and 2.8 are evaluation-only stress tests outside the training range.

The YAML files under `configs/production/` are authoritative for numerical
hyperparameters, data splits, random seeds, and checkpoint selection.

## Conservation reporting

New evaluations measure the change in the unweighted cell sum for each
conserved component. For shallow water, `mean_depth_conservation_error` and
`mean_discharge_conservation_error` are separate; `mean_conservation_error`
summarizes the maximum component error per completed trajectory. Multiplying
by the uniform cell volume gives the corresponding integral error.
The archived shallow-water summaries originally mixed h and q in this field.
Those entries are now blank (unavailable), not zero; component errors cannot
be recovered from a mixed sum. Accuracy, completion and timing results are
unchanged. Re-evaluation, not retraining, produces the corrected diagnostics.

The production `ENO-3` compatibility label denotes minimum-indicator candidate
selection. See `REPRODUCIBILITY.md` for its distinction from the standard ENO-3
used in the paper's separate smooth-solution convergence check.
