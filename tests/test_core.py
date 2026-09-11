from pathlib import Path

import torch
import yaml

from transport_attention_fv.initial_conditions import (
    periodic_rectangles_2d,
    periodic_top_hats_1d,
    shallow_water_segments_1d,
)
from transport_attention_fv.models import (
    BurgersFaceAttention1D,
    BurgersFaceAttention2D,
    BurgersStencilMLP1D,
    ShallowWaterFaceAttention1D,
)
from transport_attention_fv.evaluation import _summarize
from transport_attention_fv.diagnostics import (
    attention_region_statistics,
    burgers_rarefaction_cell_averages,
    subcell_shock_position_1_to_0,
)
from transport_attention_fv.numerics import conservative_update_1d, conservative_update_2d


def test_periodic_initial_conditions_are_finite_and_wet():
    generator = torch.Generator().manual_seed(7)
    assert torch.isfinite(periodic_top_hats_1d(64, generator)).all()
    assert torch.isfinite(periodic_rectangles_2d(16, 16, generator)).all()
    shallow = shallow_water_segments_1d(64, generator)
    assert torch.isfinite(shallow).all()
    assert shallow[:, 0].min() >= 0.5


def test_1d_burgers_model_is_conservative():
    torch.manual_seed(1)
    state = torch.randn(3, 32)
    model = BurgersFaceAttention1D(d_model=16, decoder_width=16)
    scale = torch.tensor([0.1, 0.2, 0.3])
    flux = model(state, scale)
    updated = conservative_update_1d(state, flux, scale)
    torch.testing.assert_close(updated.sum(1), state.sum(1), atol=2e-6, rtol=0)


def test_2d_shared_face_model_is_conservative():
    torch.manual_seed(2)
    state = torch.randn(2, 8, 8)
    model = BurgersFaceAttention2D(d_model=16, decoder_width=16)
    scale = torch.tensor([0.1, 0.2])
    fx, fy = model(state, scale, scale, torch.tensor([1.0, 0.5]), torch.tensor([0.0, 0.5]))
    updated = conservative_update_2d(state, fx, fy, scale, scale)
    torch.testing.assert_close(updated.sum((1, 2)), state.sum((1, 2)), atol=3e-6, rtol=0)


def test_shallow_water_model_shape_and_conservation():
    torch.manual_seed(3)
    state = torch.stack((torch.ones(2, 32), 0.1 * torch.randn(2, 32)), -1)
    model = ShallowWaterFaceAttention1D(d_model=16, decoder_width=16)
    scale = torch.tensor([0.1, 0.2])
    flux = model(state, scale)
    assert flux.shape == state.shape
    updated = conservative_update_1d(state, flux, scale)
    torch.testing.assert_close(updated.sum(1), state.sum(1), atol=2e-6, rtol=0)


def test_interventions_do_not_change_shapes():
    state = torch.randn(2, 32)
    model = BurgersFaceAttention1D(d_model=16, decoder_width=16)
    for intervention in ("learned", "uniform", "cfl_blind", "frozen_center"):
        flux, weights = model(state, 0.1, intervention=intervention, return_attention=True)
        assert flux.shape == state.shape
        assert weights.shape == (2, 32, 6, 4)
        torch.testing.assert_close(weights.sum(-2), torch.ones(2, 32, 4), atol=1e-6, rtol=0)


def test_single_head_interventions_are_local_and_normalized():
    torch.manual_seed(11)
    state = torch.randn(2, 32)
    model = BurgersFaceAttention1D(d_model=16, decoder_width=16)
    learned_flux, learned_weights = model(state, 0.4, return_attention=True)
    uniform_flux, uniform_weights = model(
        state, 0.4, return_attention=True,
        head_index=2, head_intervention="uniform_attention",
    )
    zero_flux, zero_weights = model(
        state, 0.4, return_attention=True,
        head_index=2, head_intervention="zero_context",
    )
    torch.testing.assert_close(uniform_weights[..., 2], torch.full_like(uniform_weights[..., 2], 1 / 6))
    torch.testing.assert_close(uniform_weights[..., :2], learned_weights[..., :2])
    torch.testing.assert_close(uniform_weights[..., 3], learned_weights[..., 3])
    torch.testing.assert_close(zero_weights, learned_weights)
    assert not torch.allclose(uniform_flux, learned_flux)
    assert not torch.allclose(zero_flux, learned_flux)


def test_ordered_stencil_mlp_is_parameter_matched_and_conservative():
    attention = BurgersFaceAttention1D(d_model=32, heads=4, decoder_width=64)
    control = BurgersStencilMLP1D(hidden_width=60)
    attention_parameters = sum(parameter.numel() for parameter in attention.parameters())
    control_parameters = sum(parameter.numel() for parameter in control.parameters())
    assert abs(control_parameters - attention_parameters) / attention_parameters < 0.02

    torch.manual_seed(12)
    state = torch.randn(3, 32)
    scale = torch.tensor([0.1, 0.2, 0.3])
    flux = control(state, scale)
    updated = conservative_update_1d(state, flux, scale)
    torch.testing.assert_close(updated.sum(1), state.sum(1), atol=2e-6, rtol=0)


def test_consistent_flux_matches_physical_flux_on_constant_states():
    for query_skip in (False, True):
        burgers = BurgersFaceAttention1D(
            d_model=16, decoder_width=16, query_skip=query_skip, consistent_flux=True
        )
        state = torch.tensor([[0.7] * 32, [-1.2] * 32])
        flux = burgers(state, torch.tensor([0.2, 0.3]))
        torch.testing.assert_close(flux, 0.5 * state.square(), atol=1e-7, rtol=0)

        burgers_2d = BurgersFaceAttention2D(
            d_model=16, decoder_width=16, query_skip=query_skip, consistent_flux=True
        )
        field = torch.full((1, 8, 8), 0.8)
        fx, fy = burgers_2d(field, 0.1, 0.1, 0.75, -0.25)
        torch.testing.assert_close(fx, torch.full_like(fx, 0.75 * 0.5 * 0.8**2), atol=1e-7, rtol=0)
        torch.testing.assert_close(fy, torch.full_like(fy, -0.25 * 0.5 * 0.8**2), atol=1e-7, rtol=0)

        shallow = ShallowWaterFaceAttention1D(
            d_model=16, decoder_width=16, query_skip=query_skip, consistent_flux=True
        )
        constant = torch.tensor([1.1, 0.22]).view(1, 1, 2).expand(1, 32, 2)
        predicted = shallow(constant, 0.1)
        expected = shallow.equation.flux(constant)
        torch.testing.assert_close(predicted, expected, atol=1e-7, rtol=0)


def test_consistent_flux_has_finite_gradients_on_piecewise_constant_states():
    for query_skip in (False, True):
        state = torch.zeros(2, 32)
        state[:, 8:20] = torch.tensor([[1.0], [-0.8]])
        model = BurgersFaceAttention1D(
            d_model=16, decoder_width=16, query_skip=query_skip, consistent_flux=True
        )
        loss = model(state, torch.tensor([0.4, 1.6])).square().mean()
        loss.backward()
        assert all(parameter.grad is None or torch.isfinite(parameter.grad).all()
                   for parameter in model.parameters())


def test_every_config_reaches_one_common_terminal_frame():
    root = Path(__file__).resolve().parents[1]
    for path in sorted((root / "configs").glob("*/*.yaml")):
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        base_cfl = float(config["data"]["base_cfl"])
        frames = int(config["data"]["base_frames"])
        evaluation_strides = []
        for cfl in config["evaluation"]["cfl_values"]:
            stride = int(round(float(cfl) / base_cfl))
            assert abs(stride * base_cfl - float(cfl)) < 1.0e-8, path
            assert frames % stride == 0, (path, frames, stride)
            evaluation_strides.append(stride)
        for stride in config["training"]["strides"]:
            assert frames >= int(config["training"]["rollout_max"]) * int(stride), (path, stride)


def test_primary_configs_preserve_the_paper_flux_head_design():
    root = Path(__file__).resolve().parents[1]
    for mode in ("smoke", "production"):
        for problem in ("burgers_1d", "burgers_2d", "shallow_water_1d"):
            config = yaml.safe_load((root / "configs" / mode / f"{problem}.yaml").read_text(encoding="utf-8"))
            assert config["model"]["query_skip"] is True
            assert config["model"]["consistent_flux"] is False

    direct = BurgersFaceAttention1D(
        d_model=16, decoder_width=16, query_skip=True, consistent_flux=False
    )
    constrained = BurgersFaceAttention1D(
        d_model=16, decoder_width=16, query_skip=True, consistent_flux=True
    )
    assert direct.attention.decoder[-1].weight.abs().sum() > 0
    assert constrained.attention.decoder[-1].weight.abs().sum() == 0


def test_evaluation_summary_separates_success_rate_from_conditional_error():
    common = {
        "method": "test", "cfl": 1.6, "steps": 2, "flux_evaluations": 2,
        "timing_samples": 2, "conservation": 0.0, "wall_clock_seconds": 0.1,
        "realized_cfl": 1.6, "nonpositive_depth": 0,
        "depth_l2": float("nan"), "discharge_l2": float("nan"),
    }
    records = [
        {**common, "failed": 0, "l1": 1.0, "l2": 2.0, "linf": 3.0},
        {**common, "failed": 0, "l1": 3.0, "l2": 4.0, "linf": 5.0},
        {**common, "failed": 1, "l1": float("nan"), "l2": float("nan"), "linf": float("nan")},
    ]
    row = _summarize(records)[0]
    assert row["successes"] == 2
    assert row["failures"] == 1
    assert row["success_rate"] == 2 / 3
    assert row["median_l2_successful"] == 3.0
    assert "median_l2" not in row


def test_shared_burgers_diagnostics_are_subcell_and_physically_normalized():
    nx = 32
    state = torch.ones(nx)
    state[17:] = 0.0
    state[16] = 0.75
    state[17] = 0.25
    measured = subcell_shock_position_1_to_0(state, expected=17 / nx)
    assert abs(measured - 17 / nx) < 1.0e-7

    rarefaction = burgers_rarefaction_cell_averages(
        nx, time=0.125, device=torch.device("cpu"), dtype=torch.float64
    )
    assert abs(float(rarefaction.sum())) < 1.0e-12
    assert float(rarefaction.min()) >= -1.0
    assert float(rarefaction.max()) <= 1.0

    batch = torch.zeros(1, nx)
    batch[:, 8:24] = 1.0
    weights = torch.full((1, nx, 6, 4), 1.0 / 6.0)
    offsets = torch.tensor([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
    rows = attention_region_statistics(batch, weights, offsets, cfl=1.6)
    mean_rows = {row["region"]: row for row in rows if row["head"] == "mean"}
    assert set(mean_rows) == {"shock", "smooth"}
    assert abs(mean_rows["shock"]["left_right_asymmetry"]) < 1.0e-7
    assert abs(mean_rows["smooth"]["center_mass"] - 1.0 / 3.0) < 1.0e-7
