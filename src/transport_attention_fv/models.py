from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn

from .equations import ShallowWater1D
from .numerics import burgers_godunov_flux, burgers_rusanov_flux, shallow_water_hll_flux


Intervention = Literal["learned", "uniform", "cfl_blind", "frozen_center"]
HeadIntervention = Literal["none", "uniform_attention", "zero_context"]


class MultiHeadFaceAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        value_dim: int,
        output_dim: int,
        d_model: int = 32,
        heads: int = 4,
        decoder_width: int = 64,
        use_query_skip: bool = True,
        zero_output_init: bool = False,
    ) -> None:
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        self.heads = heads
        self.head_dim = d_model // heads
        self.use_query_skip = use_query_skip
        self.query = nn.Linear(query_dim, d_model)
        self.key = nn.Linear(key_dim, d_model)
        self.value = nn.Linear(value_dim, d_model)
        self.decoder = nn.Sequential(
            nn.Linear(d_model + query_dim if use_query_skip else d_model, decoder_width),
            nn.GELU(),
            nn.Linear(decoder_width, decoder_width),
            nn.GELU(),
            nn.Linear(decoder_width, output_dim),
        )
        if zero_output_init:
            nn.init.zeros_(self.decoder[-1].weight)
            nn.init.zeros_(self.decoder[-1].bias)

    def forward(
        self,
        query_features: torch.Tensor,
        key_features: torch.Tensor,
        value_features: torch.Tensor,
        intervention: Intervention = "learned",
        frozen_index: int | None = None,
        head_index: int | None = None,
        head_intervention: HeadIntervention = "none",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        leading = query_features.shape[:-1]
        candidates = key_features.shape[-2]
        q = self.query(query_features).reshape(*leading, self.heads, self.head_dim)
        k = self.key(key_features).reshape(*leading, candidates, self.heads, self.head_dim)
        v = self.value(value_features).reshape(*leading, candidates, self.heads, self.head_dim)
        logits = (q.unsqueeze(-3) * k).sum(-1) / math.sqrt(self.head_dim)
        if intervention == "uniform":
            weights = torch.full_like(logits, 1.0 / candidates)
        elif intervention == "frozen_center":
            weights = torch.zeros_like(logits)
            weights[..., candidates // 2 - 1 if frozen_index is None else frozen_index, :] = 1.0
        else:
            weights = torch.softmax(logits, dim=-2)
        if head_intervention != "none":
            if head_index is None or not 0 <= head_index < self.heads:
                raise ValueError(f"head_index must be in [0,{self.heads - 1}]")
            if head_intervention == "uniform_attention":
                weights = weights.clone()
                weights[..., head_index] = 1.0 / candidates
            elif head_intervention != "zero_context":
                raise ValueError(f"Unknown head intervention: {head_intervention}")
        context_by_head = (weights.unsqueeze(-1) * v).sum(-3)
        if head_intervention == "zero_context":
            context_by_head = context_by_head.clone()
            context_by_head[..., head_index, :] = 0
        context = context_by_head.reshape(*leading, -1)
        decoder_input = torch.cat((context, query_features), dim=-1) if self.use_query_skip else context
        return self.decoder(decoder_input), weights


def _batch_scalar(value: torch.Tensor | float, batch: int, reference: torch.Tensor) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if result.ndim == 0:
        result = result.repeat(batch)
    return result.reshape(batch)


def _stable_rms(difference: torch.Tensor, dims: int | tuple[int, ...]) -> torch.Tensor:
    """RMS-like scale that is exactly zero at zero with finite gradients."""
    mean_square = difference.square().mean(dims)
    return mean_square / torch.sqrt(mean_square + 1.0e-12)


class BurgersFaceAttention1D(nn.Module):
    """Conservative learned face flux for one-dimensional inviscid Burgers."""

    def __init__(
        self, d_model: int = 32, heads: int = 4, decoder_width: int = 64,
        query_skip: bool = True, consistent_flux: bool = False,
    ) -> None:
        super().__init__()
        self.offsets = (-2, -1, 0, 1, 2, 3)
        self.consistent_flux = consistent_flux
        self.attention = MultiHeadFaceAttention(
            8, 9, 3, 1, d_model, heads, decoder_width, query_skip,
            zero_output_init=consistent_flux,
        )

    def forward(
        self,
        state: torch.Tensor,
        dt_dx: torch.Tensor | float,
        intervention: Intervention = "learned",
        return_attention: bool = False,
        head_index: int | None = None,
        head_intervention: HeadIntervention = "none",
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if state.ndim == 1:
            state = state[None]
        batch = state.shape[0]
        lam = _batch_scalar(dt_dx, batch, state)[:, None]
        left, right = state, torch.roll(state, -1, -1)
        mean, jump = 0.5 * (left + right), right - left
        query = torch.stack(
            (left / 2, right / 2, mean / 2, jump / 2, lam.expand_as(left) / 2,
             lam * mean / 2, 0.25 * left.square(), 0.25 * right.square()),
            dim=-1,
        )
        values = torch.stack([torch.roll(state, -offset, -1) for offset in self.offsets], dim=-1)
        position = torch.as_tensor([offset - 0.5 for offset in self.offsets], dtype=state.dtype, device=state.device)
        position = position.view(1, 1, -1).expand_as(values)
        local_cfl = lam[..., None] * values
        alignment = position + lam[..., None] * mean[..., None]
        up, down = torch.roll(state, -1, -1), torch.roll(state, 1, -1)
        smooth_field = (up - 2 * state + down).abs() / (
            (up - state).abs() + (state - down).abs() + 1.0e-6 * (state.abs().mean(1, keepdim=True) + 1)
        )
        smoothness = torch.stack([torch.roll(smooth_field, -offset, -1) for offset in self.offsets], -1)
        if intervention == "cfl_blind":
            local_cfl = torch.zeros_like(local_cfl)
            alignment = position
            query = query.clone()
            query[..., 4:6] = 0
            intervention = "learned"
        key = torch.stack(
            (
                values / 2,
                0.25 * values.square(),
                local_cfl / 2,
                position / 2.5,
                smoothness,
                (values - mean[..., None]) / 2,
                alignment / 2.5,
                alignment.abs() / 2.5,
                torch.exp(-0.5 * alignment.square()),
            ),
            dim=-1,
        )
        value = torch.stack((values / 2, 0.25 * values.square(), local_cfl / 2), dim=-1)
        residual, weights = self.attention(
            query, key, value, intervention, frozen_index=self.offsets.index(0),
            head_index=head_index, head_intervention=head_intervention,
        )
        residual = residual.squeeze(-1)
        if self.consistent_flux:
            residual = torch.tanh(residual)
            variation = _stable_rms(values - mean[..., None], -1)
            base_flux = burgers_godunov_flux(left, right)
            flux = base_flux + variation * (mean.abs() + variation) * residual
        else:
            flux = residual
        return (flux, weights) if return_attention else flux


class BurgersStencilMLP1D(nn.Module):
    """Ordered-stencil non-attention control with the same physical inputs.

    The model receives the eight face-query features and the nine key features
    for each of the same six ordered candidates used by ``BurgersFaceAttention1D``.
    Its width is chosen in the configuration so that its parameter count is
    close to the attention model. It predicts one shared face flux and therefore
    retains exactly the same conservative finite-volume update.
    """

    def __init__(self, hidden_width: int = 60) -> None:
        super().__init__()
        self.offsets = (-2, -1, 0, 1, 2, 3)
        input_dim = 8 + len(self.offsets) * 9
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, 1),
        )

    def forward(
        self,
        state: torch.Tensor,
        dt_dx: torch.Tensor | float,
        intervention: Intervention = "learned",
        return_attention: bool = False,
        **_: object,
    ) -> torch.Tensor | tuple[torch.Tensor, None]:
        if intervention != "learned":
            raise ValueError("The stencil-MLP control has no attention intervention")
        if state.ndim == 1:
            state = state[None]
        batch = state.shape[0]
        lam = _batch_scalar(dt_dx, batch, state)[:, None]
        left, right = state, torch.roll(state, -1, -1)
        mean, jump = 0.5 * (left + right), right - left
        query = torch.stack(
            (left / 2, right / 2, mean / 2, jump / 2, lam.expand_as(left) / 2,
             lam * mean / 2, 0.25 * left.square(), 0.25 * right.square()),
            dim=-1,
        )
        values = torch.stack([torch.roll(state, -offset, -1) for offset in self.offsets], dim=-1)
        position = torch.as_tensor(
            [offset - 0.5 for offset in self.offsets], dtype=state.dtype, device=state.device
        ).view(1, 1, -1).expand_as(values)
        local_cfl = lam[..., None] * values
        alignment = position + lam[..., None] * mean[..., None]
        up, down = torch.roll(state, -1, -1), torch.roll(state, 1, -1)
        smooth_field = (up - 2 * state + down).abs() / (
            (up - state).abs() + (state - down).abs()
            + 1.0e-6 * (state.abs().mean(1, keepdim=True) + 1)
        )
        smoothness = torch.stack(
            [torch.roll(smooth_field, -offset, -1) for offset in self.offsets], -1
        )
        key = torch.stack(
            (
                values / 2,
                0.25 * values.square(),
                local_cfl / 2,
                position / 2.5,
                smoothness,
                (values - mean[..., None]) / 2,
                alignment / 2.5,
                alignment.abs() / 2.5,
                torch.exp(-0.5 * alignment.square()),
            ),
            dim=-1,
        )
        flux = self.decoder(torch.cat((query, key.flatten(-2)), -1)).squeeze(-1)
        return (flux, None) if return_attention else flux


class BurgersFaceAttention2D(nn.Module):
    """One shared orientation-equivariant face model on a Cartesian periodic grid."""

    def __init__(
        self, d_model: int = 32, heads: int = 4, decoder_width: int = 64,
        query_skip: bool = True, consistent_flux: bool = False,
    ) -> None:
        super().__init__()
        self.normal_offsets = (-2, -1, 0, 1, 2, 3)
        self.tangent_offsets = (-2, -1, 0, 1, 2)
        self.consistent_flux = consistent_flux
        self.attention = MultiHeadFaceAttention(
            11, 13, 5, 1, d_model, heads, decoder_width, query_skip,
            zero_output_init=consistent_flux,
        )

    def _face_flux(
        self,
        state: torch.Tensor,
        dt_dn: torch.Tensor | float,
        beta_n: torch.Tensor | float,
        beta_t: torch.Tensor | float,
        normal_dim: int,
        tangent_dim: int,
        intervention: Intervention,
        return_attention: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch = state.shape[0]
        lam = _batch_scalar(dt_dn, batch, state)[:, None, None]
        bn = _batch_scalar(beta_n, batch, state)[:, None, None]
        bt = _batch_scalar(beta_t, batch, state)[:, None, None]
        left, right = state, torch.roll(state, -1, normal_dim)
        mean, jump = 0.5 * (left + right), right - left
        nu_n, nu_t = lam * bn * mean, lam * bt * mean
        query = torch.stack(
            (left, right, mean, jump, lam.expand_as(left), nu_n, nu_t,
             bn.expand_as(left), bt.expand_as(left), 0.5 * bn * left.square(), 0.5 * bn * right.square()),
            -1,
        )
        normal_up, normal_down = torch.roll(state, -1, normal_dim), torch.roll(state, 1, normal_dim)
        tangent_up, tangent_down = torch.roll(state, -1, tangent_dim), torch.roll(state, 1, tangent_dim)
        second = (normal_up - 2 * state + normal_down).abs() + (tangent_up - 2 * state + tangent_down).abs()
        first = ((normal_up - state).abs() + (state - normal_down).abs()
                 + (tangent_up - state).abs() + (state - tangent_down).abs())
        state_scale = state.abs().mean((1, 2), keepdim=True)
        smooth_field = second / (first + 1.0e-6 * (state_scale + 1))
        candidates, candidate_smoothness, dn_values, dt_values = [], [], [], []
        for dn in self.normal_offsets:
            for tangent in self.tangent_offsets:
                candidate = torch.roll(torch.roll(state, -dn, normal_dim), -tangent, tangent_dim)
                candidates.append(candidate)
                candidate_smoothness.append(
                    torch.roll(torch.roll(smooth_field, -dn, normal_dim), -tangent, tangent_dim)
                )
                dn_values.append(dn - 0.5)
                dt_values.append(tangent)
        values = torch.stack(candidates, -1)
        dn = torch.as_tensor(dn_values, dtype=state.dtype, device=state.device).view(1, 1, 1, -1).expand_as(values)
        tangent = torch.as_tensor(dt_values, dtype=state.dtype, device=state.device).view(1, 1, 1, -1).expand_as(values)
        align_n = dn + nu_n[..., None]
        align_t = tangent + nu_t[..., None]
        smooth = torch.stack(candidate_smoothness, -1)
        if intervention == "cfl_blind":
            align_n, align_t = dn, tangent
            query = query.clone()
            query[..., 4:7] = 0
            intervention = "learned"
        normal_flux = 0.5 * bn[..., None] * values.square()
        tangent_flux = 0.5 * bt[..., None] * values.square()
        key = torch.stack(
            (values, normal_flux, tangent_flux, lam[..., None] * bn[..., None] * values,
             lam[..., None] * bt[..., None] * values, dn, tangent, smooth, values - mean[..., None],
             align_n, align_t, torch.sqrt(align_n.square() + align_t.square()),
             torch.exp(-0.5 * (align_n.square() + align_t.square()))),
            -1,
        )
        value = torch.stack(
            (values, normal_flux, tangent_flux, lam[..., None] * bn[..., None] * values,
             lam[..., None] * bt[..., None] * values), -1
        )
        center_index = self.normal_offsets.index(0) * len(self.tangent_offsets) + self.tangent_offsets.index(0)
        residual, weights = self.attention(query, key, value, intervention, frozen_index=center_index)
        residual = residual.squeeze(-1)
        if self.consistent_flux:
            residual = torch.tanh(residual)
            variation = _stable_rms(values - mean[..., None], -1)
            base_flux = burgers_rusanov_flux(left, right, bn)
            flux = base_flux + bn * variation * (mean.abs() + variation) * residual
        else:
            flux = residual
        return (flux, weights) if return_attention else flux

    def forward(
        self,
        state: torch.Tensor,
        dt_dx: torch.Tensor | float,
        dt_dy: torch.Tensor | float,
        beta_x: torch.Tensor | float,
        beta_y: torch.Tensor | float,
        intervention: Intervention = "learned",
        return_attention: bool = False,
    ):
        if state.ndim == 2:
            state = state[None]
        x = self._face_flux(state, dt_dx, beta_x, beta_y, -1, -2, intervention, return_attention)
        y = self._face_flux(state, dt_dy, beta_y, beta_x, -2, -1, intervention, return_attention)
        return (x, y)


class ShallowWaterFaceAttention1D(nn.Module):
    def __init__(
        self, gravity: float = 1.0, d_model: int = 32, heads: int = 4,
        decoder_width: int = 64, query_skip: bool = True, consistent_flux: bool = False,
    ) -> None:
        super().__init__()
        self.equation = ShallowWater1D(gravity=gravity)
        self.offsets = (-2, -1, 0, 1, 2, 3)
        self.consistent_flux = consistent_flux
        self.attention = MultiHeadFaceAttention(
            15, 16, 6, 2, d_model, heads, decoder_width, query_skip,
            zero_output_init=consistent_flux,
        )

    def forward(
        self,
        state: torch.Tensor,
        dt_dx: torch.Tensor | float,
        intervention: Intervention = "learned",
        return_attention: bool = False,
    ):
        if state.ndim == 2:
            state = state[None]
        batch = state.shape[0]
        lam = _batch_scalar(dt_dx, batch, state)[:, None]
        left, right = state, torch.roll(state, -1, -2)
        mean, jump = 0.5 * (left + right), right - left
        lm, lp = self.equation.wave_speeds(mean)
        fl, fr = self.equation.flux(left), self.equation.flux(right)
        query = torch.cat(
            (left, right, mean, jump, lam[..., None].expand(*left.shape[:-1], 1),
             lm[..., None], lp[..., None], fl, fr), -1
        )
        candidates = torch.stack([torch.roll(state, -offset, -2) for offset in self.offsets], -2)
        flux = self.equation.flux(candidates)
        cm, cp = self.equation.wave_speeds(candidates)
        reaches = torch.stack((lam[..., None] * cm, lam[..., None] * cp), -1)
        position = torch.as_tensor([offset - 0.5 for offset in self.offsets], dtype=state.dtype, device=state.device)
        position = position.view(1, 1, -1).expand_as(cm)
        align_m, align_p = position + lam[..., None] * lm[..., None], position + lam[..., None] * lp[..., None]
        up, down = torch.roll(state, -1, -2), torch.roll(state, 1, -2)
        second = torch.sqrt((up - 2 * state + down).square().sum(-1) + 1.0e-12)
        first = (torch.sqrt((up - state).square().sum(-1) + 1.0e-12)
                 + torch.sqrt((state - down).square().sum(-1) + 1.0e-12))
        state_scale = state.abs().mean((1, 2), keepdim=False)[:, None]
        smooth_field = second / (first + 1.0e-6 * (state_scale + 1))
        smooth = torch.stack([torch.roll(smooth_field, -offset, -1) for offset in self.offsets], -1)
        if intervention == "cfl_blind":
            reaches = torch.zeros_like(reaches)
            align_m = align_p = position
            query = query.clone()
            query[..., 8:11] = 0
            intervention = "learned"
        key = torch.cat(
            (candidates, flux, reaches, position[..., None], smooth[..., None],
             candidates - mean[..., None, :], align_m[..., None], align_m.abs()[..., None],
             torch.exp(-0.5 * align_m.square())[..., None], align_p[..., None],
             align_p.abs()[..., None], torch.exp(-0.5 * align_p.square())[..., None]), -1
        )
        value = torch.cat((candidates, flux, reaches), -1)
        residual, weights = self.attention(query, key, value, intervention, frozen_index=self.offsets.index(0))
        if self.consistent_flux:
            residual = torch.tanh(residual)
            h_scale = mean[..., 0].abs().clamp_min(1.0e-6)
            q_scale = (h_scale * torch.sqrt(self.equation.gravity * h_scale)).clamp_min(1.0e-6)
            normalized_difference = torch.stack(
                ((candidates[..., 0] - mean[..., None, 0]) / h_scale[..., None],
                 (candidates[..., 1] - mean[..., None, 1]) / q_scale[..., None]), -1
            )
            variation = _stable_rms(normalized_difference, (-1, -2))
            base_flux = shallow_water_hll_flux(left, right, self.equation)
            momentum_scale = (mean[..., 1].abs().square() / h_scale +
                              0.5 * self.equation.gravity * h_scale.square()).clamp_min(1.0e-6)
            flux_scale = torch.stack((q_scale, momentum_scale), -1)
            output = base_flux + variation[..., None] * flux_scale * residual
        else:
            output = residual
        return (output, weights) if return_attention else output


def build_model(problem: str, model_config: dict) -> nn.Module:
    architecture = str(model_config.get("architecture", "attention"))
    if architecture == "ordered_stencil_mlp":
        if problem != "burgers_1d":
            raise ValueError("ordered_stencil_mlp is defined only for burgers_1d")
        return BurgersStencilMLP1D(
            hidden_width=int(model_config.get("stencil_mlp_width", 60))
        )
    if architecture != "attention":
        raise ValueError(f"Unknown model architecture: {architecture}")
    common = {
        "d_model": int(model_config.get("d_model", 32)),
        "heads": int(model_config.get("heads", 4)),
        "decoder_width": int(model_config.get("decoder_width", 64)),
        "query_skip": bool(model_config.get("query_skip", True)),
        "consistent_flux": bool(model_config.get("consistent_flux", False)),
    }
    if problem == "burgers_1d":
        return BurgersFaceAttention1D(**common)
    if problem == "burgers_2d":
        return BurgersFaceAttention2D(**common)
    if problem == "shallow_water_1d":
        return ShallowWaterFaceAttention1D(gravity=float(model_config.get("gravity", 1.0)), **common)
    raise ValueError(f"Unknown problem: {problem}")
