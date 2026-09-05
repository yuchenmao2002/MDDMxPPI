"""吸收态表达扩散模型的单调反向采样。

This module intentionally owns *sampling policy*, not network architecture or
training loss.  :class:`ReverseSampler` repeatedly evaluates one trained
denoiser on a descending time grid.

采样步数为 K，采用线性网格

    t_k = k / K,    k = 0, ..., K

代码按 k 递减枚举（``grid[step_index] = t_{K - step_index}``），初始状态
``x_{t_K}`` 是全部掩码的序列。前向生存律为 ``P(t 时刻仍被掩码) = t``，
等价地 ``P(t 时刻可见) = 1 - t``。

在 ``t_k -> t_{k-1}`` 的反向步骤中，每个**仍被掩码**的基因以

    r_k = 1 - t_{k-1} / t_k = 1 / k

的概率被揭示。代码写成 ``(current_time - next_time) / current_time``，
即从网格自身求 r，而**不是**直接代入闭式 ``1 / k``：这样存活概率的连乘会精确收缩为
``grid[-1] / grid[0]``，实测各步累计掩码率与 ``t_k`` 的偏差仅 2e-09，比代入闭式更稳。
由于 ``r_1 = 1``，最后一步揭示全部剩余 MASK；该步在代码中走独立的精确全揭示分支，
不依赖 ``rand < 1``。

本步新揭示的基因 i 从 hurdle 分布采样

    x_{t_{k-1}}^i | x_{t_k}  ~  (1 - pi_{t_k}^i) delta_0
                               + pi_{t_k}^i TN_(0,inf)(mu_{t_k}^i, (sigma_{t_k}^i)^2)

三个参数 (pi, mu, sigma) 全部取自**当前时刻 t_k** 的 denoiser 输出。

Already-visible values are immutable.  The reveal gate is independent of the
hurdle value, so the implementation samples hurdle values lazily only for the
tokens selected for reveal.  This is distributionally identical to drawing all
masked-token candidates and discarding the unselected ones, while avoiding up
to ``O(K * B * G)`` unnecessary truncated-Normal draws.  The denoiser still
predicts every masked token again at the next time.  There is no re-masking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import Generator, Tensor, nn

from src.models.types import HurdleDistributionParameters, ModelOutput


_FLOAT32_MIN_POSITIVE = torch.finfo(torch.float32).tiny
_FLOAT32_MAX = torch.finfo(torch.float32).max
_MAX_REJECTION_ROUNDS = 10_000


@dataclass(frozen=True)
class SamplingConfig:
    """Configuration of one reverse sampling run.

    ``num_steps`` is the number ``K`` of denoiser evaluations.  The descending
    linear grid is ``t_k = k / K`` for ``k = 0, ..., K``: it contains ``K + 1``
    endpoints, including exactly 1 and 0.  Only ``linear`` is accepted until
    another schedule is paired with a derived reveal transition rather than
    introduced as an unvalidated heuristic.
    """

    num_steps: int
    schedule: str = "linear"

    def __post_init__(self) -> None:
        if (
            isinstance(self.num_steps, bool)
            or not isinstance(self.num_steps, int)
            or self.num_steps < 1
        ):
            raise ValueError("num_steps must be an integer greater than or equal to 1.")
        if self.schedule != "linear":
            raise ValueError("Only the linear reverse-time schedule is supported.")


@dataclass(frozen=True)
class ReverseSamplingState:
    """One optional, memory-heavy snapshot of the reverse chain.

    ``diffusion_time`` is the scalar grid time shared by the batch.  Tensor
    snapshots are cloned, so requesting a trajectory stores ``K + 1`` full
    expression matrices and masks.  It is intended for tests and diagnostics,
    not routine 19,295-gene generation.
    """

    diffusion_time: float
    expression_values: Tensor  # FP32 [B,G,1]
    diffusion_mask: Tensor  # bool [B,G], True means still MASK


@dataclass(frozen=True)
class ReverseSamplingDiagnostics:
    """Small CPU metadata collected without retaining model-sized tensors."""

    time_grid: Tuple[float, ...]  # descending, length K+1
    masked_counts: Tuple[int, ...]  # aggregate batch counts, length K+1
    revealed_counts: Tuple[int, ...]  # aggregate counts, length K
    denoiser_calls: int


@dataclass(frozen=True)
class ReverseSamplingOutput:
    """Generated expression and optional reverse-chain metadata."""

    expression_values: Tensor  # FP32 [B,G,1], finite and nonnegative
    final_mask: Tensor  # bool [B,G], guaranteed all False
    diagnostics: Optional[ReverseSamplingDiagnostics] = None
    trajectory: Optional[Tuple[ReverseSamplingState, ...]] = None


def linear_time_grid(
    num_steps: int,
    *,
    device: Optional[Union[str, torch.device]] = None,
) -> Tensor:
    """Return the FP32 descending grid ``[1, ..., 0]`` for ``K=num_steps``.

    Endpoints are assigned explicitly so downstream final-step guarantees do
    not depend on floating-point interpolation details.
    """

    if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps < 1:
        raise ValueError("num_steps must be an integer greater than or equal to 1.")
    grid = torch.linspace(
        1.0,
        0.0,
        steps=num_steps + 1,
        dtype=torch.float32,
        device=device,
    )
    grid[0] = 1.0
    grid[-1] = 0.0
    return grid


def _canonical_device(device: torch.device) -> torch.device:
    """Resolve an index-less CUDA device to the process's current device."""

    normalized = torch.device(device)
    if normalized.type == "cuda" and normalized.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return normalized


def _validate_generator_device(
    generator: Optional[Generator],
    device: torch.device,
) -> None:
    if generator is None:
        return
    generator_device = _canonical_device(torch.device(generator.device))
    device = _canonical_device(device)
    if generator_device != device:
        raise ValueError(
            "generator and sampled tensors must be on the same device; "
            f"got generator on {generator_device} and tensors on {device}."
        )


def _uniform_open_zero(
    shape: Tuple[int, ...],
    *,
    device: torch.device,
    generator: Optional[Generator],
) -> Tensor:
    """Draw FP32 uniforms in ``(0,1)`` for logarithmic transforms."""

    uniform = torch.rand(
        shape,
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    # torch.rand may return exactly zero.  Replacing that single endpoint avoids
    # a zero exponential variate while changing only an unresolvable tail below
    # FP32 RNG precision.
    return uniform.clamp_min_(_FLOAT32_MIN_POSITIVE)


def _sample_standard_normal_above(
    lower_bound: Tensor,
    *,
    generator: Optional[Generator],
) -> Tuple[Tensor, Tensor]:
    """Exactly sample ``Normal(0,1)`` conditional on ``Z > lower_bound``.

    Ordinary Normal rejection is efficient for negative bounds (acceptance at
    least one half).  For nonnegative bounds this uses Robert's exponential
    rejection sampler, whose acceptance remains high even deep in the tail.
    This avoids inverse-CDF cancellation when ``Phi(lower_bound)`` rounds to 1.
    """

    if lower_bound.dtype != torch.float32:
        raise TypeError("lower_bound must have dtype torch.float32.")
    # The sole caller saturates its input with ``nan_to_num`` before this point,
    # so a finiteness scan here would only add a synchronizing device-to-host
    # copy inside the per-step sampling loop.

    flat_bound = lower_bound.reshape(-1)
    samples = torch.empty_like(flat_bound)
    excesses = torch.empty_like(flat_bound)

    negative_indices = torch.nonzero(flat_bound < 0.0, as_tuple=False).flatten()
    rounds = 0
    while negative_indices.numel() > 0:
        rounds += 1
        if rounds > _MAX_REJECTION_ROUNDS:
            raise RuntimeError("Truncated-Normal rejection sampler did not converge.")
        proposals = torch.randn(
            (negative_indices.numel(),),
            dtype=torch.float32,
            device=flat_bound.device,
            generator=generator,
        )
        accepted = proposals > flat_bound[negative_indices]
        accepted_indices = negative_indices[accepted]
        samples[accepted_indices] = proposals[accepted]
        excesses[accepted_indices] = (
            proposals[accepted] - flat_bound[accepted_indices]
        )
        negative_indices = negative_indices[~accepted]

    nonnegative_indices = torch.nonzero(
        flat_bound >= 0.0,
        as_tuple=False,
    ).flatten()
    rounds = 0
    while nonnegative_indices.numel() > 0:
        rounds += 1
        if rounds > _MAX_REJECTION_ROUNDS:
            raise RuntimeError("Truncated-Normal tail sampler did not converge.")

        alpha = flat_bound[nonnegative_indices]
        # lambda=(alpha+sqrt(alpha**2+4))/2, evaluated without alpha**2 or
        # adding two near-FP32-max numbers.
        root = torch.hypot(alpha, torch.full_like(alpha, 2.0))
        rate = alpha + 2.0 / (root + alpha).clamp_min(
            _FLOAT32_MIN_POSITIVE
        )
        exponential = -torch.log1p(
            -_uniform_open_zero(
                (nonnegative_indices.numel(),),
                device=flat_bound.device,
                generator=generator,
            )
        ) / rate
        proposals = alpha + exponential
        log_acceptance = -0.5 * (proposals - rate).square()
        log_uniform = torch.log(
            _uniform_open_zero(
                (nonnegative_indices.numel(),),
                device=flat_bound.device,
                generator=generator,
            )
        )
        accepted = log_uniform <= log_acceptance
        accepted_indices = nonnegative_indices[accepted]
        samples[accepted_indices] = proposals[accepted]
        # Preserve the proposal's exact excess before ``alpha + exponential``
        # rounds back to alpha in a deep FP32 tail.  This excess is what the
        # caller needs to form location + scale * Z without cancellation.
        excesses[accepted_indices] = exponential[accepted]
        nonnegative_indices = nonnegative_indices[~accepted]

    return (
        samples.reshape(lower_bound.shape),
        excesses.reshape(lower_bound.shape),
    )


def sample_zero_truncated_normal(
    location: Tensor,
    scale: Tensor,
    *,
    generator: Optional[Generator] = None,
) -> Tensor:
    """Sample ``Normal(location, scale)`` conditional on being strictly positive.

    Inputs and the result are FP32 and share shape/device.  The positive-tail
    branch forms the final value as ``scale * (Z - alpha)`` instead of
    ``location + scale * Z`` to avoid catastrophic cancellation when location
    is far below zero.  Values outside representable FP32 range are saturated;
    a positive-component draw is never converted into the hurdle's exact zero.
    """

    if not isinstance(location, Tensor) or not isinstance(scale, Tensor):
        raise TypeError("location and scale must be torch.Tensor instances.")
    if location.shape != scale.shape:
        raise ValueError("location and scale must have identical shapes.")
    if location.device != scale.device:
        raise ValueError("location and scale must be on the same device.")
    if location.dtype != torch.float32 or scale.dtype != torch.float32:
        raise TypeError("location and scale must have dtype torch.float32.")
    # Finiteness and strict positivity are contracts of the decoder's hurdle
    # parameters; ``sample_hurdle_distribution`` is the only production caller
    # and repeating the scans here costs three synchronizing copies per step.
    _validate_generator_device(generator, location.device)

    # Division can overflow even for individually finite decoder parameters.
    # Saturating alpha preserves the limiting tail regime and keeps the exact
    # rejection sampler numerically defined.
    alpha = torch.nan_to_num(
        -location / scale,
        nan=0.0,
        posinf=_FLOAT32_MAX,
        neginf=-_FLOAT32_MAX,
    )
    standard_sample, standard_excess = _sample_standard_normal_above(
        alpha,
        generator=generator,
    )
    ordinary_value = location + scale * standard_sample
    tail_value = scale * standard_excess
    value = torch.where(alpha >= 0.0, tail_value, ordinary_value)
    return torch.nan_to_num(
        value,
        nan=_FLOAT32_MIN_POSITIVE,
        posinf=_FLOAT32_MAX,
        neginf=_FLOAT32_MIN_POSITIVE,
    ).clamp_(min=_FLOAT32_MIN_POSITIVE, max=_FLOAT32_MAX)


def sample_hurdle_distribution(
    parameters: HurdleDistributionParameters,
    *,
    generator: Optional[Generator] = None,
    selection_mask: Optional[Tensor] = None,
) -> Tensor:
    """Draw exact-zero/positive values from decoder hurdle parameters.

    Parameter tensors must be FP32 ``[B,G,1]`` with identical shape/device.  If
    ``selection_mask[B,G]`` is supplied, randomness is consumed only for those
    entries and unselected result entries remain zero.  The reverse sampler
    passes the independent reveal mask, so it draws values only for tokens that
    become visible in the current transition.
    """

    if not isinstance(parameters, HurdleDistributionParameters):
        raise TypeError("parameters must be HurdleDistributionParameters.")
    logits = parameters.detection_logits
    location = parameters.positive_location
    scale = parameters.positive_scale
    for name, tensor in (
        ("detection_logits", logits),
        ("positive_location", location),
        ("positive_scale", scale),
    ):
        if not isinstance(tensor, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")
        if tensor.dtype != torch.float32:
            raise TypeError(f"{name} must have dtype torch.float32.")
        if tensor.ndim != 3 or tensor.shape[-1] != 1:
            raise ValueError(f"{name} must have shape [B,G,1].")
    if location.shape != logits.shape or scale.shape != logits.shape:
        raise ValueError("All hurdle parameter tensors must have identical shapes.")
    if location.device != logits.device or scale.device != logits.device:
        raise ValueError("All hurdle parameter tensors must share a device.")
    # Shape, dtype and device are checked because they cost nothing.  Elementwise
    # finiteness and ``scale > 0`` are not: the decoder derives scale as
    # ``min_scale + softplus(raw)`` and emits FP32 parameters outside autocast,
    # so those scans would only synchronize the reverse loop once per step.
    _validate_generator_device(generator, logits.device)

    expected_mask_shape = logits.shape[:-1]
    if selection_mask is None:
        selection_mask = torch.ones(
            expected_mask_shape,
            dtype=torch.bool,
            device=logits.device,
        )
    elif not isinstance(selection_mask, Tensor):
        raise TypeError("selection_mask must be a torch.Tensor.")
    elif selection_mask.dtype != torch.bool:
        raise TypeError("selection_mask must have dtype torch.bool.")
    elif selection_mask.shape != expected_mask_shape:
        raise ValueError(
            f"selection_mask must have shape {tuple(expected_mask_shape)}."
        )
    elif selection_mask.device != logits.device:
        raise ValueError("selection_mask and hurdle parameters must share a device.")

    result = torch.zeros_like(logits, dtype=torch.float32)
    # Materialize the selection once as an index tuple.  Every data-dependent
    # output size forces a device-to-host copy, so three ``masked_select`` calls
    # sharing one mask -- plus the later boolean indexing -- would synchronize
    # the reverse loop six times per step instead of twice.
    selection_index = selection_mask.nonzero(as_tuple=True)
    selected_logits = logits.squeeze(-1)[selection_index]
    if selected_logits.numel() == 0:
        return result

    detection_uniform = torch.rand(
        selected_logits.shape,
        dtype=torch.float32,
        device=logits.device,
        generator=generator,
    )
    selected_positive = detection_uniform < torch.sigmoid(selected_logits)
    positive_index = selected_positive.nonzero(as_tuple=True)[0]
    selected_values = torch.zeros_like(selected_logits)
    if positive_index.numel() > 0:
        # ``numel`` is already known on the host after ``nonzero``, and indexing
        # with an explicit index tensor keeps every later shape static.
        selected_values[positive_index] = sample_zero_truncated_normal(
            location.squeeze(-1)[selection_index][positive_index],
            scale.squeeze(-1)[selection_index][positive_index],
            generator=generator,
        )
    result.squeeze(-1).masked_scatter_(selection_mask, selected_values)
    return result


class ReverseSampler:
    """Generate unconditional expression batches with a monotone reverse chain.

    The sampler starts at an all-MASK state with zero numeric placeholders and
    evaluates the same denoiser exactly ``K`` times.  It temporarily switches
    the denoiser to evaluation mode and restores the exact per-module training
    flags afterwards.  Sampling runs under :func:`torch.inference_mode`, but
    parameter ``requires_grad`` flags and weights are never changed.

    Mixed-precision autocast is intentionally a caller concern.  This keeps the
    core usable on CPU and under either BF16 or FP32 inference while the decoder
    continues to expose FP32 probability parameters.
    """

    def __init__(self, denoiser: nn.Module, config: SamplingConfig) -> None:
        if not isinstance(denoiser, nn.Module):
            raise TypeError("denoiser must be a torch.nn.Module.")
        if not isinstance(config, SamplingConfig):
            raise TypeError("config must be SamplingConfig.")
        num_genes = getattr(denoiser, "num_genes", None)
        if (
            isinstance(num_genes, bool)
            or not isinstance(num_genes, int)
            or num_genes < 1
        ):
            raise ValueError("denoiser.num_genes must be a positive integer.")
        self.denoiser = denoiser
        self.config = config
        self.num_genes = num_genes

    def _default_device(self) -> torch.device:
        for tensor in self.denoiser.parameters():
            return tensor.device
        for tensor in self.denoiser.buffers():
            return tensor.device
        return torch.device("cpu")

    def _validate_model_device(self, device: torch.device) -> None:
        for tensor in tuple(self.denoiser.parameters()) + tuple(
            self.denoiser.buffers()
        ):
            if tensor.device != device:
                raise ValueError(
                    "All denoiser parameters/buffers and sampling tensors must "
                    f"share one device; found {tensor.device} and {device}."
                )

    @staticmethod
    def _make_generator(device: torch.device, seed: int) -> Generator:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer.")
        try:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
        except RuntimeError as exc:
            raise ValueError(f"Invalid random seed {seed!r}.") from exc
        return generator

    def sample(
        self,
        batch_size: int,
        *,
        device: Optional[Union[str, torch.device]] = None,
        seed: Optional[int] = None,
        generator: Optional[Generator] = None,
        return_diagnostics: bool = False,
        return_trajectory: bool = False,
    ) -> ReverseSamplingOutput:
        """Generate ``batch_size`` cells from an all-MASK initial state.

        ``seed`` creates a private device-local generator and is mutually
        exclusive with ``generator``.  Passing neither uses PyTorch's current
        device-global RNG.  A trajectory is disabled by default because its
        storage is ``O((K+1) * B * G)``.
        """

        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer.")
        if not isinstance(return_diagnostics, bool):
            raise TypeError("return_diagnostics must be a boolean.")
        if not isinstance(return_trajectory, bool):
            raise TypeError("return_trajectory must be a boolean.")
        if seed is not None and generator is not None:
            raise ValueError("Pass either seed or generator, not both.")

        target_device = _canonical_device(
            self._default_device() if device is None else torch.device(device)
        )
        self._validate_model_device(target_device)
        if seed is not None:
            generator = self._make_generator(target_device, seed)
        _validate_generator_device(generator, target_device)

        grid = linear_time_grid(self.config.num_steps, device=target_device)
        expression_values = torch.zeros(
            (batch_size, self.num_genes, 1),
            dtype=torch.float32,
            device=target_device,
        )
        diffusion_mask = torch.ones(
            (batch_size, self.num_genes),
            dtype=torch.bool,
            device=target_device,
        )

        time_values = (
            tuple(float(value) for value in grid.detach().cpu().tolist())
            if return_diagnostics
            else ()
        )
        masked_counts = [batch_size * self.num_genes] if return_diagnostics else []
        revealed_counts = []
        trajectory = (
            [
                ReverseSamplingState(
                    diffusion_time=1.0,
                    expression_values=expression_values.clone(),
                    diffusion_mask=diffusion_mask.clone(),
                )
            ]
            if return_trajectory
            else None
        )

        modules_and_modes = tuple(
            (module, module.training) for module in self.denoiser.modules()
        )
        self.denoiser.eval()
        try:
            with torch.inference_mode():
                for step_index in range(self.config.num_steps):
                    current_time = grid[step_index]
                    next_time = grid[step_index + 1]
                    diffusion_time = current_time.expand(batch_size).clone()

                    model_output = self.denoiser(
                        expression_values,
                        diffusion_time,
                        diffusion_mask,
                        return_hidden_state=False,
                        output_hidden_states=False,
                        return_diagnostics=False,
                        compute_point_prediction=False,
                    )
                    if not isinstance(model_output, ModelOutput):
                        raise TypeError(
                            "denoiser must return ModelOutput, got "
                            f"{type(model_output).__name__}."
                        )
                    decoder_output = model_output.decoder_output
                    if decoder_output is None:
                        raise ValueError("denoiser output is missing decoder_output.")
                    parameters = decoder_output.distribution_parameters
                    if parameters is None:
                        raise ValueError(
                            "denoiser output is missing hurdle distribution parameters."
                        )

                    if step_index == self.config.num_steps - 1:
                        # Exact final reveal avoids relying on rand < 1 and makes
                        # the no-MASK postcondition explicit.
                        reveal_mask = diffusion_mask.clone()
                    else:
                        reveal_probability = (
                            (current_time - next_time) / current_time
                        )
                        reveal_uniform = torch.rand(
                            diffusion_mask.shape,
                            dtype=torch.float32,
                            device=target_device,
                            generator=generator,
                        )
                        reveal_mask = diffusion_mask & (
                            reveal_uniform < reveal_probability
                        )

                    # The reveal gate is independent of decoder values.  Lazy
                    # sampling only selected entries is therefore exactly
                    # equivalent to drawing every masked-token candidate and
                    # discarding those not revealed.  This optimization is
                    # material for large K and the 19,295-gene vocabulary.
                    candidates = sample_hurdle_distribution(
                        parameters,
                        generator=generator,
                        selection_mask=reveal_mask,
                    )

                    expression_values = torch.where(
                        reveal_mask.unsqueeze(-1),
                        candidates,
                        expression_values,
                    )
                    diffusion_mask = diffusion_mask & ~reveal_mask

                    if return_diagnostics:
                        revealed_counts.append(int(reveal_mask.sum().item()))
                        masked_counts.append(int(diffusion_mask.sum().item()))
                    if trajectory is not None:
                        trajectory.append(
                            ReverseSamplingState(
                                diffusion_time=float(next_time.item()),
                                expression_values=expression_values.clone(),
                                diffusion_mask=diffusion_mask.clone(),
                            )
                        )
        finally:
            # Restore even heterogeneous submodule flags exactly, without
            # recursively overwriting child state during restoration.
            for module, was_training in modules_and_modes:
                module.training = was_training

        if bool(diffusion_mask.any().item()):
            raise RuntimeError(
                "Final reverse-sampling state still contains MASK tokens."
            )
        if not bool(torch.isfinite(expression_values).all().item()):
            raise RuntimeError("Generated expression contains non-finite values.")
        if bool((expression_values < 0.0).any().item()):
            raise RuntimeError("Generated expression contains negative values.")

        diagnostics = (
            ReverseSamplingDiagnostics(
                time_grid=time_values,
                masked_counts=tuple(masked_counts),
                revealed_counts=tuple(revealed_counts),
                denoiser_calls=self.config.num_steps,
            )
            if return_diagnostics
            else None
        )
        return ReverseSamplingOutput(
            expression_values=expression_values,
            final_mask=diffusion_mask,
            diagnostics=diagnostics,
            trajectory=tuple(trajectory) if trajectory is not None else None,
        )


__all__ = [
    "ReverseSampler",
    "ReverseSamplingDiagnostics",
    "ReverseSamplingOutput",
    "ReverseSamplingState",
    "SamplingConfig",
    "linear_time_grid",
    "sample_hurdle_distribution",
    "sample_zero_truncated_normal",
]
