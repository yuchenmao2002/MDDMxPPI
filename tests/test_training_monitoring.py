"""Regression tests for optimizer-step training-window monitoring."""

from __future__ import annotations

import json
import math
from pathlib import Path
import signal
from types import SimpleNamespace
from typing import Iterable

import pytest
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from scripts import train_masked_diffusion as training


class _ValueDataset(Dataset[Tensor]):
    def __init__(self, values: Iterable[float]) -> None:
        self.values = tuple(float(value) for value in values)

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> Tensor:
        return torch.tensor([self.values[index]], dtype=torch.float32)


class _ToyRunner:
    """Produce differentiable losses and hand-checkable sufficient statistics."""

    def __init__(self, model: nn.Module, *, request_stop_on_call: int | None = None):
        self.model = model
        self.request_stop_on_call = request_stop_on_call
        self.calls = 0

    def __call__(
        self,
        clean_expression: Tensor,
        *,
        generator: torch.Generator,
    ) -> SimpleNamespace:
        del generator
        self.calls += 1
        if self.calls == self.request_stop_on_call:
            training.STOP_REQUEST.requested = True
            training.STOP_REQUEST.signal_number = signal.SIGTERM

        prediction = self.model(clean_expression)
        loss = prediction.square().mean()
        weighted_nll_sum = clean_expression.detach().sum()
        count = torch.tensor(clean_expression.shape[0], dtype=torch.int64)
        zero = torch.zeros((), dtype=torch.float32)
        zero_count = torch.zeros((), dtype=torch.int64)
        return SimpleNamespace(
            loss=loss,
            weighted_nll_sum=weighted_nll_sum,
            normalizer=count,
            weighted_zero_nll_sum=weighted_nll_sum,
            weighted_positive_nll_sum=zero,
            masked_count=count,
            masked_zero_count=count,
            masked_positive_count=zero_count,
            cell_count=count,
        )


class _CountingScheduler:
    def __init__(self) -> None:
        self.steps = 0

    def step(self) -> None:
        self.steps += 1


class _PatternScaler:
    """CPU loss-scaler double whose scale drops on configured skipped attempts."""

    def __init__(self, skipped_attempts: Iterable[bool] = ()) -> None:
        self.skipped_attempts = tuple(skipped_attempts)
        self.attempt_index = 0
        self.current_scale = 8.0 if self.skipped_attempts else 1.0
        self._skip_current = False

    def is_enabled(self) -> bool:
        return bool(self.skipped_attempts)

    def scale(self, loss: Tensor) -> Tensor:
        return loss

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        del optimizer

    def get_scale(self) -> float:
        return self.current_scale

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        self._skip_current = (
            self.skipped_attempts[self.attempt_index]
            if self.attempt_index < len(self.skipped_attempts)
            else False
        )
        if not self._skip_current:
            optimizer.step()

    def update(self) -> None:
        if self._skip_current:
            self.current_scale /= 2.0
        self.attempt_index += 1


@pytest.fixture(autouse=True)
def _reset_stop_request() -> Iterable[None]:
    training.STOP_REQUEST.requested = False
    training.STOP_REQUEST.signal_number = None
    yield
    training.STOP_REQUEST.requested = False
    training.STOP_REQUEST.signal_number = None


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _run_epoch(
    *,
    values: Iterable[float],
    batch_size: int,
    accumulation_steps: int,
    log_every: int,
    metrics_path: Path,
    initial_global_step: int = 0,
    skipped_attempts: Iterable[bool] = (),
    max_grad_norm: float = 1.0,
    request_stop_on_call: int | None = None,
) -> tuple[dict[str, object], int, bool, _CountingScheduler]:
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    runner = _ToyRunner(model, request_stop_on_call=request_stop_on_call)
    loader = DataLoader(
        _ValueDataset(values),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    scheduler = _CountingScheduler()
    scaler = _PatternScaler(skipped_attempts)

    metrics, global_step, interrupted = training.train_one_epoch(
        runner=runner,
        model=model,
        loader=loader,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        diffusion_generator=torch.Generator(device="cpu"),
        device=torch.device("cpu"),
        autocast_enabled=False,
        autocast_dtype=torch.float32,
        accumulation_steps=accumulation_steps,
        max_grad_norm=max_grad_norm,
        log_every=log_every,
        metrics_path=metrics_path,
        epoch=2,
        global_step=initial_global_step,
    )
    return metrics, global_step, interrupted, scheduler


def test_interval_window_uses_exact_ragged_sufficient_statistics(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    epoch_metrics, global_step, interrupted, scheduler = _run_epoch(
        values=(1, 2, 3, 4, 5),
        batch_size=2,
        accumulation_steps=2,
        log_every=2,
        metrics_path=metrics_path,
        initial_global_step=11,
        max_grad_norm=1e-8,
    )

    records = _read_jsonl(metrics_path)
    assert len(records) == 1  # epoch-end flush must not duplicate a full window
    window = records[0]
    assert window["event"] == "train_window"
    assert window["flush_reason"] == "interval"
    assert window["epoch"] == 3
    assert window["batch"] == window["batches"] == 3
    assert window["global_step"] == global_step == 13
    assert window["window_optimizer_steps"] == 2
    assert window["window_optimizer_attempts"] == 2
    assert window["window_skipped_optimizer_steps"] == 0
    assert window["window_cell_count"] == 5
    assert window["window_normalizer"] == 5
    assert window["window_weighted_nll_sum"] == pytest.approx(15.0)
    assert window["train_window_loss"] == pytest.approx(3.0)
    assert window["loss"] == pytest.approx(3.0)
    assert window["gradient_clip_rate"] == pytest.approx(1.0)
    assert window["learning_rate"] == window["lr"]
    assert math.isfinite(float(window["grad_norm"]))
    assert window["cells_per_second"] == window["window_cells_per_second"]
    assert float(window["window_cells_per_second"]) > 0.0
    assert scheduler.steps == 2
    assert epoch_metrics["loss"] == pytest.approx(3.0)
    assert epoch_metrics["cell_count"] == 5
    assert interrupted is False
    assert '"event": "train_window"' in capsys.readouterr().out


def test_interval_windows_are_non_overlapping_and_reset_after_each_log(
    tmp_path: Path,
) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    _, global_step, interrupted, _ = _run_epoch(
        values=(1, 2, 30, 40),
        batch_size=1,
        accumulation_steps=1,
        log_every=2,
        metrics_path=metrics_path,
    )

    records = _read_jsonl(metrics_path)
    assert [record["flush_reason"] for record in records] == [
        "interval",
        "interval",
    ]
    assert [record["global_step"] for record in records] == [2, 4]
    assert [record["window_cell_count"] for record in records] == [2, 2]
    assert records[0]["train_window_loss"] == pytest.approx(1.5)
    assert records[1]["train_window_loss"] == pytest.approx(35.0)
    assert global_step == 4
    assert interrupted is False


def test_skipped_fp16_attempt_is_retained_but_does_not_close_window(
    tmp_path: Path,
) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    epoch_metrics, global_step, interrupted, scheduler = _run_epoch(
        values=(100, 2, 3),
        batch_size=1,
        accumulation_steps=1,
        log_every=3,
        metrics_path=metrics_path,
        initial_global_step=7,
        skipped_attempts=(True, False, False),
        max_grad_norm=1e6,
    )

    records = _read_jsonl(metrics_path)
    assert len(records) == 1
    window = records[0]
    # The resumed/global counter reaching a multiple of log_every is irrelevant:
    # only two successful steps occurred in this epoch-local window.
    assert window["flush_reason"] == "epoch_end"
    assert window["global_step"] == global_step == 9
    assert window["window_optimizer_steps"] == 2
    assert window["window_optimizer_attempts"] == 3
    assert window["window_skipped_optimizer_steps"] == 1
    assert window["window_cell_count"] == 3
    assert window["window_normalizer"] == 3
    assert window["window_weighted_nll_sum"] == pytest.approx(105.0)
    assert window["train_window_loss"] == pytest.approx(35.0)
    assert window["loss"] == pytest.approx(35.0)
    assert window["gradient_clip_rate"] == pytest.approx(0.0)
    assert scheduler.steps == 2
    assert epoch_metrics["loss"] == pytest.approx(35.0)
    assert interrupted is False


def test_signal_flushes_skipped_only_partial_window_with_null_grad_fields(
    tmp_path: Path,
) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    epoch_metrics, global_step, interrupted, scheduler = _run_epoch(
        values=(7, 8),
        batch_size=1,
        accumulation_steps=1,
        log_every=5,
        metrics_path=metrics_path,
        initial_global_step=4,
        skipped_attempts=(True,),
        request_stop_on_call=1,
    )

    records = _read_jsonl(metrics_path)
    assert len(records) == 1
    window = records[0]
    assert window["flush_reason"] == "interrupted"
    assert window["global_step"] == global_step == 4
    assert window["window_optimizer_steps"] == 0
    assert window["window_optimizer_attempts"] == 1
    assert window["window_skipped_optimizer_steps"] == 1
    assert window["window_cell_count"] == 1
    assert window["loss"] == pytest.approx(7.0)
    assert window["grad_norm"] is None
    assert window["gradient_clip_rate"] is None
    assert scheduler.steps == 0
    assert epoch_metrics["loss"] == pytest.approx(7.0)
    assert interrupted is True
