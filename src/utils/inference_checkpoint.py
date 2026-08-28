"""Strict model-only loading for PPIL training checkpoints.

Training checkpoints intentionally contain optimizer, scheduler and RNG state
for exact recovery.  Inference needs none of those objects.  This module reads
the existing v3 container, validates its model/data contracts, reconstructs the
network directly from the learned gene-embedding tensor stored in the
``state_dict``, and copies only model tensors into a fresh module.

The v3 training format contains NumPy RNG objects and therefore cannot be read
with PyTorch's restricted ``weights_only=True`` unpickler.  Callers must make an
explicit trust decision before ordinary pickle deserialization.  Never load an
untrusted checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any, Mapping, Union

import torch
from torch import Tensor

from src.models.backbone import build_performer_backbone
from src.models.config import (
    ARCHITECTURE_VERSION,
    NUM_GENES,
    DecoderConfig,
    ForwardProcessConfig,
    GeneExpressionEncoderConfig,
    GeneIdentityEncoderConfig,
    LossConfig,
    MaskedDiffusionModelConfig,
    PerformerConfig,
)
from src.models.gene_expression_decoder import GeneExpressionDecoder
from src.models.gene_expression_encoder import GeneExpressionEncoder
from src.models.gene_identity_encoder import GeneIdentityEncoder
from src.models.losses import TimeWeightedHurdleNLLLoss
from src.models.masked_diffusion_training import MaskedDiffusionTrainingModule
from src.models.masked_expression_denoiser import MaskedExpressionDenoiser
from src.models.masking import AbsorbingMaskForwardProcess, AbsorbingStateEmbedding
from src.utils.checkpoint import sha256_file


SUPPORTED_CHECKPOINT_FORMAT_VERSION = 3
PRIMARY_VALIDATION_METRIC = "val_time_weighted_hurdle_nll"
_IDENTITY_WEIGHT_KEY = "denoiser.gene_identity_encoder.embedding.weight"


@dataclass(frozen=True)
class InferenceCheckpointMetadata:
    """Audited provenance retained after training-only state is discarded."""

    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint_format_version: int
    architecture_version: str
    reason: str
    current_epoch: int
    epoch_completed: bool
    next_epoch: int
    global_step: int
    primary_validation_metric: str
    best_primary_validation_metric: float
    model_config: MaskedDiffusionModelConfig
    data_contract: dict[str, Any]


@dataclass(frozen=True)
class LoadedInferenceModel:
    """Fresh strict-loaded model and its immutable checkpoint provenance."""

    model: MaskedDiffusionTrainingModule
    metadata: InferenceCheckpointMetadata


def _require_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return dict(value)


def _component(
    root: Mapping[str, Any],
    name: str,
    config_type: type,
    *,
    path_fields: tuple[str, ...] = (),
) -> Any:
    raw = _require_mapping(root.get(name), name=f"model_config.{name}")
    for field_name in path_fields:
        if field_name not in raw or not isinstance(raw[field_name], (str, Path)):
            raise TypeError(
                f"model_config.{name}.{field_name} must be a path string."
            )
        raw[field_name] = Path(raw[field_name])
    try:
        return config_type(**raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid model_config.{name}: {exc}") from exc


def deserialize_model_config(value: Any) -> MaskedDiffusionModelConfig:
    """Recreate the frozen nested config stored as JSON-compatible mappings."""

    raw = _require_mapping(value, name="model_config")
    expected = {
        "performer",
        "gene_identity",
        "gene_expression",
        "forward_process",
        "decoder",
        "loss",
        "architecture_version",
    }
    if set(raw) != expected:
        missing = sorted(expected - set(raw))
        unexpected = sorted(set(raw) - expected)
        raise ValueError(
            "model_config keys do not match the v2 contract; "
            f"missing={missing}, unexpected={unexpected}."
        )

    try:
        return MaskedDiffusionModelConfig(
            performer=_component(raw, "performer", PerformerConfig),
            gene_identity=_component(
                raw,
                "gene_identity",
                GeneIdentityEncoderConfig,
                path_fields=("weights_path", "manifest_path"),
            ),
            gene_expression=_component(
                raw,
                "gene_expression",
                GeneExpressionEncoderConfig,
            ),
            forward_process=_component(
                raw,
                "forward_process",
                ForwardProcessConfig,
            ),
            decoder=_component(raw, "decoder", DecoderConfig),
            loss=_component(raw, "loss", LossConfig),
            architecture_version=raw["architecture_version"],
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid model_config: {exc}") from exc


def _require_int(payload: Mapping[str, Any], name: str, *, minimum: int) -> int:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"Checkpoint {name} must be an integer >= {minimum}.")
    return value


def _validate_state_dict(value: Any) -> Mapping[str, Tensor]:
    if not isinstance(value, Mapping) or not value:
        raise TypeError("Checkpoint model must be a non-empty state-dict mapping.")
    for name, tensor in value.items():
        if not isinstance(name, str) or not name:
            raise TypeError("Every checkpoint model key must be a non-empty string.")
        if not isinstance(tensor, Tensor):
            raise TypeError(f"Checkpoint model entry {name!r} is not a Tensor.")
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"Checkpoint model tensor {name!r} is non-finite.")
    return value


def _build_model_from_state(
    config: MaskedDiffusionModelConfig,
    state_dict: Mapping[str, Tensor],
) -> MaskedDiffusionTrainingModule:
    """Build without reopening the original Geneformer initialization assets."""

    initial_weight = state_dict.get(_IDENTITY_WEIGHT_KEY)
    if initial_weight is None:
        raise KeyError(f"Checkpoint is missing {_IDENTITY_WEIGHT_KEY!r}.")
    expected_shape = (
        config.gene_identity.num_genes,
        config.gene_identity.source_dim,
    )
    if tuple(initial_weight.shape) != expected_shape:
        raise ValueError(
            f"{_IDENTITY_WEIGHT_KEY} must have shape {expected_shape}, got "
            f"{tuple(initial_weight.shape)}."
        )
    if initial_weight.dtype != torch.float32:
        raise TypeError(f"{_IDENTITY_WEIGHT_KEY} must have dtype torch.float32.")

    # GeneIdentityEncoder owns a clone.  This is intentional: after strict
    # load, the checkpoint payload (including mmap-backed storage) can be freed.
    identity_encoder = GeneIdentityEncoder(
        config.gene_identity,
        initial_weight.detach().cpu(),
    )
    denoiser = MaskedExpressionDenoiser(
        gene_identity_encoder=identity_encoder,
        gene_expression_encoder=GeneExpressionEncoder(config.gene_expression),
        absorbing_state_embedding=AbsorbingStateEmbedding(config.performer.d_model),
        backbone=build_performer_backbone(config.performer),
        decoder=GeneExpressionDecoder(config.decoder),
    )
    model = MaskedDiffusionTrainingModule(
        denoiser=denoiser,
        forward_process=AbsorbingMaskForwardProcess(config.forward_process),
        reconstruction_loss=TimeWeightedHurdleNLLLoss(config.loss),
    )
    model.load_state_dict(state_dict, strict=True)
    return model


def _torch_load_trusted(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except TypeError:
        # PyTorch versions predating one of these keyword arguments are still
        # allowed by requirements.txt.  The trust decision is unchanged.
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise TypeError("Checkpoint root must be a dictionary.")
    return value


def load_inference_checkpoint(
    checkpoint_path: Union[str, Path],
    *,
    device: Union[str, torch.device] = "cpu",
    trust_checkpoint: bool = False,
) -> LoadedInferenceModel:
    """Strictly load only model weights from a trusted v3 training checkpoint.

    ``trust_checkpoint=True`` is mandatory because the current training
    container needs unrestricted pickle deserialization.  It must only be used
    for a checkpoint created by this project and obtained from a trusted path.
    Optimizer, scheduler, scaler and RNG objects are never restored.
    """

    if not isinstance(trust_checkpoint, bool):
        raise TypeError("trust_checkpoint must be a boolean.")
    if not trust_checkpoint:
        raise ValueError(
            "Refusing to deserialize an untrusted training checkpoint. Pass "
            "trust_checkpoint=True only for a checkpoint you trust."
        )
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint is not a file: {path}")
    target_device = torch.device(device)
    identity_before = path.stat()
    checkpoint_sha256 = sha256_file(path)
    payload = _torch_load_trusted(path)
    identity_after = path.stat()
    identity_fields_before = (
        identity_before.st_dev,
        identity_before.st_ino,
        identity_before.st_size,
        identity_before.st_mtime_ns,
        identity_before.st_ctime_ns,
    )
    identity_fields_after = (
        identity_after.st_dev,
        identity_after.st_ino,
        identity_after.st_size,
        identity_after.st_mtime_ns,
        identity_after.st_ctime_ns,
    )
    if identity_fields_after != identity_fields_before:
        raise RuntimeError(
            "Checkpoint changed while it was being hashed/loaded; retry after "
            "the writer has finished."
        )

    version = payload.get("checkpoint_format_version")
    if version != SUPPORTED_CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            "Unsupported checkpoint format version: "
            f"expected {SUPPORTED_CHECKPOINT_FORMAT_VERSION}, got {version!r}."
        )
    architecture = payload.get("architecture_version")
    if architecture != ARCHITECTURE_VERSION:
        raise ValueError(
            f"Checkpoint architecture {architecture!r} is incompatible with "
            f"{ARCHITECTURE_VERSION!r}."
        )
    config = deserialize_model_config(payload.get("model_config"))
    if config.architecture_version != architecture:
        raise ValueError("Checkpoint top-level and model-config architectures differ.")

    data_contract = _require_mapping(
        payload.get("data_contract"),
        name="data_contract",
    )
    if data_contract.get("n_vars") != NUM_GENES:
        raise ValueError(f"Checkpoint data contract must contain {NUM_GENES} genes.")
    gene_hash = data_contract.get("gene_order_sha256")
    if (
        not isinstance(gene_hash, str)
        or re.fullmatch(r"[0-9a-fA-F]{64}", gene_hash) is None
    ):
        raise ValueError("Checkpoint gene_order_sha256 is invalid.")

    state_dict = _validate_state_dict(payload.get("model"))
    model = _build_model_from_state(config, state_dict)
    model.to(target_device)
    model.eval()

    epoch_completed = payload.get("epoch_completed")
    if not isinstance(epoch_completed, bool):
        raise ValueError("Checkpoint epoch_completed must be a boolean.")
    best_metric = payload.get("best_primary_validation_metric")
    if (
        isinstance(best_metric, bool)
        or not isinstance(best_metric, (int, float))
        or not math.isfinite(float(best_metric))
    ):
        raise ValueError("Checkpoint best primary validation metric is invalid.")
    primary_metric = payload.get("primary_validation_metric")
    if primary_metric != PRIMARY_VALIDATION_METRIC:
        raise ValueError(
            "Checkpoint primary_validation_metric is incompatible: "
            f"expected {PRIMARY_VALIDATION_METRIC!r}, got {primary_metric!r}."
        )
    reason = payload.get("reason")
    if not isinstance(reason, str) or not reason:
        raise ValueError("Checkpoint reason is invalid.")

    current_epoch = _require_int(payload, "current_epoch", minimum=-1)
    next_epoch = _require_int(payload, "next_epoch", minimum=0)
    expected_next_epoch = current_epoch + 1 if epoch_completed else current_epoch
    if next_epoch != expected_next_epoch:
        raise ValueError(
            "Checkpoint epoch counters are inconsistent with epoch_completed."
        )

    metadata = InferenceCheckpointMetadata(
        checkpoint_path=path,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_format_version=version,
        architecture_version=architecture,
        reason=reason,
        current_epoch=current_epoch,
        epoch_completed=epoch_completed,
        next_epoch=next_epoch,
        global_step=_require_int(payload, "global_step", minimum=0),
        primary_validation_metric=primary_metric,
        best_primary_validation_metric=float(best_metric),
        model_config=config,
        data_contract=data_contract,
    )
    return LoadedInferenceModel(model=model, metadata=metadata)


__all__ = [
    "InferenceCheckpointMetadata",
    "LoadedInferenceModel",
    "SUPPORTED_CHECKPOINT_FORMAT_VERSION",
    "deserialize_model_config",
    "load_inference_checkpoint",
]
