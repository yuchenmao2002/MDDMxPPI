"""Small CPU tests for the unconditional generation entry point."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pytest
from scipy import sparse
import torch

import scripts.sample_masked_diffusion as cli
from src.models.config import MaskedDiffusionModelConfig, PerformerConfig
from src.utils.inference_checkpoint import InferenceCheckpointMetadata


class _FakeSampler:
    def __init__(self, num_genes: int) -> None:
        self.num_genes = num_genes
        self.batch_sizes: list[int] = []

    def sample(self, batch_size: int, *, generator, **_kwargs):
        self.batch_sizes.append(batch_size)
        values = torch.rand(
            (batch_size, self.num_genes, 1),
            generator=generator,
            dtype=torch.float32,
        )
        values[values < 0.5] = 0.0
        return SimpleNamespace(expression_values=values)


def _metadata(tmp_path: Path) -> InferenceCheckpointMetadata:
    model_config = MaskedDiffusionModelConfig(
        backbone=PerformerConfig(num_layers=1),
        backbone_variant="performer",
    )
    return InferenceCheckpointMetadata(
        checkpoint_path=tmp_path / "best.pt",
        checkpoint_sha256="c" * 64,
        checkpoint_format_version=4,
        architecture_version=model_config.architecture_version,
        reason="epoch_end",
        current_epoch=2,
        epoch_completed=True,
        next_epoch=3,
        global_step=123,
        primary_validation_metric="val_time_weighted_hurdle_nll",
        best_primary_validation_metric=0.25,
        model_config=model_config,
        data_contract={"n_vars": 3, "gene_order_sha256": "d" * 64},
        ppi_asset_contract=None,
    )


def test_validate_args_requires_positive_values_and_explicit_trust() -> None:
    args = argparse.Namespace(
        num_cells=2,
        num_steps=3,
        batch_size=1,
        seed=7,
        trust_checkpoint=False,
        gene_index_column="Index",
        gene_name_column="Symbol",
    )
    with pytest.raises(ValueError, match="trust-checkpoint"):
        cli.validate_args(args)
    args.trust_checkpoint = True
    cli.validate_args(args)
    args.num_steps = 0
    with pytest.raises(ValueError, match="num-steps"):
        cli.validate_args(args)


def test_gene_vocabulary_is_ordered_and_hash_is_stable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "NUM_GENES", 3)
    mapping = tmp_path / "mapping.csv"
    mapping.write_text("Symbol,Index\nA,0\nB,1\nC,2\n", encoding="utf-8")
    genes = cli.read_gene_vocabulary(
        mapping,
        index_column="Index",
        name_column="Symbol",
    )
    assert genes == ["A", "B", "C"]
    assert cli.sha256_strings(genes) == cli.sha256_strings(["A", "B", "C"])

    mapping.write_text("Symbol,Index\nA,0\nB,2\nC,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ordered"):
        cli.read_gene_vocabulary(
            mapping,
            index_column="Index",
            name_column="Symbol",
        )


def test_batch_generation_is_reproducible_and_covers_ragged_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "NUM_GENES", 3)
    first_sampler = _FakeSampler(3)
    second_sampler = _FakeSampler(3)
    first = cli.generate_expression_matrix(
        sampler=first_sampler,
        num_cells=5,
        batch_size=2,
        device=torch.device("cpu"),
        precision="fp32",
        seed=11,
    )
    second = cli.generate_expression_matrix(
        sampler=second_sampler,
        num_cells=5,
        batch_size=2,
        device=torch.device("cpu"),
        precision="fp32",
        seed=11,
    )

    assert first_sampler.batch_sizes == [2, 2, 1]
    assert first.shape == (5, 3)
    assert first.dtype == np.float32
    assert sparse.isspmatrix_csr(first)
    assert np.array_equal(first.toarray(), second.toarray())


def test_h5ad_output_preserves_sparse_values_gene_order_and_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "NUM_GENES", 3)
    matrix = sparse.csr_matrix(
        np.asarray([[0.0, 1.0, 2.0], [3.0, 0.0, 4.0]], dtype=np.float32)
    )
    generated = cli.build_anndata(
        matrix,
        genes=["A", "B", "C"],
        metadata=_metadata(tmp_path),
        num_steps=2,
        schedule="linear",
        seed=9,
        batch_size=2,
        precision="fp32",
        device=torch.device("cpu"),
        mapping_path=tmp_path / "mapping.csv",
        mapping_sha256="e" * 64,
    )
    output = tmp_path / "generated.h5ad"
    cli.atomic_write_h5ad(
        generated,
        output,
        compression=None,
        overwrite=False,
    )

    restored = ad.read_h5ad(output)
    assert sparse.isspmatrix_csr(restored.X)
    assert restored.X.dtype == np.float32
    assert restored.var_names.tolist() == ["A", "B", "C"]
    assert restored.obs_names.tolist() == [
        "generated_cell_00000000",
        "generated_cell_00000001",
    ]
    assert restored.uns["ppil_generation"]["num_steps"] == 2
    assert restored.uns["ppil_generation"]["gene_mapping_sha256"] == "e" * 64
    # The generated file records which of the four backbones produced it.
    assert restored.uns["ppil_generation"]["backbone_variant"] == "performer"
    assert restored.uns["ppil_generation"]["backbone_signature"] == "performer*1"
    assert restored.uns["ppil_generation"]["architecture_version"].startswith(
        "performer*1|"
    )
    assert "not raw integer counts" in restored.uns["ppil_generation"][
        "expression_domain"
    ]
    with pytest.raises(FileExistsError):
        cli.atomic_write_h5ad(
            generated,
            output,
            compression=None,
            overwrite=False,
        )
