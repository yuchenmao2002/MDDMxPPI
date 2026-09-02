#!/usr/bin/env python3
"""第二阶段：在 stage 2 训练网络上按秩计算宏平均 pairwise concordance A(r)。"""

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch

from spherical_ppi_embedding import (
    NUM_GENES,
    build_spherical_ppi_embedding,
    load_ppi_edges,
)


# 第一阶段选定的最优候选；本阶段只在它上面扫秩。
BEST_TAU = 700
BEST_K_DIFF = 4
BEST_B = 1
RANK_VALUES: Tuple[int, ...] = (8, 16, 24, 32, 48, 64, 80, 96, 128)
OBSERVED_SCORE_THRESHOLD = 700
SEED = 0
SIMILARITY_BLOCK = 512

DEFAULT_TRAIN_PATH = Path(
    "/home/svu/e1538713/CodeNo0/data/evaluation/PPI_train_stage2.csv"
)
DEFAULT_VALIDATION_PATH = Path(
    "/home/svu/e1538713/CodeNo0/data/evaluation/PPI_validation_rank.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/svu/e1538713/CodeNo0/outputs/evaluation/PPI_embedding"
)
OUTPUT_NAME = "PPI_rank_concordance.csv"
PROVENANCE_NAME = "PPI_rank_concordance.json"

OUTPUT_COLUMNS = [
    "r",
    "A",
    "n_validation_genes",
    "mean_validation_size",
    "mean_background_size",
    "n_free_validation_genes",
    "sigma_1",
    "elapsed_seconds",
]


@dataclass(frozen=True)
class RankEvaluationContext:
    """所有秩共用的 V_val^r、V_i,val^r 与 V_i,bg^r 的排除项。"""

    validation_genes: Tuple[int, ...]
    validation_neighbors: Dict[int, torch.Tensor]
    excluded_neighbors: Dict[int, torch.Tensor]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_rank_evaluation_context(
    train_index1: torch.Tensor,
    train_index2: torch.Tensor,
    train_score: torch.Tensor,
    validation_path: Path = DEFAULT_VALIDATION_PATH,
) -> RankEvaluationContext:
    """由 E_val^r 与 E_700 \\ E_val^r 构建固定的节点级评价集合。"""
    val_index1, val_index2, _ = load_ppi_edges(validation_path)

    # 一次扫描 r 验证边并同时登记两个方向，得到 V_i,val^r 与 V_val^r。
    validation_sets: Dict[int, set] = {}
    for first, second in zip(val_index1.tolist(), val_index2.tolist()):
        validation_sets.setdefault(first, set()).add(second)
        validation_sets.setdefault(second, set()).add(first)
    validation_genes = tuple(sorted(validation_sets))

    # V_i,eval^r 排除 E_700 \ E_val^r 的邻居。stage 2 训练网络本身就是
    # 原始网络删去 E_val^r 的结果，因此它里面 score >= 700 的边恰好就是这一集合。
    excluded_sets = {gene: set() for gene in validation_genes}
    visible = train_score >= OBSERVED_SCORE_THRESHOLD
    for first, second in zip(
        train_index1[visible].tolist(), train_index2[visible].tolist()
    ):
        if first in excluded_sets:
            excluded_sets[first].add(second)
        if second in excluded_sets:
            excluded_sets[second].add(first)

    for gene in validation_genes:
        if validation_sets[gene] & excluded_sets[gene]:
            raise RuntimeError("r 验证边仍存在于 V_i,eval^r 的排除邻居中")

    return RankEvaluationContext(
        validation_genes=validation_genes,
        validation_neighbors={
            gene: torch.tensor(sorted(validation_sets[gene]), dtype=torch.int64)
            for gene in validation_genes
        },
        excluded_neighbors={
            gene: torch.tensor(sorted(excluded_sets[gene]), dtype=torch.int64)
            for gene in validation_genes
        },
    )


def concordance(
    embedding: torch.Tensor,
    context: RankEvaluationContext,
    num_genes: int = NUM_GENES,
) -> Tuple[float, float]:
    """计算宏平均 pairwise concordance A(r) 及平均背景集合大小。"""
    genes = context.validation_genes
    gene_scores: List[float] = []
    background_sizes: List[int] = []
    blocked = torch.zeros(num_genes, dtype=torch.bool)

    for start in range(0, len(genes), SIMILARITY_BLOCK):
        block = genes[start : start + SIMILARITY_BLOCK]
        rows = torch.tensor(block, dtype=torch.int64)
        # s_ij = z_i^T z_j；按块计算以避免一次性展开 |V_val^r| x |V| 的相似度矩阵。
        similarity = embedding[rows] @ embedding.T

        for offset, gene in enumerate(block):
            validation = context.validation_neighbors[gene]
            excluded = context.excluded_neighbors[gene]
            blocked[gene] = True
            blocked[excluded] = True
            blocked[validation] = True
            background = similarity[offset][~blocked]
            blocked[gene] = False
            blocked[excluded] = False
            blocked[validation] = False

            # 对每个验证伙伴统计背景中严格更小的个数，并把并列各计一半。
            ordered = torch.sort(background).values
            targets = similarity[offset][validation]
            strictly_less = torch.searchsorted(ordered, targets, right=False)
            not_greater = torch.searchsorted(ordered, targets, right=True)
            concordant = (
                strictly_less.to(torch.float64).sum()
                + 0.5 * (not_greater - strictly_less).to(torch.float64).sum()
            )
            gene_scores.append(
                float(concordant) / (validation.numel() * ordered.numel())
            )
            background_sizes.append(int(ordered.numel()))

    return (
        sum(gene_scores) / len(gene_scores),
        sum(background_sizes) / len(background_sizes),
    )


def evaluate_all_ranks(
    train_path: Path = DEFAULT_TRAIN_PATH,
    validation_path: Path = DEFAULT_VALIDATION_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    device: str = "cpu",
) -> Tuple[Path, Path]:
    """对每个候选秩执行完整嵌入算法并计算 A(r)。"""
    index1, index2, combined_score = load_ppi_edges(train_path)
    context = build_rank_evaluation_context(
        index1, index2, combined_score, validation_path
    )
    validation_sizes = [
        context.validation_neighbors[gene].numel()
        for gene in context.validation_genes
    ]
    print(
        f"|V_val^r|={len(context.validation_genes)} "
        f"mean|V_i,val^r|={sum(validation_sizes) / len(validation_sizes):.3f}",
        flush=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OUTPUT_NAME
    partial_path = output_dir / f".{OUTPUT_NAME}.partial"
    records: List[Dict] = []

    for rank in RANK_VALUES:
        started = time.perf_counter()
        tensors = build_spherical_ppi_embedding(
            index1=index1,
            index2=index2,
            combined_score=combined_score,
            tau=float(BEST_TAU),
            k_diff=BEST_K_DIFF,
            b=float(BEST_B),
            rank=rank,
            device=device,
            seed=SEED,
        )
        embedding = tensors["embedding"]
        free_mask = tensors["free_mask"]
        area, mean_background = concordance(embedding, context)
        elapsed = time.perf_counter() - started

        free_validation = int(
            free_mask[torch.tensor(context.validation_genes, dtype=torch.int64)]
            .sum()
            .item()
        )
        records.append(
            {
                "r": rank,
                "A": area,
                "n_validation_genes": len(context.validation_genes),
                "mean_validation_size": sum(validation_sizes) / len(validation_sizes),
                "mean_background_size": mean_background,
                "n_free_validation_genes": free_validation,
                "sigma_1": float(tensors["eigenvalues"][0]),
                "elapsed_seconds": elapsed,
            }
        )
        print(
            f"r={rank:3d} A={area:.9f} free_val_genes={free_validation} "
            f"sigma_1={float(tensors['eigenvalues'][0]):.3f} "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )
        del tensors, embedding, free_mask

        pd.DataFrame(records, columns=OUTPUT_COLUMNS).to_csv(
            partial_path, index=False, float_format="%.17g"
        )

    partial_path.replace(output_path)

    provenance_path = output_dir / PROVENANCE_NAME
    provenance = {
        "schema_version": "ppi_embedding_rank_concordance.v1",
        "artifact_type": "ppi_embedding_rank_concordance",
        "source": {
            "train_path": str(train_path),
            "train_sha256": _sha256_file(train_path),
            "validation_path": str(validation_path),
            "validation_sha256": _sha256_file(validation_path),
            "observed_score_threshold": OBSERVED_SCORE_THRESHOLD,
        },
        "candidate": {"tau": BEST_TAU, "K_diff": BEST_K_DIFF, "b": BEST_B},
        "ranks": list(RANK_VALUES),
        "num_genes": NUM_GENES,
        "seed": SEED,
        "device": device,
        "definitions": {
            "s_ij": "z_i^T z_j on the unit sphere",
            "V_i_val": "{j : {i,j} in E_val^r}",
            "V_i_eval": "V \\ ({i} u {j : {i,j} in E_700 \\ E_val^r})",
            "V_i_bg": "V_i_eval \\ V_i_val",
            "A": (
                "macro average over i in V_val^r of "
                "mean_{j,u} [1(s_ij > s_iu) + 0.5 * 1(s_ij == s_iu)]"
            ),
        },
        "outputs": {
            "concordance_csv": output_path.name,
            "concordance_sha256": _sha256_file(output_path),
        },
        "torch_version": torch.__version__,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with provenance_path.open("w", encoding="utf-8") as handle:
        json.dump(provenance, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return output_path, provenance_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按秩计算最优候选 (700, 4, 1) 的宏平均 pairwise concordance"
    )
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument(
        "--validation-path", type=Path, default=DEFAULT_VALIDATION_PATH
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cpu", help="cpu（默认）、auto 或 CUDA 设备")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    for path in evaluate_all_ranks(
        train_path=arguments.train_path,
        validation_path=arguments.validation_path,
        output_dir=arguments.output_dir,
        device=arguments.device,
    ):
        print(f"wrote {path}")
