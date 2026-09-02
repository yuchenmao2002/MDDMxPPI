#!/usr/bin/env python3
"""计算球面 PPI 嵌入第一阶段的全部超参数评价得分。"""

import argparse
import math
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple

import pandas as pd
import torch

from spherical_ppi_embedding import (
    ALPHA,
    NUM_GENES,
    _normalized_adjacency,
    _resolve_device,
    _shifted_ppmi,
    load_ppi_edges,
)


# 第一阶段搜索空间：共 6 × 8 × 3 = 144 个候选组合。
TAU_VALUES: Tuple[int, ...] = (400, 500, 600, 700, 800, 900)
K_DIFF_VALUES: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)
B_VALUES: Tuple[int, ...] = (1, 2, 5)
DEFAULT_TRAIN_PPI_PATH = Path(
    "/home/svu/e1538713/CodeNo0/data/evaluation/PPI_train_stage1.csv"
)
DEFAULT_THETA_VALIDATION_PATH = Path(
    "/home/svu/e1538713/CodeNo0/data/evaluation/PPI_validation_theta.csv"
)
DEFAULT_RANK_VALIDATION_PATH = Path(
    "/home/svu/e1538713/CodeNo0/data/evaluation/PPI_validation_rank.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/svu/e1538713/CodeNo0/outputs/evaluation/PPI_embedding"
)
DEFAULT_OUTPUT_NAME = "PPI_hyperparameter_J.csv"
OBSERVED_SCORE_THRESHOLD = 700
TOTAL_CANDIDATES = len(TAU_VALUES) * len(K_DIFF_VALUES) * len(B_VALUES)

SEARCH_SPACE = {
    "tau": TAU_VALUES,
    "k_diff": K_DIFF_VALUES,
    "b": B_VALUES,
}


@dataclass(frozen=True)
class EvaluationContext:
    """所有候选共用的 V_val^θ，以及每个基因的 V_i,val^θ 与 V_i,eval^θ。"""

    validation_genes: Tuple[int, ...]
    validation_neighbors: Dict[int, torch.Tensor]
    excluded_neighbors: Dict[int, torch.Tensor]
    candidate_sizes: Dict[int, int]


def iter_candidates() -> Iterator[Dict[str, int]]:
    """按固定顺序遍历搜索空间的全部候选组合。"""
    for tau, k_diff, b in product(TAU_VALUES, K_DIFF_VALUES, B_VALUES):
        yield {"tau": tau, "k_diff": k_diff, "b": b}


def _without_diagonal(
    matrix: torch.Tensor, scale: float = 1.0
) -> torch.Tensor:
    """仅在完整扩散累加后缩放矩阵并清除对角线。"""
    indices = matrix.indices()
    off_diagonal = indices[0] != indices[1]
    return torch.sparse_coo_tensor(
        indices[:, off_diagonal],
        matrix.values()[off_diagonal] * scale,
        matrix.shape,
        device=matrix.device,
        check_invariants=False,
    ).coalesce()


def _iter_diffused_matrices(
    normalized_adjacency: torch.Tensor,
    k_values: Sequence[int],
) -> Iterator[Tuple[int, torch.Tensor]]:
    """递推复用矩阵幂，依次生成各扩散阶数对应的 M。"""
    requested = set(k_values)
    power = normalized_adjacency
    weighted_sum = (power * (1.0 - ALPHA)).coalesce()

    for k_diff in range(1, max(k_values) + 1):
        if k_diff > 1:
            power = torch.sparse.mm(power, normalized_adjacency).coalesce()
            coefficient = ALPHA ** (k_diff - 1) * (1.0 - ALPHA)
            weighted_sum = (weighted_sum + power * coefficient).coalesce()

        if k_diff in requested:
            diffused = _without_diagonal(
                weighted_sum, 1.0 / (1.0 - ALPHA**k_diff)
            )
            yield k_diff, diffused
            del diffused


def iter_candidate_ppmi(
    index1: torch.Tensor,
    index2: torch.Tensor,
    combined_score: torch.Tensor,
    num_genes: int = NUM_GENES,
    device: str = "cpu",
    tau_values: Sequence[int] = TAU_VALUES,
    k_values: Sequence[int] = K_DIFF_VALUES,
    b_values: Sequence[int] = B_VALUES,
) -> Iterator[Tuple[Dict[str, int], torch.Tensor]]:
    """逐个生成候选超参数及其 PPMI 矩阵 P，不执行后续嵌入。"""
    compute_device = _resolve_device(device)

    for tau in tau_values:
        # 步骤 1–2：构建 S，并完成不含额外超参数的对称度归一化。
        normalized_adjacency = _normalized_adjacency(
            index1,
            index2,
            combined_score,
            tau,
            num_genes,
            compute_device,
        )

        # 步骤 3–4：复用扩散幂，并对每个 b 生成 shifted PPMI 矩阵。
        for k_diff, diffused in _iter_diffused_matrices(
            normalized_adjacency, k_values
        ):
            for b in b_values:
                candidate = {"tau": tau, "k_diff": k_diff, "b": b}
                yield candidate, _shifted_ppmi(diffused, b)
            del diffused
        del normalized_adjacency


def load_training_edges(
    ppi_path: Path = DEFAULT_TRAIN_PPI_PATH,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """加载所有候选统一使用的删边后训练网络。"""
    return load_ppi_edges(ppi_path)


def build_evaluation_context(
    train_index1: torch.Tensor,
    train_index2: torch.Tensor,
    train_score: torch.Tensor,
    theta_validation_path: Path = DEFAULT_THETA_VALIDATION_PATH,
    rank_validation_path: Path = DEFAULT_RANK_VALIDATION_PATH,
    num_genes: int = NUM_GENES,
) -> EvaluationContext:
    """由 E_val^θ 与 E_700 \\ E_val^θ 构建固定的节点级评价集合。"""
    val_index1, val_index2, _ = load_ppi_edges(theta_validation_path)

    # 一次扫描 θ 验证边并同时登记两个方向，得到 V_i,val^θ 与 V_val^θ。
    validation_sets: Dict[int, set] = {}
    for first, second in zip(val_index1.tolist(), val_index2.tolist()):
        validation_sets.setdefault(first, set()).add(second)
        validation_sets.setdefault(second, set()).add(first)
    validation_genes = tuple(sorted(validation_sets))

    # V_i,eval^θ 排除的是 E_700 \\ E_val^θ 的邻居：stage 1 训练网络里可见的高置信边，
    # 再加上 E_val^r——后者虽然不在 stage 1 的训练网络里，但已知是真实高置信边，
    # 不能当作候选负样本。两个方向同时登记。
    rank_index1, rank_index2, _ = load_ppi_edges(rank_validation_path)
    visible = train_score >= OBSERVED_SCORE_THRESHOLD
    excluded_index1 = torch.cat((train_index1[visible], rank_index1))
    excluded_index2 = torch.cat((train_index2[visible], rank_index2))

    excluded_sets = {gene: set() for gene in validation_genes}
    for first, second in zip(excluded_index1.tolist(), excluded_index2.tolist()):
        if first in excluded_sets:
            excluded_sets[first].add(second)
        if second in excluded_sets:
            excluded_sets[second].add(first)

    validation_neighbors = {
        gene: torch.tensor(sorted(validation_sets[gene]), dtype=torch.int64)
        for gene in validation_genes
    }
    excluded_neighbors = {
        gene: torch.tensor(sorted(excluded_sets[gene]), dtype=torch.int64)
        for gene in validation_genes
    }
    candidate_sizes = {
        gene: num_genes - 1 - len(excluded_sets[gene])
        for gene in validation_genes
    }

    for gene in validation_genes:
        if validation_sets[gene] & excluded_sets[gene]:
            raise RuntimeError("θ 验证边仍存在于 V_i,eval^θ 的排除邻居中")

    return EvaluationContext(
        validation_genes=validation_genes,
        validation_neighbors=validation_neighbors,
        excluded_neighbors=excluded_neighbors,
        candidate_sizes=candidate_sizes,
    )


def _logaddexp(first: float, second: float) -> float:
    """稳定计算两个对数标量对应指数之和的对数。"""
    maximum = max(first, second)
    return maximum + math.log(
        math.exp(first - maximum) + math.exp(second - maximum)
    )


def score_ppmi(
    ppmi: torch.Tensor,
    context: EvaluationContext,
    num_genes: int = NUM_GENES,
) -> float:
    """精确计算一个 PPMI 候选在 V_val^θ 上的宏平均 J。"""
    ppmi = ppmi.coalesce()
    row, col = ppmi.indices()
    values = ppmi.values()
    row_counts = torch.bincount(row, minlength=num_genes)
    row_ends = row_counts.cumsum(0)
    excluded = torch.zeros(num_genes, dtype=torch.bool)
    gene_scores: List[float] = []

    for gene in context.validation_genes:
        start = 0 if gene == 0 else int(row_ends[gene - 1])
        end = int(row_ends[gene])
        row_col = col[start:end]
        row_values = values[start:end]

        # V_i,eval^θ 从全部节点中删除 i 和排除邻居；其余隐式零均贡献 exp(0)=1。
        neighbors = context.excluded_neighbors[gene]
        excluded[gene] = True
        excluded[neighbors] = True
        evaluation_values = row_values[~excluded[row_col]].to(torch.float64)
        excluded[gene] = False
        excluded[neighbors] = False

        candidate_size = context.candidate_sizes[gene]
        zero_count = candidate_size - evaluation_values.numel()
        if evaluation_values.numel() == 0:
            log_denominator = math.log(candidate_size)
        else:
            positive_logsum = float(torch.logsumexp(evaluation_values, dim=0))
            log_denominator = positive_logsum
            if zero_count > 0:
                log_denominator = _logaddexp(
                    math.log(zero_count), positive_logsum
                )

        # 验证边未出现在稀疏 P 中时，其 P_ij 按零计入验证均值。
        targets = context.validation_neighbors[gene]
        positions = torch.searchsorted(row_col, targets)
        in_range = positions < row_col.numel()
        positions = positions[in_range]
        targets = targets[in_range]
        matched = row_col[positions] == targets
        validation_sum = float(
            row_values[positions[matched]].sum(dtype=torch.float64)
        )

        gene_scores.append(
            validation_sum / context.validation_neighbors[gene].numel()
            + math.log(candidate_size)
            - log_denominator
        )

    return math.fsum(gene_scores) / len(gene_scores)


def compute_all_candidate_scores(
    train_path: Path = DEFAULT_TRAIN_PPI_PATH,
    theta_validation_path: Path = DEFAULT_THETA_VALIDATION_PATH,
    rank_validation_path: Path = DEFAULT_RANK_VALIDATION_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """在 CPU 上顺序计算全部候选的 J，并保存供独立报告脚本读取的 CSV。"""
    index1, index2, combined_score = load_training_edges(train_path)
    context = build_evaluation_context(
        index1, index2, combined_score, theta_validation_path, rank_validation_path
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / DEFAULT_OUTPUT_NAME
    partial_path = output_dir / f".{DEFAULT_OUTPUT_NAME}.partial"
    records = []
    completed = 0

    # 先计算较稀疏的高阈值候选；最终 CSV 仍按注册搜索空间排序。
    for tau in reversed(TAU_VALUES):
        iterator = iter_candidate_ppmi(
            index1,
            index2,
            combined_score,
            device="cpu",
            tau_values=(tau,),
        )
        while True:
            try:
                candidate, ppmi = next(iterator)
            except StopIteration:
                break
            started = time.perf_counter()
            score = score_ppmi(ppmi, context)
            elapsed = time.perf_counter() - started
            records.append(
                {
                    "tau": candidate["tau"],
                    "K_diff": candidate["k_diff"],
                    "b": candidate["b"],
                    "J": score,
                }
            )
            completed += 1
            print(
                f"[{completed:03d}/{TOTAL_CANDIDATES}] tau={candidate['tau']} "
                f"K_diff={candidate['k_diff']} b={candidate['b']} "
                f"J={score:.9f} score_time={elapsed:.2f}s nnz(P)={ppmi._nnz()}",
                flush=True,
            )
            del ppmi

            pd.DataFrame(records).sort_values(
                ["tau", "K_diff", "b"]
            ).to_csv(partial_path, index=False)

    partial_path.replace(output_path)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="计算第一阶段全部球面 PPI 超参数候选的 J"
    )
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PPI_PATH)
    parser.add_argument(
        "--theta-validation-path", type=Path, default=DEFAULT_THETA_VALIDATION_PATH
    )
    parser.add_argument(
        "--rank-validation-path", type=Path, default=DEFAULT_RANK_VALIDATION_PATH
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    result_path = compute_all_candidate_scores(
        train_path=arguments.train_path,
        theta_validation_path=arguments.theta_validation_path,
        rank_validation_path=arguments.rank_validation_path,
        output_dir=arguments.output_dir,
    )
    print(f"Saved {TOTAL_CANDIDATES} candidate scores to: {result_path}")
