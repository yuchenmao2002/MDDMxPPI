#!/usr/bin/env python3
"""从完整 PPI 网络构建冻结的正负边评估集。"""

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


DEFAULT_PPI_PATH = Path(
    "/home/svu/e1538713/CodeNo0/data/processed/STRING/PPI_index.csv"
)
DEFAULT_OUTPUT_DIR = Path("/home/svu/e1538713/CodeNo0/data/processed/PPI")
OUTPUT_NAME = "PPI_evaluation_set.csv"
NUM_GENES = 19_295
NUM_BINS = 10
CANDIDATE_SCORE = 700
MIN_DEGREE = 2
HOLDOUT_FRACTION = 0.1
RANDOM_SEED = 0


def load_ppi_edges(ppi_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """只读取构建评估集所需的三列。"""
    frame = pd.read_csv(
        ppi_path,
        usecols=["Index1", "Index2", "combined_score"],
        dtype={"Index1": "int32", "Index2": "int32", "combined_score": "int32"},
    )
    return tuple(
        frame[column].to_numpy(copy=True)
        for column in ("Index1", "Index2", "combined_score")
    )


def build_undirected_adjacency(
    index1: np.ndarray, index2: np.ndarray, num_genes: int
) -> csr_matrix:
    """由 CSV 中的半边恢复无向、无权、零对角邻接矩阵。"""
    rows = np.concatenate((index1, index2))
    cols = np.concatenate((index2, index1))
    adjacency = csr_matrix(
        (np.ones(rows.size, dtype=np.uint8), (rows, cols)),
        shape=(num_genes, num_genes),
    )
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    adjacency.data.fill(1)
    adjacency.sort_indices()
    return adjacency


def equal_frequency_degree_bins(degree: np.ndarray) -> np.ndarray:
    """按 (degree, gene_id) 稳定排序，将全部基因等频分入 10 箱。"""
    gene_ids = np.arange(degree.size, dtype=np.int32)
    order = np.lexsort((gene_ids, degree))
    degree_bin = np.empty(degree.size, dtype=np.int8)
    for bin_id, genes in enumerate(np.array_split(order, NUM_BINS), start=1):
        degree_bin[genes] = bin_id
    return degree_bin


def _undirected_edge_key(i: int, j: int, num_genes: int) -> int:
    return min(i, j) * num_genes + max(i, j)


def sample_degree_matched_negatives(
    positive_edges: np.ndarray,
    adjacency: csr_matrix,
    degree_bin: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """固定每条正边的 Index1，从其 Index2 的同箱非邻居中无放回抽样。"""
    groups: DefaultDict[int, DefaultDict[int, List[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for pair_id, (fixed_i, matched_j) in enumerate(positive_edges):
        groups[int(fixed_i)][int(degree_bin[matched_j])].append(pair_id)

    bin_members = [
        np.flatnonzero(degree_bin == bin_id).astype(np.int32, copy=False)
        for bin_id in range(1, NUM_BINS + 1)
    ]
    negative_edges = np.empty_like(positive_edges)
    selected_keys = set()
    excluded = np.zeros(adjacency.shape[0], dtype=bool)

    for fixed_i, positions_by_bin in groups.items():
        neighbors = adjacency.indices[
            adjacency.indptr[fixed_i] : adjacency.indptr[fixed_i + 1]
        ]
        excluded[neighbors] = True
        excluded[fixed_i] = True

        for bin_id, positions in positions_by_bin.items():
            members = bin_members[bin_id - 1]
            eligible = members[~excluded[members]]
            chosen = []
            for candidate_j in rng.permutation(eligible):
                key = _undirected_edge_key(fixed_i, int(candidate_j), adjacency.shape[0])
                if key not in selected_keys:
                    selected_keys.add(key)
                    chosen.append(int(candidate_j))
                    if len(chosen) == len(positions):
                        break
            if len(chosen) != len(positions):
                raise RuntimeError(
                    f"基因 {fixed_i} 在度箱 {bin_id} 中没有足够的唯一非邻居"
                )

            negative_edges[positions, 0] = fixed_i
            negative_edges[positions, 1] = chosen

        excluded[neighbors] = False
        excluded[fixed_i] = False

    return negative_edges


def build_evaluation_set(
    index1: np.ndarray,
    index2: np.ndarray,
    combined_score: np.ndarray,
    seed: int = RANDOM_SEED,
    num_genes: int = NUM_GENES,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """构建一一配对的冻结正负边评估集。"""
    rng = np.random.default_rng(seed)

    # 1. 在完整无向图上计算每个基因的无权度。
    adjacency = build_undirected_adjacency(index1, index2, num_genes)
    degree = np.diff(adjacency.indptr)

    # 2. 按无权度把全部基因等频分入 10 个箱。
    degree_bin = equal_frequency_degree_bins(degree)

    # 3. 仅保留置信度不低于 700 且两端无权度均不低于 2 的候选边。
    candidate_mask = (
        (combined_score >= CANDIDATE_SCORE)
        & (degree[index1] >= MIN_DEGREE)
        & (degree[index2] >= MIN_DEGREE)
    )
    candidate_rows = np.flatnonzero(candidate_mask)

    # 4. 从候选池中均匀、无放回抽取向下取整的 10% 作为正样本。
    n_positive = int(candidate_rows.size * HOLDOUT_FRACTION)
    selected_rows = rng.choice(candidate_rows, size=n_positive, replace=False)
    positive_edges = np.column_stack((index1[selected_rows], index2[selected_rows]))

    # 5. 保持正边的 Index1 端点，并匹配 Index2 所在的度箱生成负样本。
    negative_edges = sample_degree_matched_negatives(
        positive_edges, adjacency, degree_bin, rng
    )

    positive_keys = np.fromiter(
        (
            _undirected_edge_key(int(i), int(j), num_genes)
            for i, j in positive_edges
        ),
        dtype=np.int64,
        count=n_positive,
    )
    negative_keys = np.fromiter(
        (
            _undirected_edge_key(int(i), int(j), num_genes)
            for i, j in negative_edges
        ),
        dtype=np.int64,
        count=n_positive,
    )

    # 按无向边语义执行用户指定的集合断言，并核对逐行度匹配关系。
    assert np.unique(positive_keys).size == n_positive
    assert np.unique(negative_keys).size == n_positive
    assert np.intersect1d(positive_keys, negative_keys).size == 0
    assert not np.asarray(
        adjacency[negative_edges[:, 0], negative_edges[:, 1]]
    ).any()
    assert np.array_equal(positive_edges[:, 0], negative_edges[:, 0])
    assert np.array_equal(
        degree_bin[positive_edges[:, 1]], degree_bin[negative_edges[:, 1]]
    )

    evaluation_set = pd.DataFrame(
        {
            "positive_Index1": positive_edges[:, 0],
            "positive_Index2": positive_edges[:, 1],
            "negative_Index1": negative_edges[:, 0],
            "negative_Index2": negative_edges[:, 1],
        }
    )

    counts = {
        "candidate_edges": int(candidate_rows.size),
        "positive_edges": n_positive,
        "negative_edges": int(negative_edges.shape[0]),
    }
    return evaluation_set, counts


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建冻结的 PPI 正负边评估集")
    parser.add_argument("--ppi-path", type=Path, default=DEFAULT_PPI_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    index1, index2, combined_score = load_ppi_edges(args.ppi_path)
    evaluation_set, counts = build_evaluation_set(
        index1, index2, combined_score, seed=args.seed
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / OUTPUT_NAME
    evaluation_set.to_csv(output_path, index=False)

    print(f"Candidate edges: {counts['candidate_edges']}")
    print(f"Positive edges: {counts['positive_edges']}")
    print(f"Negative edges: {counts['negative_edges']}")
    print(f"Seed: {args.seed}")
    print(f"Saved: {output_path}")
    print(f"SHA-256: {_sha256_file(output_path)}")


if __name__ == "__main__":
    main()
