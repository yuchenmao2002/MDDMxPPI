#!/usr/bin/env python3
"""从原始 PPI 网络中抽取高置信验证边，并生成两个阶段各自的训练网络。"""

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


DEFAULT_PPI_PATH = Path(
    "/home/svu/e1538713/CodeNo0/data/processed/STRING/PPI_index.csv"
)
DEFAULT_OUTPUT_DIR = Path("/home/svu/e1538713/CodeNo0/data/evaluation")
TRAIN_STAGE1_FILE_NAME = "PPI_train_stage1.csv"
TRAIN_STAGE2_FILE_NAME = "PPI_train_stage2.csv"
VALIDATION_THETA_FILE_NAME = "PPI_validation_theta.csv"
VALIDATION_RANK_FILE_NAME = "PPI_validation_rank.csv"
USE_COLUMNS = ["Index1", "Index2", "combined_score"]
NUM_GENES = 19_295
SCORE_THRESHOLD = 700
VALIDATION_FRACTION = 0.1
RANDOM_SEED = 0


def load_ppi_network(ppi_path: Path) -> pd.DataFrame:
    """只读取输出需要的三列原始 PPI 网络。"""
    return pd.read_csv(
        ppi_path,
        usecols=USE_COLUMNS,
        dtype={"Index1": "int32", "Index2": "int32", "combined_score": "int32"},
    )


def select_validation_rows(
    ppi: pd.DataFrame,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """随机选择验证边，同时保证高置信子图中的剩余度不低于 1。"""
    index1 = ppi["Index1"].to_numpy(copy=False)
    index2 = ppi["Index2"].to_numpy(copy=False)
    combined_score = ppi["combined_score"].to_numpy(copy=False)

    candidate_rows = np.flatnonzero(combined_score >= SCORE_THRESHOLD)
    candidate_index1 = index1[candidate_rows]
    candidate_index2 = index2[candidate_rows]
    high_confidence_degree = np.bincount(
        np.concatenate((candidate_index1, candidate_index2)), minlength=NUM_GENES
    )
    remaining_degree = high_confidence_degree.copy()

    target_size = int(candidate_rows.size * VALIDATION_FRACTION)
    selected_rows = np.empty(target_size, dtype=np.int64)
    selected_count = 0

    # 随机遍历高置信边；只有删除后两个端点仍至少保留一条高置信边时才抽取。
    for row_id in rng.permutation(candidate_rows):
        first = index1[row_id]
        second = index2[row_id]
        if remaining_degree[first] > 1 and remaining_degree[second] > 1:
            selected_rows[selected_count] = row_id
            selected_count += 1
            remaining_degree[first] -= 1
            remaining_degree[second] -= 1
            if selected_count == target_size:
                break

    if selected_count != target_size:
        raise RuntimeError(
            f"在剩余度约束下只能抽取 {selected_count} 条边，少于目标 {target_size} 条"
        )

    selected_rows.sort()
    assert np.all(remaining_degree[high_confidence_degree > 0] >= 1)
    return selected_rows, high_confidence_degree, remaining_degree


def partition_validation_rows(
    validation_rows: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """把验证边均匀地划分为互斥的两组：stage 1 选 θ 用，stage 2 选 r 用。"""
    shuffled = rng.permutation(validation_rows)
    half = shuffled.size // 2
    theta_rows = np.sort(shuffled[:half])
    rank_rows = np.sort(shuffled[half:])
    return theta_rows, rank_rows


def split_ppi_network(
    ppi: pd.DataFrame,
    seed: int = RANDOM_SEED,
) -> Tuple[Dict[str, pd.DataFrame], np.ndarray, np.ndarray]:
    """返回两个阶段各自的训练网络与两组互斥的验证边，按输出文件名索引。"""
    rng = np.random.default_rng(seed)
    validation_rows, degree_before, degree_after = select_validation_rows(ppi, rng)
    theta_rows, rank_rows = partition_validation_rows(validation_rows, rng)

    is_theta = np.zeros(len(ppi), dtype=bool)
    is_theta[theta_rows] = True
    is_rank = np.zeros(len(ppi), dtype=bool)
    is_rank[rank_rows] = True

    # stage 1 选 θ：两组验证边都要删除；stage 2 选 r：只删除 r 的那一组，
    # θ 的验证边此时可以回到训练网络里，因为 r 的验证边始终未被 stage 1 使用过。
    frames = {
        TRAIN_STAGE1_FILE_NAME: ppi.loc[~(is_theta | is_rank), USE_COLUMNS].copy(),
        TRAIN_STAGE2_FILE_NAME: ppi.loc[~is_rank, USE_COLUMNS].copy(),
        VALIDATION_THETA_FILE_NAME: ppi.loc[is_theta, USE_COLUMNS].copy(),
        VALIDATION_RANK_FILE_NAME: ppi.loc[is_rank, USE_COLUMNS].copy(),
    }

    theta_frame = frames[VALIDATION_THETA_FILE_NAME]
    rank_frame = frames[VALIDATION_RANK_FILE_NAME]
    assert not np.any(is_theta & is_rank)
    assert abs(len(theta_frame) - len(rank_frame)) <= 1
    assert (
        len(frames[TRAIN_STAGE1_FILE_NAME]) + len(theta_frame) + len(rank_frame)
        == len(ppi)
    )
    assert len(frames[TRAIN_STAGE2_FILE_NAME]) + len(rank_frame) == len(ppi)
    assert (theta_frame["combined_score"] >= SCORE_THRESHOLD).all()
    assert (rank_frame["combined_score"] >= SCORE_THRESHOLD).all()
    return frames, degree_before, degree_after


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="构建两阶段超参数选择所需的 PPI 训练网络与验证边"
    )
    parser.add_argument("--ppi-path", type=Path, default=DEFAULT_PPI_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    ppi = load_ppi_network(args.ppi_path)
    frames, degree_before, degree_after = split_ppi_network(ppi, seed=args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"High-confidence edges: {(ppi['combined_score'] >= SCORE_THRESHOLD).sum()}")
    print(
        "Minimum retained high-confidence degree (stage 1): "
        f"{degree_after[degree_before > 0].min()}"
    )
    print(f"Seed: {args.seed}")
    for file_name, frame in frames.items():
        output_path = args.output_dir / file_name
        frame.to_csv(output_path, index=False)
        print(f"Saved: {output_path} ({len(frame)} edges)")


if __name__ == "__main__":
    main()
