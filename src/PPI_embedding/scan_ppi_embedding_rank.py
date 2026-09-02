#!/usr/bin/env python3
"""扫描四个 PPI 超参数候选的前 256 个代数最大特征值。"""

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from select_ppi_hyperparameters import iter_candidate_ppmi
from spherical_ppi_embedding import (
    ALPHA,
    DEFAULT_PPI_PATH,
    LOBPCG_NITER,
    LOBPCG_TOL,
    NUM_GENES,
    _resolve_device,
    load_ppi_edges,
)


R_SCAN = 256
SEED = 0
LOBPCG_METHOD = "ortho"
DEFAULT_SELECTION_PATH = Path(
    "/home/svu/e1538713/CodeNo0/outputs/evaluation/PPI_embedding/"
    "PPI_hyperparameter_J.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/svu/e1538713/CodeNo0/outputs/evaluation/PPI_embedding"
)
SPECTRUM_NAME = "PPI_rank_scan_spectrum.csv"
SUMMARY_NAME = "PPI_rank_scan_summary.csv"
PROVENANCE_NAME = "PPI_rank_scan.json"


@dataclass(frozen=True)
class Candidate:
    selection_rank: int
    tau: int
    k_diff: int
    b: int

    @property
    def candidate_id(self) -> str:
        return f"tau{self.tau}_k{self.k_diff}_b{self.b}"


# 这四个就是评分文件里按 J 排名的前四名。差别只在最优的指定：J 最高的是
# (700,5,1)，但它与 (700,4,1) 极为接近，出于实际考量把 selection_rank=1
# 给了 (700,4,1)，其余三个视为次优。哪个是最优只在绘制谱图时单独标出。
CANDIDATES: Tuple[Candidate, ...] = (
    Candidate(1, 700, 4, 1),
    Candidate(2, 700, 5, 1),
    Candidate(3, 700, 6, 1),
    Candidate(4, 700, 7, 1),
)
CANDIDATE_BY_PARAMETERS = {
    (candidate.tau, candidate.k_diff, candidate.b): candidate
    for candidate in CANDIDATES
}

SPECTRUM_COLUMNS = [
    "selection_rank",
    "candidate_id",
    "tau",
    "K_diff",
    "b",
    "j",
    "sigma",
    "sigma_normalized",
    "rho",
    "delta",
    "status",
]
SUMMARY_COLUMNS = [
    "selection_rank",
    "candidate_id",
    "tau",
    "K_diff",
    "b",
    "validation_J",
    "R_scan",
    "sigma_1",
    "r_plus",
    "r_plus_right_censored",
    "r_plus_label",
    "r_minus",
    "r_minus_right_censored",
    "r_minus_label",
    "n_positive",
    "n_unresolved",
    "n_negative",
]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_validation_scores(selection_path: Path) -> Dict[Tuple[int, int, int], float]:
    frame = pd.read_csv(selection_path)
    key_columns = ["tau", "K_diff", "b"]
    if frame.duplicated(key_columns).any() or not np.isfinite(frame["J"]).all():
        raise RuntimeError("超参数评分文件包含重复候选或非有限 J")
    scores = {
        (int(row.tau), int(row.K_diff), int(row.b)): float(row.J)
        for row in frame.itertuples(index=False)
    }

    # 候选集合与 J 的前四名一致，但「谁排第一」由实际考量指定、与 J 的顺序不同，
    # 因此只要求评分文件里能查到这四个候选的 J，不再比较排名顺序。
    missing = [
        candidate.candidate_id
        for candidate in CANDIDATES
        if (candidate.tau, candidate.k_diff, candidate.b) not in scores
    ]
    if missing:
        raise RuntimeError(f"评分文件中缺少候选：{', '.join(missing)}")
    return scores


def _make_tracker(candidate_id: str):
    """每十次迭代打印一次 LOBPCG 的收敛进度。"""

    def tracker(state) -> None:
        step = state.ivars["istep"]
        converged = state.ivars["converged_count"]
        if step == 1 or step % 10 == 0 or converged == R_SCAN:
            max_relative_residual = float(state.tvars["rerr"].max())
            print(
                f"[{candidate_id}] LOBPCG iteration={step} "
                f"converged={converged}/{R_SCAN} "
                f"max_relative_residual={max_relative_residual:.3e}",
                flush=True,
            )

    return tracker


def _classify_spectrum(
    candidate: Candidate,
    validation_j: float,
    eigenvalues: np.ndarray,
    residuals: np.ndarray,
) -> Tuple[List[Dict], Dict]:
    """逐特征计算容差与状态，并汇总连续正谱和首个可靠负值。"""
    sigma_1 = float(eigenvalues[0])
    deltas = np.maximum(10.0 * residuals, 1e-6 * sigma_1)
    statuses = np.full(eigenvalues.shape, "unresolved", dtype=object)
    statuses[eigenvalues > deltas] = "positive"
    statuses[eigenvalues < -deltas] = "negative"

    r_plus = 0
    for status in statuses:
        if status != "positive":
            break
        r_plus += 1

    negative = np.flatnonzero(statuses == "negative")
    r_minus: Optional[int] = int(negative[0] + 1) if negative.size else None
    r_plus_censored = r_plus == eigenvalues.size
    r_minus_censored = r_minus is None

    spectrum_rows = []
    for offset, (sigma, rho, delta, status) in enumerate(
        zip(eigenvalues, residuals, deltas, statuses), start=1
    ):
        spectrum_rows.append(
            {
                "selection_rank": candidate.selection_rank,
                "candidate_id": candidate.candidate_id,
                "tau": candidate.tau,
                "K_diff": candidate.k_diff,
                "b": candidate.b,
                "j": offset,
                "sigma": float(sigma),
                "sigma_normalized": float(sigma / sigma_1),
                "rho": float(rho),
                "delta": float(delta),
                "status": status,
            }
        )

    summary_row = {
        "selection_rank": candidate.selection_rank,
        "candidate_id": candidate.candidate_id,
        "tau": candidate.tau,
        "K_diff": candidate.k_diff,
        "b": candidate.b,
        "validation_J": validation_j,
        "R_scan": int(eigenvalues.size),
        "sigma_1": sigma_1,
        "r_plus": r_plus,
        "r_plus_right_censored": r_plus_censored,
        "r_plus_label": f">={eigenvalues.size}" if r_plus_censored else str(r_plus),
        "r_minus": "" if r_minus_censored else r_minus,
        "r_minus_right_censored": r_minus_censored,
        "r_minus_label": f">{eigenvalues.size}" if r_minus_censored else str(r_minus),
        "n_positive": int(np.count_nonzero(statuses == "positive")),
        "n_unresolved": int(np.count_nonzero(statuses == "unresolved")),
        "n_negative": int(np.count_nonzero(statuses == "negative")),
    }
    return spectrum_rows, summary_row


def _solve_candidate(
    ppmi: torch.Tensor,
    candidate: Candidate,
    validation_j: float,
    eigensolver_device: torch.device,
) -> Tuple[List[Dict], Dict]:
    """求解一个候选的特征对，并在释放特征向量前计算绝对残差。"""
    matrix = ppmi.to(eigensolver_device).coalesce()
    generator = torch.Generator(device=eigensolver_device)
    generator.manual_seed(SEED)
    initial = torch.randn(
        (matrix.shape[0], R_SCAN),
        dtype=matrix.dtype,
        device=eigensolver_device,
        generator=generator,
    )

    eigenvalues, eigenvectors = torch.lobpcg(
        matrix,
        k=R_SCAN,
        X=initial,
        niter=LOBPCG_NITER,
        tol=LOBPCG_TOL,
        largest=True,
        method=LOBPCG_METHOD,
        tracker=_make_tracker(candidate.candidate_id),
    )
    del initial
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    eigenvectors = eigenvectors / torch.linalg.vector_norm(
        eigenvectors, dim=0, keepdim=True
    )

    # 残差使用 float64 累加；P 和特征向量仅在本候选计算期间保留。
    matrix64 = matrix.to(dtype=torch.float64)
    vectors64 = eigenvectors.to(dtype=torch.float64)
    values64 = eigenvalues.to(dtype=torch.float64)
    residual = torch.sparse.mm(matrix64, vectors64) - vectors64 * values64.unsqueeze(0)
    residuals = torch.linalg.vector_norm(residual, dim=0)

    eigenvalues_numpy = values64.detach().cpu().numpy()
    residuals_numpy = residuals.detach().cpu().numpy()
    if (
        not np.isfinite(eigenvalues_numpy).all()
        or not np.isfinite(residuals_numpy).all()
        or eigenvalues_numpy[0] <= 0
    ):
        raise RuntimeError(f"{candidate.candidate_id} 的 LOBPCG 谱结果无效")

    return _classify_spectrum(
        candidate, validation_j, eigenvalues_numpy, residuals_numpy
    )


def _iter_selected_ppmi(
    index1: torch.Tensor,
    index2: torch.Tensor,
    combined_score: torch.Tensor,
):
    """在 CPU 上逐候选生成 P，并在特征求解前释放扩散中间矩阵。"""
    # 按 K_diff 从小到大执行：扩散阶数越低，P 越稀疏、峰值内存越小。
    execution_order = sorted(CANDIDATES, key=lambda candidate: candidate.k_diff)
    for candidate in execution_order:
        iterator = iter_candidate_ppmi(
            index1,
            index2,
            combined_score,
            device="cpu",
            tau_values=(candidate.tau,),
            k_values=(candidate.k_diff,),
            b_values=(candidate.b,),
        )
        parameters, ppmi = next(iterator)
        iterator.close()
        del iterator
        yield parameters, ppmi
        del parameters, ppmi


def scan_rank(
    ppi_path: Path = DEFAULT_PPI_PATH,
    selection_path: Path = DEFAULT_SELECTION_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    device: str = "auto",
) -> Tuple[Path, Path, Path]:
    """计算四个候选的谱统计并保存长表、汇总表和 provenance。"""
    eigensolver_device = _resolve_device(device)
    validation_scores = _load_validation_scores(selection_path)
    index1, index2, combined_score = load_ppi_edges(ppi_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    spectrum_path = output_dir / SPECTRUM_NAME
    summary_path = output_dir / SUMMARY_NAME
    provenance_path = output_dir / PROVENANCE_NAME
    spectrum_partial = output_dir / f".{SPECTRUM_NAME}.partial"
    summary_partial = output_dir / f".{SUMMARY_NAME}.partial"
    spectrum_rows: List[Dict] = []
    summary_rows: List[Dict] = []

    for parameters, ppmi in _iter_selected_ppmi(index1, index2, combined_score):
        candidate = CANDIDATE_BY_PARAMETERS[
            (parameters["tau"], parameters["k_diff"], parameters["b"])
        ]
        print(
            f"[{candidate.candidate_id}] start LOBPCG on {eigensolver_device}; "
            f"nnz(P)={ppmi._nnz()}",
            flush=True,
        )
        started = time.perf_counter()
        rows, summary = _solve_candidate(
            ppmi,
            candidate,
            validation_scores[(candidate.tau, candidate.k_diff, candidate.b)],
            eigensolver_device,
        )
        del ppmi
        spectrum_rows.extend(rows)
        summary_rows.append(summary)

        pd.DataFrame(spectrum_rows, columns=SPECTRUM_COLUMNS).sort_values(
            ["selection_rank", "j"]
        ).to_csv(spectrum_partial, index=False, float_format="%.17g")
        pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS).sort_values(
            "selection_rank"
        ).to_csv(summary_partial, index=False, float_format="%.17g")
        print(
            f"[{candidate.candidate_id}] completed in "
            f"{time.perf_counter() - started:.1f}s; "
            f"r_plus={summary['r_plus_label']} "
            f"r_minus={summary['r_minus_label']}",
            flush=True,
        )

    spectrum_partial.replace(spectrum_path)
    summary_partial.replace(summary_path)

    candidates = []
    for candidate in CANDIDATES:
        candidates.append(
            {
                "selection_rank": candidate.selection_rank,
                "candidate_id": candidate.candidate_id,
                "tau": candidate.tau,
                "K_diff": candidate.k_diff,
                "b": candidate.b,
                "validation_J": validation_scores[
                    (candidate.tau, candidate.k_diff, candidate.b)
                ],
            }
        )
    provenance = {
        "schema_version": "ppi_embedding_rank_scan.v1",
        "artifact_type": "ppi_embedding_rank_scan",
        "source": {
            "ppi_path": str(ppi_path),
            "ppi_sha256": _sha256_file(ppi_path),
            "edge_storage": "one_triangle_of_undirected_graph",
            "selection_scores_path": str(selection_path),
            "selection_scores_sha256": _sha256_file(selection_path),
        },
        "num_genes": NUM_GENES,
        "candidate_count": len(CANDIDATES),
        "candidates": candidates,
        "constants": {"alpha": ALPHA, "R_scan": R_SCAN},
        "lobpcg": {
            "largest": True,
            "eigenvalue_order": "descending_algebraic",
            "method": LOBPCG_METHOD,
            "niter": LOBPCG_NITER,
            "tol": LOBPCG_TOL,
            "seed": SEED,
            "matrix_dtype": "float32",
            "residual_dtype": "float64",
            "diffusion_device": "cpu",
            "eigensolver_device": str(eigensolver_device),
            "torch_version": torch.__version__,
        },
        "definitions": {
            "sigma_normalized": "sigma_j / sigma_1",
            "rho": "||P u_j - sigma_j u_j||_2",
            "delta": "max(10 rho_j, 1e-6 sigma_1)",
            "positive": "sigma_j > delta_j",
            "unresolved": "abs(sigma_j) <= delta_j",
            "negative": "sigma_j < -delta_j",
            "r_plus": "longest positive prefix among j=1..R_scan",
            "r_minus": "first negative position among j=1..R_scan",
        },
        "censoring": {
            "r_plus": "right-censored when r_plus == R_scan; label >=R_scan",
            "r_minus": "right-censored when no negative is detected; label >R_scan",
        },
        "outputs": {
            "spectrum_csv": spectrum_path.name,
            "spectrum_sha256": _sha256_file(spectrum_path),
            "summary_csv": summary_path.name,
            "summary_sha256": _sha256_file(summary_path),
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with provenance_path.open("w", encoding="utf-8") as handle:
        json.dump(provenance, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return spectrum_path, summary_path, provenance_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="扫描四个 PPI 候选的前256个特征值")
    parser.add_argument("--ppi-path", type=Path, default=DEFAULT_PPI_PATH)
    parser.add_argument("--selection-path", type=Path, default=DEFAULT_SELECTION_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    outputs = scan_rank(
        ppi_path=arguments.ppi_path,
        selection_path=arguments.selection_path,
        output_dir=arguments.output_dir,
        device=arguments.device,
    )
    for output in outputs:
        print(f"wrote {output}")
