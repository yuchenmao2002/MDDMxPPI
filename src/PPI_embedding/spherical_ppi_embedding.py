#!/usr/bin/env python3
"""球面 PPI 嵌入的最小实现。"""

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import torch
from safetensors.torch import save_file


DEFAULT_PPI_PATH = Path(
    "/home/svu/e1538713/CodeNo0/data/processed/STRING/PPI_index.csv"
)
DEFAULT_OUTPUT_DIR = Path("/home/svu/e1538713/CodeNo0/data/processed/PPI")
NUM_GENES = 19_295
ALPHA = 0.5
EPSILON = 1e-3
SCORE_SCALE = 1_000.0
LOBPCG_NITER = 200
LOBPCG_TOL = 1e-5


def load_ppi_edges(
    ppi_path: Path = DEFAULT_PPI_PATH,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """只加载超参数搜索可复用的三列 PPI 边数据。"""
    frame = pd.read_csv(
        ppi_path,
        usecols=["Index1", "Index2", "combined_score"],
        dtype={"Index1": "int32", "Index2": "int32", "combined_score": "int32"},
    )
    return tuple(
        torch.from_numpy(frame[column].to_numpy(copy=True))
        for column in ("Index1", "Index2", "combined_score")
    )


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA，但当前 PyTorch 环境没有可用 GPU")
    return resolved


def _normalized_adjacency(
    index1: torch.Tensor,
    index2: torch.Tensor,
    combined_score: torch.Tensor,
    tau: float,
    num_genes: int,
    device: torch.device,
) -> torch.Tensor:
    # 第一步：阈值过滤、置信度缩放，并由半边恢复对称加权邻接矩阵 S。
    keep = (combined_score >= tau) & (index1 != index2)
    first = index1[keep].to(device=device, dtype=torch.int64)
    second = index2[keep].to(device=device, dtype=torch.int64)
    half_values = combined_score[keep].to(device=device, dtype=torch.float32)
    half_values = half_values / SCORE_SCALE

    indices = torch.cat(
        (torch.stack((first, second)), torch.stack((second, first))), dim=1
    )
    values = torch.cat((half_values, half_values))
    adjacency = torch.sparse_coo_tensor(
        indices,
        values,
        (num_genes, num_genes),
        device=device,
        check_invariants=False,
    ).coalesce()

    # 第二步：D_ii 为 S 的加权行和；孤立节点的 D_ii^(-1/2) 取零。
    row, col = adjacency.indices()
    degree = torch.zeros(num_genes, dtype=torch.float32, device=device)
    degree.scatter_add_(0, row, adjacency.values())
    inv_sqrt_degree = torch.zeros_like(degree)
    nonisolated = degree > 0
    inv_sqrt_degree[nonisolated] = degree[nonisolated].rsqrt()
    normalized_values = (
        adjacency.values() * inv_sqrt_degree[row] * inv_sqrt_degree[col]
    )
    return torch.sparse_coo_tensor(
        adjacency.indices(),
        normalized_values,
        adjacency.shape,
        device=device,
        check_invariants=False,
    ).coalesce()


def _diffuse(normalized_adjacency: torch.Tensor, k_diff: int) -> torch.Tensor:
    # 第三步：递推计算各阶矩阵幂并加权；只在全部阶数累加后清除对角线。
    normalizer = 1.0 - ALPHA**k_diff
    coefficient = (1.0 - ALPHA) / normalizer
    power = normalized_adjacency
    diffused = (power * coefficient).coalesce()

    for k in range(2, k_diff + 1):
        power = torch.sparse.mm(power, normalized_adjacency).coalesce()
        coefficient = ALPHA ** (k - 1) * (1.0 - ALPHA) / normalizer
        diffused = (diffused + power * coefficient).coalesce()

    indices = diffused.indices()
    off_diagonal = indices[0] != indices[1]
    return torch.sparse_coo_tensor(
        indices[:, off_diagonal],
        diffused.values()[off_diagonal],
        diffused.shape,
        device=diffused.device,
        check_invariants=False,
    ).coalesce()


def _shifted_ppmi(diffused: torch.Tensor, b: float) -> torch.Tensor:
    # 第四步：先由 M 计算 d_i^M 与 vol，再仅在 M 的非零项上计算 shifted PPMI。
    row, col = diffused.indices()
    values = diffused.values()
    degree = torch.zeros(diffused.shape[0], dtype=values.dtype, device=values.device)
    degree.scatter_add_(0, row, values)
    vol = values.sum()

    shifted = (
        values.log()
        + vol.log()
        - degree[row].log()
        - degree[col].log()
        - math.log(b)
    )
    positive = shifted > 0
    return torch.sparse_coo_tensor(
        diffused.indices()[:, positive],
        shifted[positive],
        diffused.shape,
        device=diffused.device,
        check_invariants=False,
    ).coalesce()


def build_spherical_ppi_embedding(
    index1: torch.Tensor,
    index2: torch.Tensor,
    combined_score: torch.Tensor,
    tau: float,
    k_diff: int,
    b: float,
    rank: int,
    num_genes: int = NUM_GENES,
    device: str = "cpu",
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    """按六个步骤计算固定秩的球面 PPI 嵌入。"""
    compute_device = _resolve_device(device)
    normalized_adjacency = _normalized_adjacency(
        index1, index2, combined_score, tau, num_genes, compute_device
    )
    diffused = _diffuse(normalized_adjacency, k_diff)
    del normalized_adjacency
    ppmi = _shifted_ppmi(diffused, b)
    del diffused

    # 第五步：LOBPCG 求最大的 r 个代数特征值，并仅接受全部为正的谱截断。
    generator = torch.Generator(device=compute_device)
    generator.manual_seed(seed)
    initial = torch.randn(
        (num_genes, rank),
        dtype=torch.float32,
        device=compute_device,
        generator=generator,
    )
    eigenvalues, eigenvectors = torch.lobpcg(
        ppmi,
        k=rank,
        X=initial,
        niter=LOBPCG_NITER,
        tol=LOBPCG_TOL,
        largest=True,
        method="ortho",
    )
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    if not bool(torch.all(eigenvalues > 0)):
        raise RuntimeError(f"P 的正特征值不足请求的嵌入秩 r={rank}")

    # 特征向量的整体符号是任意的；固定为「每列绝对值最大的分量为正」，
    # 使不同超参数下产出的 eigenvectors 可以直接比较。
    pivot = eigenvectors.abs().argmax(dim=0)
    column = torch.arange(rank, device=compute_device)
    eigenvectors = eigenvectors * eigenvectors[pivot, column].sign()

    raw_embedding = eigenvectors * eigenvalues.sqrt().unsqueeze(0)
    del ppmi, initial

    # 第六步：按原始行范数识别退化集合；其余行去均值并投影到单位球面。
    raw_norm = torch.linalg.vector_norm(raw_embedding, dim=1)
    free_mask = raw_norm < EPSILON * raw_norm.median()
    retained = ~free_mask
    row_mean = raw_embedding[retained].mean(dim=0)
    centered = raw_embedding[retained] - row_mean

    embedding = torch.empty_like(raw_embedding)
    embedding[retained] = centered / torch.linalg.vector_norm(
        centered, dim=1, keepdim=True
    )

    # 退化行用标准高斯向量归一化，得到球面均匀的自由向量初始化。
    free_vectors = torch.randn(
        (int(free_mask.sum().item()), rank),
        dtype=torch.float32,
        device=compute_device,
        generator=generator,
    )
    embedding[free_mask] = free_vectors / torch.linalg.vector_norm(
        free_vectors, dim=1, keepdim=True
    )

    tensors = {
        "embedding": embedding,
        "gene_ids": torch.arange(num_genes, dtype=torch.int64, device=compute_device),
        "free_mask": free_mask,
        "row_mean": row_mean,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
    }
    return {name: tensor.detach().cpu().contiguous() for name, tensor in tensors.items()}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gene_ids_sha256(gene_ids: torch.Tensor) -> str:
    payload = json.dumps(
        gene_ids.tolist(), ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _number_token(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def save_embedding(
    tensors: Dict[str, torch.Tensor],
    ppi_path: Path,
    output_dir: Path,
    tau: float,
    k_diff: int,
    b: float,
    rank: int,
    device: str,
    seed: int,
    output_stem: Optional[str] = None,
) -> Tuple[Path, Path]:
    """保存固定秩嵌入及其同名 provenance JSON。"""
    if output_stem is None:
        output_stem = (
            f"spherical_ppi_tau{_number_token(tau)}_"
            f"k{k_diff}_b{_number_token(b)}_r{rank}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = output_dir / f"{output_stem}.safetensors"
    json_path = output_dir / f"{output_stem}.json"

    save_file(tensors, str(tensor_path))
    provenance = {
        "schema_version": "spherical_ppi_embedding.v2",
        "artifact_type": "spherical_ppi_embedding",
        "tensor_file": tensor_path.name,
        "safetensors_sha256": _sha256_file(tensor_path),
        "num_genes": int(tensors["gene_ids"].numel()),
        "embedding_rank": rank,
        "gene_ids_sha256": _gene_ids_sha256(tensors["gene_ids"]),
        "gene_ids_hash_spec": "compact-json-utf8-v1",
        "gene_id_semantics": "zero_based_contiguous_integer_index",
        "source": {
            "ppi_path": str(ppi_path),
            "ppi_sha256": _sha256_file(ppi_path),
            "columns": ["Index1", "Index2", "combined_score"],
            "edge_storage": "one_triangle_of_undirected_graph",
            "score_scale": SCORE_SCALE,
        },
        "hyperparameters": {
            "tau": tau,
            "K_diff": k_diff,
            "b": b,
            "r": rank,
        },
        "constants": {"alpha": ALPHA, "epsilon": EPSILON},
        "compute": {
            "eigensolver": "torch.lobpcg",
            "lobpcg_method": "ortho",
            "lobpcg_niter": LOBPCG_NITER,
            "lobpcg_tol": LOBPCG_TOL,
            "dtype": "float32",
            "device": str(_resolve_device(device)),
            "seed": seed,
            "torch_version": torch.__version__,
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(provenance, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return tensor_path, json_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建固定秩球面 PPI 嵌入")
    parser.add_argument("--tau", type=float, required=True, help="combined_score 置信阈值")
    parser.add_argument("--k-diff", type=int, required=True, help="扩散阶数")
    parser.add_argument("--b", type=float, required=True, help="PPMI 负采样位移")
    parser.add_argument("--rank", type=int, required=True, help="PPI 嵌入秩")
    parser.add_argument("--ppi-path", type=Path, default=DEFAULT_PPI_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-stem", default=None)
    parser.add_argument("--device", default="cpu", help="cpu（默认）、auto 或 CUDA 设备")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print(f"Loading PPI network: {args.ppi_path}")
    index1, index2, combined_score = load_ppi_edges(args.ppi_path)
    print(f"Computing spherical PPI embedding on {_resolve_device(args.device)}")
    tensors = build_spherical_ppi_embedding(
        index1=index1,
        index2=index2,
        combined_score=combined_score,
        tau=args.tau,
        k_diff=args.k_diff,
        b=args.b,
        rank=args.rank,
        device=args.device,
        seed=args.seed,
    )
    tensor_path, json_path = save_embedding(
        tensors=tensors,
        ppi_path=args.ppi_path,
        output_dir=args.output_dir,
        tau=args.tau,
        k_diff=args.k_diff,
        b=args.b,
        rank=args.rank,
        device=args.device,
        seed=args.seed,
        output_stem=args.output_stem,
    )
    print(f"Saved: {tensor_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
