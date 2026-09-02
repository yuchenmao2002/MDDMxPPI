#!/usr/bin/env python3
"""由冻结的球面 PPI 嵌入离线导出 MoE 静态路由表。"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from safetensors.torch import load_file, save_file


DEFAULT_EMBEDDING_PATH = Path(
    "/home/svu/e1538713/CodeNo0/data/processed/PPI/"
    "spherical_ppi_tau700_k4_b1_r64.safetensors"
)
DEFAULT_OUTPUT_DIR = Path("/home/svu/e1538713/CodeNo0/data/processed/PPI")
OUTPUT_STEM = "ppi_moe_routing"

NUM_EXPERTS = 4
K_ROUTE = 2
T_ROUTE = 0.03
SEEDS: Tuple[int, ...] = (0, 1, 2, 3)
MAX_KMEANS_ITERS = 200
FREE_WEIGHT = 0.5


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def balanced_capacity(num_genes: int) -> torch.Tensor:
    """均衡的整数容量：前 r_C 个专家各多承担一条路径。"""
    quotient, remainder = divmod(K_ROUTE * num_genes, NUM_EXPERTS)
    return torch.tensor(
        [quotient + 1] * remainder + [quotient] * (NUM_EXPERTS - remainder),
        dtype=torch.int64,
    )


def fit_prototypes(
    real_embedding: torch.Tensor, seed: int
) -> Tuple[torch.Tensor, float, int]:
    """在真实基因上做无容量约束的 top-k 球面 k-means，返回原型与目标值。"""
    generator = torch.Generator()
    generator.manual_seed(seed)
    initial = torch.randperm(len(real_embedding), generator=generator)[:NUM_EXPERTS]
    prototypes = real_embedding[initial].clone()

    previous = None
    for iteration in range(1, MAX_KMEANS_ITERS + 1):
        support = (real_embedding @ prototypes.T).topk(K_ROUTE, dim=1).indices
        if previous is not None and torch.equal(support, previous):
            break
        previous = support
        for expert in range(NUM_EXPERTS):
            total = real_embedding[(support == expert).any(dim=1)].sum(dim=0)
            if total.norm() > 0:
                prototypes[expert] = total / total.norm()

    objective = float(
        (real_embedding @ prototypes.T).topk(K_ROUTE, dim=1).values.sum()
    )
    return prototypes, objective, iteration


def ballast_deficits(
    real_load: torch.Tensor, capacity: torch.Tensor, num_free: int
) -> torch.Tensor:
    """自由基因只能填补缺口：逐条注水到当前负载最小、且未顶到 |F| 上界的专家。"""
    deficits = torch.zeros(NUM_EXPERTS, dtype=torch.int64)
    load = real_load.clone()
    for _ in range(K_ROUTE * num_free):
        expert = min(
            (e for e in range(NUM_EXPERTS) if int(deficits[e]) < num_free),
            key=lambda e: (int(load[e]), e),
        )
        deficits[expert] += 1
        load[expert] += 1
    assert int(deficits.sum()) == K_ROUTE * num_free
    assert int(deficits.max()) <= num_free
    return deficits


def assign_free_genes(num_free: int, deficits: torch.Tensor) -> torch.Tensor:
    """按基因编号从小到大，每个自由基因取当前缺口最大的两个专家（并列取编号小者）。"""
    remaining = deficits.clone()
    support = torch.empty((num_free, K_ROUTE), dtype=torch.int64)
    for row in range(num_free):
        order = sorted(
            range(NUM_EXPERTS), key=lambda e: (-int(remaining[e]), e)
        )[:K_ROUTE]
        chosen = sorted(order)
        support[row] = torch.tensor(chosen, dtype=torch.int64)
        for expert in chosen:
            remaining[expert] -= 1
    assert int(remaining.abs().sum()) == 0, "自由基因未能恰好填满缺口"
    return support


def build_routing(
    embedding_path: Path = DEFAULT_EMBEDDING_PATH,
) -> Tuple[Dict[str, torch.Tensor], Dict]:
    """返回路由表张量与配套的 provenance 记录。"""
    tensors = load_file(str(embedding_path))
    embedding, free_mask = tensors["embedding"], tensors["free_mask"]
    num_genes, rank = embedding.shape
    real_mask = ~free_mask
    real_index = real_mask.nonzero().squeeze(1)
    free_index = free_mask.nonzero().squeeze(1)
    num_real, num_free = len(real_index), len(free_index)

    # 第一步：原型只在真实基因上拟合，自由向量完全不参与。
    fits = [fit_prototypes(embedding[real_index], seed) for seed in SEEDS]
    objectives = [objective for _, objective, _ in fits]
    chosen = max(range(len(SEEDS)), key=lambda i: objectives[i])
    prototypes, objective, iterations = fits[chosen]

    # 第二步：真实基因无容量约束地取 top-k，保留全部真实路径。
    real_top = (embedding[real_index] @ prototypes.T).topk(K_ROUTE, dim=1)
    real_load = torch.bincount(
        real_top.indices.reshape(-1), minlength=NUM_EXPERTS
    )

    # 第三步：自由基因作为压舱物填补缺口，权重固定为均分。
    capacity = balanced_capacity(num_genes)
    deficits = ballast_deficits(real_load, capacity, num_free)
    free_support = assign_free_genes(num_free, deficits)

    # 第四步：真实基因在支持集内做温度 softmax；自由基因固定 0.5/0.5。
    real_weight = torch.softmax(real_top.values / T_ROUTE, dim=1)
    free_similarity = torch.gather(
        embedding[free_index] @ prototypes.T, 1, free_support
    )

    expert_ids = torch.empty((num_genes, K_ROUTE), dtype=torch.int64)
    weights = torch.empty((num_genes, K_ROUTE), dtype=torch.float32)
    similarity = torch.empty((num_genes, K_ROUTE), dtype=torch.float32)

    # 行内一律按专家编号升序排列；第 0 列并不是主专家。
    order = real_top.indices.argsort(dim=1)
    expert_ids[real_index] = real_top.indices.gather(1, order)
    weights[real_index] = real_weight.gather(1, order)
    similarity[real_index] = real_top.values.gather(1, order)
    expert_ids[free_index] = free_support
    weights[free_index] = FREE_WEIGHT
    similarity[free_index] = free_similarity

    final_load = torch.bincount(expert_ids.reshape(-1), minlength=NUM_EXPERTS)
    assert int(final_load.sum()) == K_ROUTE * num_genes
    assert bool((expert_ids[:, 0] < expert_ids[:, 1]).all()), "行内专家未严格升序"
    assert bool(((weights.sum(dim=1) - 1.0).abs() < 1e-5).all()), "权重未归一"
    assert bool((weights[free_index] == FREE_WEIGHT).all())

    routing = {
        "expert_ids": expert_ids,
        "weights": weights,
        "similarity": similarity,
        "free_mask": free_mask,
        "prototypes": prototypes,
    }
    report = {
        "num_genes": num_genes,
        "num_real": num_real,
        "num_free": num_free,
        "seed_objectives": {str(s): objectives[i] for i, s in enumerate(SEEDS)},
        "chosen_seed": SEEDS[chosen],
        "objective": objective,
        "kmeans_iterations": iterations,
        "balanced_capacity": capacity.tolist(),
        "real_load": real_load.tolist(),
        "ballast_deficits": deficits.tolist(),
        "final_load": final_load.tolist(),
        "free_share": [
            float(deficits[e]) / float(final_load[e]) for e in range(NUM_EXPERTS)
        ],
        "load_max_over_min": float(final_load.max()) / float(final_load.min()),
        "rank": rank,
    }
    return routing, report


def save_routing(
    routing: Dict[str, torch.Tensor],
    report: Dict,
    embedding_path: Path,
    output_dir: Path,
) -> Tuple[Path, Path]:
    """保存路由表及其同名 provenance JSON。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = output_dir / f"{OUTPUT_STEM}.safetensors"
    json_path = output_dir / f"{OUTPUT_STEM}.json"
    save_file(
        {name: tensor.contiguous() for name, tensor in routing.items()},
        str(tensor_path),
    )

    provenance = {
        "schema_version": "ppi_moe_static_routing.v1",
        "artifact_type": "ppi_moe_static_routing",
        "tensor_file": tensor_path.name,
        "safetensors_sha256": _sha256_file(tensor_path),
        "source": {
            "embedding_path": str(embedding_path),
            "embedding_sha256": _sha256_file(embedding_path),
            "embedding_rank": report["rank"],
        },
        "routing": {
            "num_genes": report["num_genes"],
            "num_experts": NUM_EXPERTS,
            "k_route": K_ROUTE,
            "T_route": T_ROUTE,
        },
        "tensors": {
            "expert_ids": (
                f"({report['num_genes']}, {K_ROUTE}) int64。**行内按专家编号升序排列，"
                "第 0 列不是主专家**；主专家是 weights 较大的那一列。"
            ),
            "weights": (
                f"({report['num_genes']}, {K_ROUTE}) float32，与 expert_ids 逐列对齐，"
                "每行和为 1。"
            ),
            "similarity": (
                f"({report['num_genes']}, {K_ROUTE}) float32，对应的 z_i^T mu_e，"
                "仅供诊断，不参与路由。"
            ),
            "free_mask": "(G,) bool，True 表示该基因的嵌入是自由向量（压舱物）。",
            "prototypes": f"({NUM_EXPERTS}, {report['rank']}) float32 单位球面原型 mu_e。",
        },
        "prototype_fit": {
            "genes_used": "仅真实基因（free_mask 为 False），自由向量完全不参与拟合",
            "num_real": report["num_real"],
            "algorithm": "unconstrained top-k spherical k-means, alternating optimisation",
            "init": "从真实基因中随机取 E 行作为初始原型（torch.randperm，逐 seed）",
            "stopping": f"支持集不再变化，或达到 {MAX_KMEANS_ITERS} 轮",
            "seeds": list(SEEDS),
            "seed_objectives": report["seed_objectives"],
            "selection_rule": "取目标值 J 最大的 seed",
            "chosen_seed": report["chosen_seed"],
            "objective": report["objective"],
            "iterations_of_chosen_seed": report["kmeans_iterations"],
        },
        "real_genes": {
            "assignment": "无容量约束，直接取 z_i^T mu_e 的前 k 个专家（全部真实路径均被保留）",
            "weights": "支持集内的温度 softmax，pi = softmax(z^T mu / T_route)",
        },
        "free_genes": {
            "num_free": report["num_free"],
            "assignment": (
                "作为压舱物填补容量缺口：逐条注水到当前负载最小且未顶到 |F| 上界的专家，"
                "再按基因编号从小到大、每次取缺口最大的两个专家（并列取编号小者）实现配对"
            ),
            "weights": f"固定 {FREE_WEIGHT}/{FREE_WEIGHT}，不使用 T_route",
            "note": "自由向量对原型、指派与权重均无影响",
        },
        "load": {
            "balanced_capacity": report["balanced_capacity"],
            "real_load": report["real_load"],
            "ballast_deficits": report["ballast_deficits"],
            "final_load": report["final_load"],
            "free_share_per_expert": report["free_share"],
            "max_over_min": report["load_max_over_min"],
        },
        "torch_version": torch.__version__,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(provenance, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return tensor_path, json_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="由球面 PPI 嵌入构建 MoE 静态路由表")
    parser.add_argument("--embedding-path", type=Path, default=DEFAULT_EMBEDDING_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    routing, report = build_routing(args.embedding_path)
    for key in ("seed_objectives", "chosen_seed", "objective", "real_load",
                "ballast_deficits", "final_load", "load_max_over_min"):
        print(f"{key}: {report[key]}")
    print("free_share: " + ", ".join(f"{100*x:.1f}%" for x in report["free_share"]))
    tensor_path, json_path = save_routing(
        routing, report, args.embedding_path, args.output_dir
    )
    print(f"Saved: {tensor_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
