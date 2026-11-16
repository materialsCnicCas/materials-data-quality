#!/usr/bin/env python3
"""
filter_xyz.py
=============
读取 extxyz 文件，过滤掉"最小原子间距 < min_dist_ang Å"或
"最大邻居数 > max_neighbors（r_max 半径内）"的问题结构，
输出干净的 xyz 供 MACE 训练。

用法示例（服务器上直接运行）：
  python filter_xyz.py \
    --input  mp_20_train_noise_split.xyz \
    --output mp_20_train_noise_split_clean.xyz \
    --min-dist 1.0 \
    --max-neighbors 120 \
    --r-max 4.0

依赖：ase（mace 环境已有）
"""

import argparse
from pathlib import Path

import numpy as np
from ase.io import read, write


def min_pairwise_dist(atoms) -> float:
    """返回结构中所有原子对的最小欧氏距离（Å，考虑周期性边界）。"""
    pos = atoms.get_positions()
    n = len(pos)
    if n <= 1:
        return float("inf")
    # 使用 ASE 的 get_all_distances（考虑 PBC）
    try:
        dists = atoms.get_all_distances(mic=True)
        np.fill_diagonal(dists, np.inf)
        return float(dists.min())
    except Exception:
        # fallback：直接欧氏距离（无 PBC）
        diff = pos[:, None, :] - pos[None, :, :]
        d2 = (diff ** 2).sum(-1)
        np.fill_diagonal(d2, np.inf)
        return float(np.sqrt(d2.min()))


def max_neighbors_in_cutoff(atoms, r_max: float) -> int:
    """返回该结构中任意原子在 r_max 球内的最大邻居数。"""
    try:
        from ase.neighborlist import NeighborList
        cutoffs = [r_max / 2.0] * len(atoms)
        nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
        nl.update(atoms)
        max_nb = 0
        for i in range(len(atoms)):
            indices, _ = nl.get_neighbors(i)
            max_nb = max(max_nb, len(indices))
        return max_nb
    except Exception:
        return 0


def filter_xyz(
    input_path: Path,
    output_path: Path,
    min_dist_ang: float = 1.0,
    max_neighbors: int = 120,
    r_max: float = 4.0,
    check_neighbors: bool = True,
) -> None:
    print(f"Reading {input_path} ...")
    all_atoms = read(str(input_path), index=":")
    total = len(all_atoms)
    print(f"  Total structures: {total}")

    kept = []
    n_min_dist = 0
    n_max_nb = 0

    for i, atoms in enumerate(all_atoms):
        if i % 5000 == 0:
            print(f"  Processing {i}/{total}...")

        d = min_pairwise_dist(atoms)
        if d < min_dist_ang:
            n_min_dist += 1
            continue

        if check_neighbors:
            nb = max_neighbors_in_cutoff(atoms, r_max)
            if nb > max_neighbors:
                n_max_nb += 1
                continue

        kept.append(atoms)

    print(f"\n{'='*55}")
    print(f"  Input            : {input_path}")
    print(f"  Output           : {output_path}")
    print(f"  Total            : {total}")
    print(f"  Removed (dist)   : {n_min_dist}  (min_dist < {min_dist_ang} Å)")
    print(f"  Removed (nb)     : {n_max_nb}  (max_neighbors > {max_neighbors}, r={r_max} Å)")
    print(f"  Kept             : {len(kept)}")
    print(f"  Retention rate   : {len(kept)/total:.1%}")
    print(f"{'='*55}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write(str(output_path), kept, format="extxyz")
    print(f"Saved to {output_path}")


def main():
    ap = argparse.ArgumentParser(
        description="过滤 extxyz 中含极近原子对或邻居暴增的问题结构，避免 MACE OOM。"
    )
    ap.add_argument("--input",          required=True,       help="输入 xyz 路径")
    ap.add_argument("--output",         required=True,       help="输出 xyz 路径")
    ap.add_argument("--min-dist",       type=float, default=1.0,
                    help="最小允许原子间距 Å（默认 1.0）")
    ap.add_argument("--max-neighbors",  type=int,   default=120,
                    help="最大允许邻居数（默认 120）")
    ap.add_argument("--r-max",          type=float, default=4.0,
                    help="邻居搜索截断半径 Å，应与训练时 --r_max 一致（默认 4.0）")
    ap.add_argument("--no-neighbor-check", action="store_true",
                    help="跳过邻居数检查（仅做最小间距过滤，速度更快）")
    args = ap.parse_args()

    filter_xyz(
        input_path=Path(args.input),
        output_path=Path(args.output),
        min_dist_ang=args.min_dist,
        max_neighbors=args.max_neighbors,
        r_max=args.r_max,
        check_neighbors=not args.no_neighbor_check,
    )


if __name__ == "__main__":
    main()
