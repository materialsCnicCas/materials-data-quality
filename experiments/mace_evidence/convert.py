#!/usr/bin/env python3
"""
将宽表 CSV（mp_20_train_raw/clean 等）直接转换为 ASE extxyz。
兼容字段：
- 晶格：优先 `lattice_mat_?_?` 3x3 矩阵；否则使用 `abc_*` + `angle_*`
- 坐标系：`cartesian` 为真则笛卡尔，否则分数坐标
- 原子：`atom_element_i` + `coord_i_x/y/z`，自动扫描 i
- 能量列：默认 `formation_energy_per_atom`，可用 --energy-col 指定

输出：每个结构写入 energy 属性，extxyz 可被 MACE/ASE 直接读取。

示例：
  python convert_to_mace_xyz.py \
    --csv mp_20_train_raw.csv \
    --out mp_20_train_raw.xyz
"""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import write
from pymatgen.core import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.io.ase import AseAtomsAdaptor


TARGET_DEFAULT = "formation_energy_per_atom"
MAX_ATOMS = 40


def _is_per_atom_energy(col_name: str) -> bool:
    """简单判断能量列是否为“每原子”量纲。"""
    col = col_name.lower()
    return "per_atom" in col or "peratom" in col


def _collect_atom_indices(columns: List[str]) -> List[int]:
    idx = []
    for c in columns:
        if c.startswith("atom_element_"):
            try:
                idx.append(int(c.split("_")[-1]))
            except Exception:
                continue
    return sorted(set(idx))


def _build_structure(row: pd.Series, atom_idx: List[int]) -> Optional[Structure]:
    has_mat = all(f"lattice_mat_{i}_{j}" in row for i in range(3) for j in range(3))
    if has_mat:
        mat = [[float(row[f"lattice_mat_{i}_{j}"]) for j in range(3)] for i in range(3)]
        lattice = Lattice(mat)
    else:
        for k in ["abc_0", "abc_1", "abc_2", "angle_0", "angle_1", "angle_2"]:
            if k not in row:
                return None
        lattice = Lattice.from_lengths_and_angles(
            [float(row["abc_0"]), float(row["abc_1"]), float(row["abc_2"])],
            [float(row["angle_0"]), float(row["angle_1"]), float(row["angle_2"])],
        )

    cartesian = str(row.get("cartesian", "False")).lower() in ["1", "true", "t", "yes"]

    species, coords = [], []
    for i in atom_idx:
        elem_col = f"atom_element_{i}"
        if elem_col not in row:
            continue
        elem = row[elem_col]
        if pd.isna(elem) or str(elem).strip() == "":
            continue
        cx, cy, cz = (f"coord_{i}_x", f"coord_{i}_y", f"coord_{i}_z")
        if not all(col in row for col in (cx, cy, cz)):
            continue
        try:
            pos = [float(row[cx]), float(row[cy]), float(row[cz])]
        except Exception:
            continue
        species.append(str(elem))
        coords.append(pos)

    if not species:
        return None

    return Structure(lattice=lattice, species=species, coords=coords, coords_are_cartesian=cartesian)


def _train_val_split(n: int, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    split = int(n * (1 - val_ratio))
    return idx[:split], idx[split:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="input wide-table csv")
    ap.add_argument("--out", required=True, help="output extxyz path or prefix")
    ap.add_argument("--val-ratio", type=float, default=0.0, help="split a val set from CSV (0=no split)")
    ap.add_argument("--seed", type=int, default=42, help="shuffle seed for train/val split")
    ap.add_argument("--energy-col", default=TARGET_DEFAULT, help="energy column name")
    ap.add_argument("--append", action="store_true", help="append to existing xyz")
    ap.add_argument("--max-atoms", type=int, default=MAX_ATOMS, help="fallback atom slots if columns not explicit")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.energy_col not in df.columns:
        raise ValueError(f"找不到能量列 {args.energy_col}")

    df[args.energy_col] = pd.to_numeric(df[args.energy_col], errors="coerce")
    df = df.dropna(subset=[args.energy_col])

    atom_idx = _collect_atom_indices(list(df.columns))
    if not atom_idx:
        atom_idx = list(range(args.max_atoms))

    adaptor = AseAtomsAdaptor()

    def rows_to_atoms(sub_df: pd.DataFrame) -> Tuple[List[Atoms], int]:
        atoms_list: List[Atoms] = []
        n_fail = 0
        energy_is_per_atom = _is_per_atom_energy(args.energy_col)
        for _, row in sub_df.iterrows():
            s = _build_structure(row, atom_idx)
            if s is None:
                n_fail += 1
                continue
            atoms = adaptor.get_atoms(s)
            atoms.info = atoms.info or {}
            e_val = float(row[args.energy_col])
            if energy_is_per_atom:
                e_val *= len(atoms)
            atoms.info["energy"] = e_val
            atoms_list.append(atoms)
        return atoms_list, n_fail

    if args.val_ratio > 0:
        tr_idx, va_idx = _train_val_split(len(df), args.val_ratio, args.seed)
        df_tr = df.iloc[tr_idx].reset_index(drop=True)
        df_va = df.iloc[va_idx].reset_index(drop=True)

        atoms_tr, fail_tr = rows_to_atoms(df_tr)
        atoms_va, fail_va = rows_to_atoms(df_va)

        out_pref = Path(args.out)
        out_pref.parent.mkdir(parents=True, exist_ok=True)
        train_out = out_pref if out_pref.suffix else out_pref.with_suffix(".train.xyz")
        val_out = out_pref.with_suffix(".val.xyz") if out_pref.suffix else out_pref.with_suffix(".val.xyz")
        write(train_out, atoms_tr, format="extxyz", append=args.append)
        write(val_out, atoms_va, format="extxyz", append=args.append)
        print(
            f"Wrote train {len(atoms_tr)} (fail={fail_tr}) to {train_out}; val {len(atoms_va)} (fail={fail_va}) to {val_out}"
        )
    else:
        atoms_list, n_fail = rows_to_atoms(df)
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write(out_path, atoms_list, format="extxyz", append=args.append)
        print(f"Wrote {len(atoms_list)} structures to {out_path} (fail={n_fail}, append={args.append})")


if __name__ == "__main__":
    main()
