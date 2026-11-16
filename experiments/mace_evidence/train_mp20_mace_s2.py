import os
import re
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
from ase.io import read, write

# 静默 TorchScript 关于 instance-level annotations 的警告
warnings.filterwarnings(
    "ignore",
    message=r"The TorchScript type system doesn't support instance-level annotations",
    category=UserWarning,
)


def _energy_var_per_atom(xyz_path: Path, energy_key: str) -> float:
    """计算给定 extxyz 的能量方差（单位 eV/atom）。

    若 ASE 无法解析 info.energy（当前数据集即如此），则回退为行级解析：
    利用原子计数行 + 下一行中的 energy=... 提取能量。
    """
    energies: list[float] = []

    # 1) 尝试 ASE 解析
    try:
        atoms_list = read(xyz_path, index=":")
        for atoms in atoms_list:
            e = atoms.info.get(energy_key)
            if e is None:
                continue
            energies.append(float(e) / len(atoms))
    except Exception:
        pass

    # 2) 若未取到能量，回退手工解析
    if not energies:
        lines = xyz_path.read_text().splitlines()
        prev_n = None
        for ln in lines:
            ln = ln.strip()
            if not ln:
                continue
            if prev_n is None and ln.isdigit():
                prev_n = int(ln)
                continue
            if prev_n is not None:
                m = re.search(r"energy=([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)", ln)
                if m:
                    e = float(m.group(1))
                    energies.append(e / max(prev_n, 1))
                prev_n = None

    if not energies:
        return 0.0
    arr = np.array(energies, dtype=float)
    return float(np.var(arr))


def main():
    # 配置
    num_folds = 10
    seed = 42  # 固定随机种子
    base_dir = Path(__file__).resolve().parent
    src_xyz = base_dir / "mp_20_goad_split.xyz"
    folds_dir = base_dir / "cv_folds"
    folds_dir.mkdir(parents=True, exist_ok=True)

    # 加载完整数据并划分索引
    atoms_all = read(src_xyz, index=":")
    n_total = len(atoms_all)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total)
    fold_indices = np.array_split(perm, num_folds)
    print(f"Total structures: {n_total}, folds: {num_folds}, seed: {seed}")

    # mace_run_train 路径
    exe = shutil.which("mace_run_train")
    if not exe:
        raise FileNotFoundError("mace_run_train not found in PATH. Please activate the 'mace' environment.")

    # 正则和工具函数
    rmse_re = re.compile(r"RMSE_E_per_atom=\s*([0-9.eE+-]+)\s*(meV|eV)")
    table_re = re.compile(r"\|\s*Default_Default\s*\|\s*([0-9.]+)")

    def _rmse_to_r2(rmse_match, var):
        rmse_val = float(rmse_match.group(1))
        unit = rmse_match.group(2).lower()
        rmse_ev = rmse_val / 1000.0 if unit == "mev" else rmse_val
        return 1.0 - (rmse_ev**2) / var if var > 0 else None

    # 环境设置
    base_env = os.environ.copy()
    base_env["PYTHONWARNINGS"] = "ignore:The TorchScript type system doesn't support instance-level annotations:UserWarning"
    base_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    fold_r2_list: list[float] = []

    # 逐折训练
    for fold_id, test_idx in enumerate(fold_indices, start=1):
        train_idx = np.concatenate([idx for i, idx in enumerate(fold_indices) if i != fold_id - 1])
        fold_dir = folds_dir / f"fold_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_file = fold_dir / "train.xyz"
        test_file = fold_dir / "test.xyz"

        # 写出当前折数据
        train_atoms = [atoms_all[i] for i in train_idx]
        test_atoms = [atoms_all[i] for i in test_idx]
        write(train_file, train_atoms, format="extxyz")
        write(test_file, test_atoms, format="extxyz")

        energy_key = "energy"
        var_test = _energy_var_per_atom(test_file, energy_key)
        print(f"\n===== Fold {fold_id}/{num_folds} =====")
        print(f"Train size: {len(train_atoms)}, Test size: {len(test_atoms)}, Var(test) eV^2/atom: {var_test:.6f}")

        results_dir = base_dir / "results" / f"fold_{fold_id}"
        checkpoints_dir = base_dir / "checkpoints" / f"fold_{fold_id}"
        results_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        name = f"mp20_mace_s2_totalE_fold{fold_id}"

        cmd = [
            exe,
            f"--name={name}",
            f"--train_file={train_file}",
            f"--valid_file={test_file}",  # 本任务不单独验证，直接用测试折评估
            f"--test_file={test_file}",
            f"--results_dir={results_dir}",
            f"--checkpoints_dir={checkpoints_dir}",
            "--energy_key=energy",
            "--E0s=average",
            "--model=MACE",
            "--hidden_irreps=128x0e + 128x1o",
            "--max_L=1",
            "--num_interactions=2",
            "--r_max=4.0",
            "--batch_size=32",
            "--valid_batch_size=64",
            "--max_num_epochs=10",
            "--lr=0.005",
            "--scheduler=ReduceLROnPlateau",
            "--lr_factor=0.5",
            "--scheduler_patience=10",
            "--clip_grad=10.0",
            "--loss=weighted",
            "--forces_weight=0.0",
            "--energy_weight=1.0",
            "--stage_two",
            "--start_stage_two=160",
            "--amsgrad",
            "--eval_interval=1",
            "--error_table=PerAtomRMSE",
            "--default_dtype=float32",
            "--device=cuda",
        ]

        print("Launching:")
        print(" ".join(str(x) for x in cmd))

        best_test_r2 = None

        proc = subprocess.Popen(
            cmd,
            cwd=base_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=base_env,
        )

        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip()
            print(line, flush=True)

            m = rmse_re.search(line)
            if m and "test" in line.lower():
                r2 = _rmse_to_r2(m, var_test)
                if r2 is not None:
                    best_test_r2 = r2 if best_test_r2 is None else max(best_test_r2, r2)
                    print(f"[Fold {fold_id} R2@test] {r2:.4f}", flush=True)

            t = table_re.search(line)
            if t:
                rmse_mev = float(t.group(1))
                r2 = 1.0 - (rmse_mev / 1000.0) ** 2 / var_test if var_test > 0 else None
                if r2 is not None:
                    best_test_r2 = r2 if best_test_r2 is None else max(best_test_r2, r2)
                    print(f"[Fold {fold_id} R2@test_table] {r2:.4f}", flush=True)

        proc.wait()
        if proc.returncode != 0:
            sys.exit(proc.returncode)

        if best_test_r2 is not None:
            print(f"[Fold {fold_id} R2@test_best] {best_test_r2:.4f}")
            fold_r2_list.append(best_test_r2)
        else:
            print(f"[Fold {fold_id} R2@test] 未在日志中出现测试集 RMSE，可能未执行测试评估。")

    if fold_r2_list:
        avg_r2 = float(np.mean(fold_r2_list))
        print(f"\n===== CV Summary =====")
        print(f"Folds completed: {len(fold_r2_list)}/{num_folds}")
        print(f"Mean R2@test across folds: {avg_r2:.4f}")
    else:
        print("\n未能汇总 R2：所有折均未得到测试集 R2。")


if __name__ == "__main__":
    main()
