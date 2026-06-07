#!/usr/bin/env python3
"""Lightweight sequential experiment scheduler for train.py."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class RunItem:
    scene: str
    sparse_ratio: float
    seed: Optional[int]
    extra_args: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run train.py sequentially from a plan file")
    parser.add_argument("--plan", type=str, default=r"E:\PycharmProject\mipnerf-pytorch-main\aaa_optimized_version\sparsenerf_only_rank_loss\exp_plan.json",  help="Path to JSON/YAML experiment plan")
    parser.add_argument("--python", type=str, default=r"E:\PycharmProject\mipnerf-pytorch-main\.venv\Scripts\python.exe" , help="Python executable")
    parser.add_argument(
        "--logs-dir",
        type=str,
        default=r"E:\PycharmProject\mipnerf-pytorch-main\aaa_optimized_version\sparsenerf_only_rank_loss\log_log",
        help="Directory for per-run log files",
    )
    return parser.parse_args()


def load_plan(plan_path: Path) -> Dict[str, Any]:
    suffix = plan_path.suffix.lower()
    raw = plan_path.read_text(encoding="utf-8")

    if suffix in {".json"}:
        return json.loads(raw)
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError("YAML plan requires PyYAML installed.") from exc
        return yaml.safe_load(raw)
    raise ValueError(f"Unsupported plan format: {suffix}. Use .json/.yaml/.yml")


def normalize_extra_args(extra: Any) -> List[str]:
    if extra is None:
        return []
    if isinstance(extra, list):
        return [str(item) for item in extra]
    if isinstance(extra, str):
        return shlex.split(extra)
    raise TypeError("extra_args must be a string or list of strings")


def expand_runs(runs: Iterable[Dict[str, Any]]) -> List[RunItem]:
    expanded: List[RunItem] = []
    for idx, run in enumerate(runs):
        if "scene" not in run or "sparse_ratio" not in run:
            raise ValueError(f"runs[{idx}] must include scene and sparse_ratio")

        repeat = int(run.get("repeat", 1))
        if repeat < 1:
            raise ValueError(f"runs[{idx}].repeat must be >=1")

        base_seed = run.get("seed")
        extra_args = normalize_extra_args(run.get("extra_args"))

        for n in range(repeat):
            seed = None if base_seed is None else int(base_seed) + n
            expanded.append(
                RunItem(
                    scene=str(run["scene"]),
                    sparse_ratio=float(run["sparse_ratio"]),
                    seed=seed,
                    extra_args=deepcopy(extra_args),
                )
            )
    return expanded


def build_cli_args(global_cfg: Dict[str, Any]) -> List[str]:
    args: List[str] = []
    for key, value in global_cfg.items():
        if value is None:
            continue
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                args.append(flag)
            continue
        if isinstance(value, list):
            args.append(flag)
            args.extend(str(item) for item in value)
            continue
        args.extend([flag, str(value)])
    return args

# ===================== 里程碑控制 =====================
def apply_milestone_stop(global_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(global_cfg)
    exit_after_milestone = bool(cfg.pop("exit_after_milestone", False))
    milestone_steps = cfg.pop("milestone_steps", None)
    if not exit_after_milestone:
        return cfg
    if milestone_steps is None:
        raise ValueError("exit_after_milestone=true requires milestone_steps")

    milestone = int(milestone_steps)
    cfg["save_every"] = milestone
    return cfg

import torch

def get_checkpoint_step(log_dir):
    # 优先读取 last_step.txt（解决延迟问题）
    step_file = Path(log_dir) / "last_step.txt"
    if step_file.exists():
        try:
            return int(step_file.read_text().strip())
        except:
            pass
    # 读不到再读pt
    try:
        ckpt = sorted(Path(log_dir).glob("*.pt"))[-1]
        state = torch.load(ckpt, map_location="cpu")
        return state.get("global_step", 0)
    except:
        return 0

def has_checkpoint(log_dir: Path) -> bool:
    if not log_dir.exists():
        return False
    return any(log_dir.glob("*.pt"))


def make_run_log_dir(base_log_dir: str, run: RunItem, round_idx: int) -> str:
    safe_scene = run.scene.replace("/", "_")
    sparse_tag = str(run.sparse_ratio).replace(".", "p")
    seed_tag = f"_seed{run.seed}" if run.seed is not None else ""
    return f"{base_log_dir}/{safe_scene}_sr{sparse_tag}{seed_tag}_log_{round_idx:03d}"

def main() -> int:
    cli = parse_args()
    plan_path = Path(cli.plan)
    logs_dir = Path(cli.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    plan = load_plan(plan_path)
    global_cfg = apply_milestone_stop(dict(plan.get("global", {})))
    milestone = global_cfg.get("save_every", 0)
    base_log_dir = str(global_cfg.pop("log_dir", r"E:\PycharmProject\mipnerf-pytorch-main\aaa_optimized_version\sparsenerf_only_rank_loss"))
    continue_on_error = bool(plan.get("continue_on_error", False))
    max_rounds = plan.get("max_rounds")

    runs = expand_runs(plan.get("runs", []))
    if max_rounds is not None:
        max_rounds = int(max_rounds)
        if max_rounds < 0:
            raise ValueError("max_rounds must be >= 0")
        runs = runs[:max_rounds]

    if not runs:
        print("No runs to execute. Exiting.")
        return 0

    global_args = build_cli_args(global_cfg)
    failures = 0

    for i, run in enumerate(runs, start=1):
        run_log_dir = make_run_log_dir(base_log_dir, run, i)
        if Path(run_log_dir).exists() and any(Path(run_log_dir).glob("*.pt")):
            step = get_checkpoint_step(run_log_dir)
            if step >= milestone > 0:
                print(f"[跳过] {run_log_dir} 已完成 {step} >= {milestone}")
                continue  # 直接下一轮
        cmd = [
            cli.python,
            "train.py",
            *global_args,
            "--log_dir",
            run_log_dir,
            "--scene",
            run.scene,
            "--sparse_ratio",
            str(run.sparse_ratio),
        ]
        if has_checkpoint(Path(run_log_dir)):
            cmd.append("--continue_training")
        cmd.extend(run.extra_args)

        start = datetime.utcnow()
        log_path = logs_dir / f"run_{i:03d}_{run.scene}.log"
        with log_path.open("w", encoding="utf-8") as f:
            f.write(f"round: {i}\n")
            f.write(f"start_utc: {start.isoformat()}\n")
            if run.seed is not None:
                f.write(f"seed: {run.seed}\n")
            f.write(f"train_log_dir: {run_log_dir}\n")
            f.write(f"auto_continue: {has_checkpoint(Path(run_log_dir))}\n")
            f.write(f"command: {shlex.join(cmd)}\n\n")
            f.flush()

            # ===================== 自动监控里程碑 =====================
            import time
            proc = subprocess.Popen(cmd)

            while proc.poll() is None:
                time.sleep(5)  # 每5秒检查一次
                current_step = get_checkpoint_step(run_log_dir)
                if current_step >= milestone > 0:
                    print(f"\n[✅ 已达到里程碑 {milestone} 步，自动停止]")
                    proc.terminate()
                    proc.wait()
                    break

            return_code = proc.returncode
            # ==========================================================

            end = datetime.utcnow()
            f.write("\n")
            f.write(f"end_utc: {end.isoformat()}\n")
            f.write(f"return_code: {return_code}\n")

        print(f"[{i}/{len(runs)}] scene={run.scene} sparse_ratio={run.sparse_ratio} rc={return_code} log={log_path}")

        if return_code != 0:
            failures += 1
            if not continue_on_error:
                print("Stopping due to failure (continue_on_error=false).")
                return return_code

    if failures:
        print(f"Completed with {failures} failed runs.")
        return 1

    print("All runs completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())