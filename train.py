#!/usr/bin/env python3
"""
Entry point for Autodidact: Curriculum Selection via Differential Soft Q-Learning.

Each run produces:
    - A timestamped JSONL log file in logs/ with every metric at every step.
    - A periodically-refreshed dashboard.png (4x4 panel figure) for live monitoring.

Usage:
    # Run Q-learning curriculum (default, H100-tuned)
    python train.py

    # Run with custom hyperparameters
    python train.py --num_steps 5000 --beta 0.5 --num_candidates 32

    # Run with baselines for comparison (overlaid on same dashboard)
    python train.py --run_baselines --num_steps 1000

    # Run with wandb logging
    python train.py --use_wandb --wandb_project my_project

    # Use a specific device
    python train.py --device cuda:0
"""

import argparse
import os
import sys

from autodidact.config import AutodidactConfig
from autodidact.trainer import AutodidactTrainer, BaselineTrainer
from autodidact.baselines import RandomSelector, LossBasedSelector, UncertaintyBasedSelector
from autodidact.logging import DashboardPlotter


def parse_args() -> AutodidactConfig:
    parser = argparse.ArgumentParser(
        description="Autodidact: Curriculum Selection via Differential Soft Q-Learning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument("--model_name", type=str, default="gpt2", help="HuggingFace model name")

    # Data
    parser.add_argument("--dataset_name", type=str, default="openwebtext", help="HuggingFace dataset name")
    parser.add_argument("--seq_len", type=int, default=1024, help="Context window length in tokens")
    parser.add_argument("--num_candidates", type=int, default=64, help="N: candidates per step")
    parser.add_argument("--held_out_subset_size", type=int, default=256, help="M: held-out subset size per step")
    parser.add_argument("--held_out_total_size", type=int, default=4096, help="|D|: total held-out set size")

    # Q-learning
    parser.add_argument("--beta", type=float, default=1.0, help="Boltzmann temperature")
    parser.add_argument("--q_lr", type=float, default=1e-3, help="Q-network learning rate (eta)")
    parser.add_argument("--tau", type=float, default=0.01, help="EMA rate for average reward (rho)")

    # LM training
    parser.add_argument("--lm_lr", type=float, default=5e-5, help="LM learning rate (alpha)")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping threshold (G)")

    # Training
    parser.add_argument("--num_steps", type=int, default=10000, help="Number of training steps")
    parser.add_argument("--log_interval", type=int, default=10, help="Steps between stdout prints")
    parser.add_argument("--eval_interval", type=int, default=100, help="Steps between full evaluations")
    parser.add_argument("--dashboard_interval", type=int, default=50, help="Steps between dashboard refreshes")

    # Logging
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory for JSONL log files")
    parser.add_argument("--dashboard_path", type=str, default="dashboard.png", help="Path for live dashboard PNG")
    parser.add_argument("--use_wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="autodidact", help="Wandb project name")

    # Device
    parser.add_argument("--device", type=str, default=None, help="Device (e.g., cuda:0, cpu)")

    # Baselines
    parser.add_argument("--run_baselines", action="store_true", help="Also run baseline methods")
    parser.add_argument(
        "--baseline_methods",
        type=str,
        default="random,loss_based,uncertainty",
        help="Comma-separated baseline methods",
    )

    args = parser.parse_args()
    return AutodidactConfig(**vars(args))


def get_baseline_selector(name: str):
    selectors = {
        "random": RandomSelector,
        "loss_based": LossBasedSelector,
        "uncertainty": UncertaintyBasedSelector,
    }
    if name not in selectors:
        raise ValueError(f"Unknown baseline: {name}. Available: {list(selectors.keys())}")
    return selectors[name]()


def main():
    config = parse_args()

    # Print config
    print("=" * 80)
    print("Autodidact: Curriculum Selection via Differential Soft Q-Learning")
    print("=" * 80)
    for key, val in vars(config).items():
        print(f"  {key}: {val}")
    print("=" * 80)

    # Collect all log file paths for multi-method dashboard overlay
    all_log_files = []

    # --- Run Q-learning curriculum ---
    print("\n--- Q-Learning Curriculum ---\n")
    trainer = AutodidactTrainer(
        model_name=config.model_name,
        dataset_name=config.dataset_name,
        seq_len=config.seq_len,
        num_candidates=config.num_candidates,
        held_out_subset_size=config.held_out_subset_size,
        held_out_total_size=config.held_out_total_size,
        beta=config.beta,
        q_lr=config.q_lr,
        tau=config.tau,
        lm_lr=config.lm_lr,
        grad_clip=config.grad_clip,
        log_interval=config.log_interval,
        eval_interval=config.eval_interval,
        dashboard_interval=config.dashboard_interval,
        log_dir=config.log_dir,
        dashboard_path=config.dashboard_path,
        use_wandb=config.use_wandb,
        wandb_project=config.wandb_project,
        device=config.device,
    )
    q_log = trainer.train(num_steps=config.num_steps)
    all_log_files.append(q_log)

    # --- Optionally run baselines ---
    if config.run_baselines:
        for method_name in config.baseline_methods.split(","):
            method_name = method_name.strip()
            print(f"\n--- Baseline: {method_name} ---\n")
            selector = get_baseline_selector(method_name)
            baseline_trainer = BaselineTrainer(
                selector=selector,
                model_name=config.model_name,
                dataset_name=config.dataset_name,
                seq_len=config.seq_len,
                num_candidates=config.num_candidates,
                held_out_subset_size=config.held_out_subset_size,
                held_out_total_size=config.held_out_total_size,
                lm_lr=config.lm_lr,
                grad_clip=config.grad_clip,
                log_interval=config.log_interval,
                eval_interval=config.eval_interval,
                dashboard_interval=config.dashboard_interval,
                log_dir=config.log_dir,
                dashboard_path=config.dashboard_path,
                use_wandb=config.use_wandb,
                wandb_project=config.wandb_project,
                device=config.device,
            )
            baseline_log = baseline_trainer.train(
                num_steps=config.num_steps,
                all_log_files=all_log_files,
            )
            all_log_files.append(baseline_log)

    # --- Final combined dashboard ---
    if len(all_log_files) > 1:
        print("\nRendering final combined dashboard...")
        dashboard = DashboardPlotter(output_path=config.dashboard_path)
        dashboard.render(all_log_files)

    print(f"\nAll runs complete.")
    print(f"  Log files: {all_log_files}")
    print(f"  Dashboard: {config.dashboard_path}")


if __name__ == "__main__":
    main()
