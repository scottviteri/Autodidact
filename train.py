#!/usr/bin/env python3
"""
Entry point for Autodidact: Curriculum Selection via Discounted Soft Q-Learning.

Each run produces:
    - A timestamped JSONL log file in logs/ with every metric at every step.
    - A matching per-run dashboard PNG in logs/ (same stem as the log file).
    - When --run_baselines is used, a combined dashboard overlaying all methods.

Usage:
    # Run Q-learning curriculum (default, H100-tuned)
    python train.py

    # Run with custom hyperparameters
    python train.py --num_steps 5000 --beta 0.5 --gamma 0.95

    # Run with baselines for comparison
    python train.py --run_baselines --num_steps 1000

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
        description="Autodidact: Curriculum Selection via Discounted Soft Q-Learning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument("--model_name", type=str, default="gpt2")

    # Data
    parser.add_argument("--dataset_name", type=str, default="openwebtext")
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--num_candidates", type=int, default=256, help="N: candidates per step")
    parser.add_argument("--held_out_subset_size", type=int, default=512, help="M: held-out subset size")
    parser.add_argument("--held_out_total_size", type=int, default=8192, help="|D|: total held-out set")

    # Batching
    parser.add_argument("--extract_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)

    # Q-learning
    parser.add_argument("--beta", type=float, default=1.0, help="Boltzmann temperature")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor")
    parser.add_argument("--q_lr", type=float, default=1e-4, help="Q-network learning rate")
    parser.add_argument("--q_grad_clip", type=float, default=1.0, help="Q-network gradient clip")
    parser.add_argument("--tau", type=float, default=0.01, help="Polyak averaging rate for target Q-network")

    # LM training
    parser.add_argument("--lm_lr", type=float, default=5e-5, help="LM learning rate")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="LM gradient clip")

    # Training
    parser.add_argument("--num_steps", type=int, default=10000)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--dashboard_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=500, help="Save checkpoint every N steps")

    # Logging
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="autodidact")

    # Device
    parser.add_argument("--device", type=str, default=None)

    # Soft mixture training
    parser.add_argument("--soft_mixture", action="store_true",
                        help="Train on policy-weighted mixture of all N candidates instead of one")
    parser.add_argument("--mixture_batch_size", type=int, default=8,
                        help="Mini-batch size for gradient-accumulated mixture training")

    # Baselines
    parser.add_argument("--run_baselines", action="store_true")
    parser.add_argument("--baseline_methods", type=str, default="random,loss_based,uncertainty")

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

    print("=" * 80)
    print("Autodidact: Curriculum Selection via Discounted Soft Q-Learning")
    print("=" * 80)
    for key, val in vars(config).items():
        print(f"  {key}: {val}")
    print("=" * 80)

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
        extract_batch_size=config.extract_batch_size,
        eval_batch_size=config.eval_batch_size,
        beta=config.beta,
        gamma=config.gamma,
        q_lr=config.q_lr,
        q_grad_clip=config.q_grad_clip,
        tau=config.tau,
        lm_lr=config.lm_lr,
        grad_clip=config.grad_clip,
        log_interval=config.log_interval,
        eval_interval=config.eval_interval,
        dashboard_interval=config.dashboard_interval,
        save_interval=config.save_interval,
        log_dir=config.log_dir,
        use_wandb=config.use_wandb,
        wandb_project=config.wandb_project,
        device=config.device,
        soft_mixture=config.soft_mixture,
        mixture_batch_size=config.mixture_batch_size,
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
                eval_batch_size=config.eval_batch_size,
                lm_lr=config.lm_lr,
                grad_clip=config.grad_clip,
                log_interval=config.log_interval,
                eval_interval=config.eval_interval,
                dashboard_interval=config.dashboard_interval,
                save_interval=config.save_interval,
                log_dir=config.log_dir,
                use_wandb=config.use_wandb,
                wandb_project=config.wandb_project,
                device=config.device,
            )
            baseline_log = baseline_trainer.train(
                num_steps=config.num_steps, all_log_files=all_log_files,
            )
            all_log_files.append(baseline_log)

        print("\nRendering combined dashboard...")
        combined_path = os.path.join(config.log_dir, "combined_dashboard.png")
        DashboardPlotter(output_path=combined_path).render(all_log_files)
        print(f"  Combined dashboard: {combined_path}")

    print(f"\nAll runs complete.")
    print(f"  Log files: {all_log_files}")


if __name__ == "__main__":
    main()
