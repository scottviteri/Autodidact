#!/usr/bin/env python3
"""
Entry point for Autodidact: Curriculum Selection via Normalised Discounted Soft Q-Learning.

Each run produces:
    - A timestamped JSONL log file in logs/ with every metric at every step.
    - A matching per-run dashboard PNG in logs/ (same stem as the log file).
    - When --run_baselines is used, a combined dashboard overlaying all methods.

Usage:
    # Run Langevin-RAG curriculum (default, H100-tuned)
    python train.py

    # Run with custom hyperparameters
    python train.py --num_steps 5000 --langevin_steps 200

    # Run the discrete-Q mode instead
    python train.py --no_langevin_rag

    # Run with baselines for comparison
    python train.py --no_langevin_rag --run_baselines --num_steps 1000

    # Use a specific device
    python train.py --device cuda:0
"""

import argparse
import os
import sys

from autodidact.config import AutodidactConfig
from autodidact.trainer import AutodidactTrainer, BaselineTrainer, LangevinRAGTrainer
from autodidact.baselines import RandomSelector, LossBasedSelector, UncertaintyBasedSelector
from autodidact.logging import DashboardPlotter
from autodidact.needle_experiment import NeedleExperiment


def parse_args() -> AutodidactConfig:
    parser = argparse.ArgumentParser(
        description="Autodidact: Curriculum Selection via Normalised Discounted Soft Q-Learning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument("--model_name", type=str, default="gpt2")

    # Data
    parser.add_argument("--dataset_name", type=str, default="openwebtext")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--num_candidates", type=int, default=256, help="N: candidates per step (discrete-Q mode)")
    parser.add_argument("--held_out_subset_size", type=int, default=256, help="M: held-out subset size")
    parser.add_argument("--held_out_total_size", type=int, default=2048, help="|D|: total held-out set")

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

    # Mixture training
    parser.add_argument("--mixture_batch_size", type=int, default=8,
                        help="Mini-batch size for gradient-accumulated mixture training")
    parser.add_argument("--no_q_weighting", action="store_true",
                        help="Use uniform weights instead of Q-derived Boltzmann weights for LM mixture")

    # Baselines
    parser.add_argument("--run_baselines", action="store_true")
    parser.add_argument("--baseline_methods", type=str, default="random,loss_based,uncertainty")

    # Needle-in-a-haystack experiment
    parser.add_argument("--needle", action="store_true",
                        help="Run the needle-in-a-haystack Q-value validation experiment")
    parser.add_argument("--needle_text", type=str, default=None,
                        help="Custom text for the needle (default: repeating proverbs)")

    # Langevin-RAG mode (default)
    parser.add_argument("--langevin_rag", action="store_true", default=True,
                        help="Use Langevin Q-guided search + RAG retrieval (default)")
    parser.add_argument("--no_langevin_rag", dest="langevin_rag", action="store_false",
                        help="Disable Langevin-RAG; use discrete-Q candidate selection instead")
    parser.add_argument("--langevin_seq_len", type=int, default=64,
                        help="Sequence length for Langevin embedding optimization")
    parser.add_argument("--langevin_num_chains", type=int, default=8,
                        help="K: parallel Langevin chains")
    parser.add_argument("--langevin_num_samples", type=int, default=8,
                        help="Total samples to collect from Langevin dynamics")
    parser.add_argument("--langevin_steps", type=int, default=100,
                        help="Total Langevin steps (burn-in + collection)")
    parser.add_argument("--langevin_burn_in", type=int, default=50,
                        help="Discard first N Langevin steps")
    parser.add_argument("--langevin_thin", type=int, default=5,
                        help="Keep every N-th sample after burn-in")
    parser.add_argument("--langevin_step_size", type=float, default=0.01,
                        help="Langevin step size epsilon")
    parser.add_argument("--langevin_temperature", type=float, default=1.0,
                        help="Langevin sampling temperature (scales the energy)")
    parser.add_argument("--langevin_noise_scale", type=float, default=1.0,
                        help="Multiplier on Gaussian noise in Langevin updates")
    parser.add_argument("--langevin_grad_clip", type=float, default=1.0,
                        help="Clip embedding gradients per chain")
    parser.add_argument("--langevin_batch_size", type=int, default=32,
                        help="Chains to process in parallel during Langevin energy computation")
    parser.add_argument("--lm_micro_batch_size", type=int, default=32,
                        help="Micro-batch size for gradient-accumulated LM training on retrieved examples")
    parser.add_argument("--rag_embedding_model", type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Sentence-transformer model for RAG embeddings")
    parser.add_argument("--rag_index_size", type=int, default=50000,
                        help="Number of dataset windows to index for RAG")
    parser.add_argument("--rag_top_k", type=int, default=8,
                        help="Number of examples to retrieve per query")

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
    print("Autodidact: Curriculum Selection via Normalised Discounted Soft Q-Learning")
    print("=" * 80)
    for key, val in vars(config).items():
        print(f"  {key}: {val}")
    print("=" * 80)

    # --- Needle-in-a-haystack experiment (standalone) ---
    if config.needle:
        print("\n--- Needle-in-a-Haystack Q-Value Validation ---\n")
        experiment = NeedleExperiment(
            model_name=config.model_name,
            seq_len=config.seq_len,
            num_candidates=config.num_candidates,
            held_out_size=config.held_out_subset_size,
            extract_batch_size=config.extract_batch_size,
            eval_batch_size=config.eval_batch_size,
            beta=config.beta,
            gamma=config.gamma,
            q_lr=config.q_lr,
            q_grad_clip=config.q_grad_clip,
            tau=config.tau,
            lm_lr=config.lm_lr,
            grad_clip=config.grad_clip,
            num_steps=config.num_steps,
            log_interval=config.log_interval,
            dashboard_interval=config.dashboard_interval,
            log_dir=config.log_dir,
            device=config.device,
            needle_text=config.needle_text,
        )
        experiment.run()
        return

    all_log_files = []

    # --- Langevin-RAG mode (default) ---
    if config.langevin_rag:
        print("\n--- Langevin Q-Guided Search + RAG Retrieval (default) ---\n")
        trainer = LangevinRAGTrainer(
            model_name=config.model_name,
            dataset_name=config.dataset_name,
            seq_len=config.seq_len,
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
            langevin_seq_len=config.langevin_seq_len,
            langevin_num_chains=config.langevin_num_chains,
            langevin_num_samples=config.langevin_num_samples,
            langevin_steps=config.langevin_steps,
            langevin_burn_in=config.langevin_burn_in,
            langevin_thin=config.langevin_thin,
            langevin_step_size=config.langevin_step_size,
            langevin_temperature=config.langevin_temperature,
            langevin_noise_scale=config.langevin_noise_scale,
            langevin_grad_clip=config.langevin_grad_clip,
            langevin_batch_size=config.langevin_batch_size,
            lm_micro_batch_size=config.lm_micro_batch_size,
            rag_embedding_model=config.rag_embedding_model,
            rag_index_size=config.rag_index_size,
            rag_top_k=config.rag_top_k,
            log_interval=config.log_interval,
            eval_interval=config.eval_interval,
            dashboard_interval=config.dashboard_interval,
            save_interval=config.save_interval,
            log_dir=config.log_dir,
            use_wandb=config.use_wandb,
            wandb_project=config.wandb_project,
            device=config.device,
        )
        log_file = trainer.train(num_steps=config.num_steps)
        all_log_files.append(log_file)
        print(f"\nLangevin-RAG run complete.")
        print(f"  Log files: {all_log_files}")
        return

    # --- Discrete-Q mode (--no_langevin_rag) ---
    print("\n--- Discrete-Q Curriculum (--no_langevin_rag) ---\n")
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
        mixture_batch_size=config.mixture_batch_size,
        no_q_weighting=config.no_q_weighting,
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
