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
    # Use dataclass defaults as the single source of truth
    _defaults = AutodidactConfig()

    parser = argparse.ArgumentParser(
        description="Autodidact: Curriculum Selection via Normalised Discounted Soft Q-Learning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument("--model_name", type=str, default=_defaults.model_name)

    # Data
    parser.add_argument("--dataset_name", type=str, default=_defaults.dataset_name)
    parser.add_argument("--seq_len", type=int, default=_defaults.seq_len)
    parser.add_argument("--num_candidates", type=int, default=_defaults.num_candidates, help="N: candidates per step (discrete-Q mode)")
    parser.add_argument("--held_out_subset_size", type=int, default=_defaults.held_out_subset_size, help="M: held-out subset size")
    parser.add_argument("--held_out_total_size", type=int, default=_defaults.held_out_total_size, help="|D|: total held-out set")
    parser.add_argument("--topic_coherent", action="store_true",
                        help="Sample held-out set around a random topic anchor (sentence-embedding similarity)")
    parser.add_argument("--topic_pool_size", type=int, default=_defaults.topic_pool_size,
                        help="Number of candidate windows to embed when building topic-coherent held-out set")
    parser.add_argument("--topic_temperature", type=float, default=_defaults.topic_temperature,
                        help="Softmax temperature for topic-coherent sampling (low=tight cluster, high=diffuse)")

    # Batching
    parser.add_argument("--extract_batch_size", type=int, default=_defaults.extract_batch_size)
    parser.add_argument("--eval_batch_size", type=int, default=_defaults.eval_batch_size)

    # Q-learning
    parser.add_argument("--beta", type=float, default=_defaults.beta, help="Boltzmann temperature")
    parser.add_argument("--gamma", type=float, default=_defaults.gamma, help="Discount factor")
    parser.add_argument("--q_lr", type=float, default=_defaults.q_lr, help="Q-network learning rate")
    parser.add_argument("--q_grad_clip", type=float, default=_defaults.q_grad_clip, help="Q-network gradient clip")
    parser.add_argument("--tau", type=float, default=_defaults.tau, help="Polyak averaging rate for target Q-network")

    # LM training
    parser.add_argument("--lm_lr", type=float, default=_defaults.lm_lr, help="LM learning rate")
    parser.add_argument("--grad_clip", type=float, default=_defaults.grad_clip, help="LM gradient clip")

    # Training
    parser.add_argument("--num_steps", type=int, default=_defaults.num_steps)
    parser.add_argument("--log_interval", type=int, default=_defaults.log_interval)
    parser.add_argument("--eval_interval", type=int, default=_defaults.eval_interval)
    parser.add_argument("--dashboard_interval", type=int, default=_defaults.dashboard_interval)
    parser.add_argument("--save_interval", type=int, default=_defaults.save_interval, help="Save checkpoint every N steps")

    # Logging
    parser.add_argument("--log_dir", type=str, default=_defaults.log_dir)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default=_defaults.wandb_project)

    # Device
    parser.add_argument("--device", type=str, default=_defaults.device)

    # Mixture training
    parser.add_argument("--mixture_batch_size", type=int, default=_defaults.mixture_batch_size,
                        help="Mini-batch size for gradient-accumulated mixture training")
    parser.add_argument("--no_q_weighting", action="store_true", default=False,
                        help="Use uniform weights instead of Q-derived Boltzmann weights for LM loss")
    parser.add_argument("--reset_lm_each_step", action="store_true",
                        help="Reset LM weights to initial theta after each step; only Q-network learns across steps")

    # Baselines
    parser.add_argument("--run_baselines", action="store_true")
    parser.add_argument("--baseline_methods", type=str, default=_defaults.baseline_methods)

    # Needle-in-a-haystack experiment
    parser.add_argument("--needle", action="store_true",
                        help="Run the needle-in-a-haystack Q-value validation experiment")
    parser.add_argument("--needle_text", type=str, default=_defaults.needle_text,
                        help="Custom text for the needle (default: repeating proverbs)")

    # Langevin-RAG mode (default)
    parser.add_argument("--langevin_rag", action="store_true", default=_defaults.langevin_rag,
                        help="Use Langevin Q-guided search + RAG retrieval (default)")
    parser.add_argument("--no_langevin_rag", dest="langevin_rag", action="store_false",
                        help="Disable Langevin-RAG; use discrete-Q candidate selection instead")
    parser.add_argument("--langevin_seq_len", type=int, default=_defaults.langevin_seq_len,
                        help="Sequence length for Langevin embedding optimization")
    parser.add_argument("--langevin_num_chains", type=int, default=_defaults.langevin_num_chains,
                        help="K: parallel Langevin chains (= number of output query samples)")
    parser.add_argument("--langevin_burn_in", type=int, default=_defaults.langevin_burn_in,
                        help="Number of Langevin steps before collecting (one sample per chain)")
    parser.add_argument("--langevin_step_size", type=float, default=_defaults.langevin_step_size,
                        help="Langevin step size epsilon")
    parser.add_argument("--langevin_temperature", type=float, default=_defaults.langevin_temperature,
                        help="Langevin sampling temperature (scales the energy)")
    parser.add_argument("--langevin_noise_scale", type=float, default=_defaults.langevin_noise_scale,
                        help="Multiplier on Gaussian noise in Langevin updates")
    parser.add_argument("--langevin_grad_clip", type=float, default=_defaults.langevin_grad_clip,
                        help="Clip embedding gradients per chain")
    parser.add_argument("--langevin_batch_size", type=int, default=_defaults.langevin_batch_size,
                        help="Chains to process in parallel during Langevin energy computation")
    parser.add_argument("--softmax_tau_start", type=float, default=_defaults.softmax_tau_start,
                        help="Softmax bridge temperature at start of SGLD (high = soft/exploratory)")
    parser.add_argument("--softmax_tau_end", type=float, default=_defaults.softmax_tau_end,
                        help="Softmax bridge temperature at end of SGLD (low = sharp/peaked)")
    parser.add_argument("--sampler_type", type=str, default=_defaults.sampler_type,
                        choices=["softmax", "embedding"],
                        help="SGLD sampler: 'softmax' (softmax-relaxed in logit space, default) or 'embedding' (legacy)")
    parser.add_argument("--lm_micro_batch_size", type=int, default=_defaults.lm_micro_batch_size,
                        help="Micro-batch size for gradient-accumulated LM training on retrieved examples")
    parser.add_argument("--rag_embedding_model", type=str, default=_defaults.rag_embedding_model,
                        help="Sentence-transformer model for RAG embeddings")
    parser.add_argument("--rag_index_size", type=int, default=_defaults.rag_index_size,
                        help="Number of dataset windows to index for RAG")
    parser.add_argument("--rag_top_k", type=int, default=_defaults.rag_top_k,
                        help="FAISS candidates per query (scored, then sampled or top-1)")
    parser.add_argument("--rag_sample_from_topk", action="store_true",
                        default=_defaults.rag_sample_from_topk,
                        help="Softmax-sample 1 from top-k per query for diversity (default)")
    parser.add_argument("--no_rag_sample", dest="rag_sample_from_topk", action="store_false",
                        help="Deterministic top-1 retrieval (no sampling)")
    parser.add_argument("--rag_sample_temperature", type=float,
                        default=_defaults.rag_sample_temperature,
                        help="Temperature for softmax sampling over top-k RAG scores")
    parser.add_argument("--rag_ingest_per_step", type=int,
                        default=_defaults.rag_ingest_per_step,
                        help="Rolling refresh: fresh windows consumed per step into RAG ring buffer (0=static index)")

    # Q-head warmup
    parser.add_argument("--q_warmup_steps", type=int, default=_defaults.q_warmup_steps,
                        help="Number of warmup steps: random data + theta reset, only Q learns")

    # Alternating training
    parser.add_argument("--alternating_period", type=int, default=_defaults.alternating_period,
                        help="Alternate reset-theta / regular mode every N steps (0=disabled)")

    # TD-into-theta
    parser.add_argument("--td_into_theta", action="store_true",
                        help="Let TD loss gradients flow into theta (LM weights), shaping representations for Q-prediction")
    parser.add_argument("--td_lambda", type=float, default=_defaults.td_lambda,
                        help="Scaling coefficient for TD gradients flowing into theta (1.0 = full strength)")

    # Q experience replay
    parser.add_argument("--q_replay_buffer_size", type=int, default=_defaults.q_replay_buffer_size,
                        help="Ring buffer capacity for Q-learning experience replay (0=disabled)")
    parser.add_argument("--q_replay_batch_size", type=int, default=_defaults.q_replay_batch_size,
                        help="Mini-batch size for each replay gradient step")
    parser.add_argument("--q_replay_updates_per_step", type=int, default=_defaults.q_replay_updates_per_step,
                        help="Extra Q gradient steps from replay after each online TD update")

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
            langevin_burn_in=config.langevin_burn_in,
            langevin_step_size=config.langevin_step_size,
            langevin_temperature=config.langevin_temperature,
            langevin_noise_scale=config.langevin_noise_scale,
            langevin_grad_clip=config.langevin_grad_clip,
            langevin_batch_size=config.langevin_batch_size,
            softmax_tau_start=config.softmax_tau_start,
            softmax_tau_end=config.softmax_tau_end,
            sampler_type=config.sampler_type,
            lm_micro_batch_size=config.lm_micro_batch_size,
            rag_embedding_model=config.rag_embedding_model,
            rag_index_size=config.rag_index_size,
            rag_top_k=config.rag_top_k,
            rag_sample_from_topk=config.rag_sample_from_topk,
            rag_sample_temperature=config.rag_sample_temperature,
            rag_ingest_per_step=config.rag_ingest_per_step,
            log_interval=config.log_interval,
            eval_interval=config.eval_interval,
            dashboard_interval=config.dashboard_interval,
            save_interval=config.save_interval,
            log_dir=config.log_dir,
            use_wandb=config.use_wandb,
            wandb_project=config.wandb_project,
            device=config.device,
            reset_lm_each_step=config.reset_lm_each_step,
            q_warmup_steps=config.q_warmup_steps,
            td_into_theta=config.td_into_theta,
            td_lambda=config.td_lambda,
            alternating_period=config.alternating_period,
            topic_coherent=config.topic_coherent,
            topic_pool_size=config.topic_pool_size,
            topic_temperature=config.topic_temperature,
            no_q_weighting=config.no_q_weighting,
            q_replay_buffer_size=config.q_replay_buffer_size,
            q_replay_batch_size=config.q_replay_batch_size,
            q_replay_updates_per_step=config.q_replay_updates_per_step,
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
        reset_lm_each_step=config.reset_lm_each_step,
        td_into_theta=config.td_into_theta,
        td_lambda=config.td_lambda,
        topic_coherent=config.topic_coherent,
        topic_pool_size=config.topic_pool_size,
        topic_temperature=config.topic_temperature,
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
