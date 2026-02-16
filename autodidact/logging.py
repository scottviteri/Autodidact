"""
Logging and live dashboard for Autodidact training.

Provides:
- MetricsLogger: Writes every metric to a timestamped JSONL log file.
  Each logger also owns a per-run dashboard PNG (same name as the log, .png).
- DashboardPlotter: Renders a 4x4 matplotlib figure from one or more logs.
"""

import json
import os
import time
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np


class MetricsLogger:
    """
    Logs every metric to a JSONL file (one JSON object per line).
    
    Each run gets a fresh file under logs/ with a timestamp + method name,
    e.g. logs/20260212_143022_q_learning.jsonl
    
    A matching per-run dashboard is also created:
    e.g. logs/20260212_143022_q_learning.png
    
    Every line contains the full metric dict for that step, plus a wall-clock
    timestamp, so we can reconstruct any plot after the fact.
    """

    def __init__(self, method_name: str, log_dir: str = "logs", config: Optional[Dict[str, Any]] = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.method_name = method_name
        self.log_file = self.log_dir / f"{timestamp}_{method_name}.jsonl"
        # Per-run dashboard: same stem, .png extension
        self.dashboard_file = self.log_dir / f"{timestamp}_{method_name}.png"
        # Per-run checkpoint directory: same stem
        self.checkpoint_dir = self.log_dir / f"{timestamp}_{method_name}_ckpt"

        # Write config as the first line (metadata header)
        header = {
            "_type": "config",
            "method": method_name,
            "timestamp": timestamp,
            "config": config or {},
        }
        with open(self.log_file, "w") as f:
            f.write(json.dumps(header) + "\n")

        print(f"Logging to: {self.log_file}")
        print(f"Dashboard:  {self.dashboard_file}")

    def log(self, metrics: Dict[str, Any]):
        """Append one metrics dict as a JSONL line."""
        row = {
            "_type": "metrics",
            "wall_time": time.time(),
            **metrics,
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(row) + "\n")

    def log_eval(self, metrics: Dict[str, Any]):
        """Append an evaluation metrics dict."""
        row = {
            "_type": "eval",
            "wall_time": time.time(),
            **metrics,
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(row) + "\n")


def _read_log(log_path: str) -> tuple:
    """Read a JSONL log file and return (config_dict, metrics_list, eval_list)."""
    config = {}
    metrics = []
    evals = []
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            t = obj.get("_type", "metrics")
            if t == "config":
                config = obj.get("config", {})
            elif t == "eval":
                evals.append(obj)
            else:
                metrics.append(obj)
    return config, metrics, evals


class DashboardPlotter:
    """
    Renders a 4x4 (16-panel) live dashboard from one or more JSONL log files.
    
    Panels (row-major):
      Row 1: Reward (r_k)          | Topic Similarity        | LM Loss              | Eval Perplexity
      Row 2: TD Loss               | Q-value Mean            | Q-value Std           | Soft Value (V_k)
      Row 3: Policy Entropy        | Entropy Ratio           | Selected Action Hist  | Q-value Min/Max
      Row 4: Step Time (s)         | Q Grad Norm             | LM Grad Norm          | (empty)
    
    Saves to a PNG that can be viewed in any image viewer / IDE preview.
    The file is atomically replaced so viewers see a consistent image.
    """

    # Panel layout: (title, y_key, plot_type)
    # plot_type: "line", "semilogy", "hist", "eval_line", "topic_sim"
    PANELS = [
        # Row 1
        ("Reward (r_k)",           "reward",               "line"),
        ("Topic Similarity",       "best_q_topic_sim",     "topic_sim"),
        ("LM Loss",               "lm_loss",              "line"),
        ("Eval Perplexity",       "eval_perplexity",      "eval_line"),
        # Row 2
        ("TD Loss",               "td_loss",              "semilogy"),
        ("Q-value Mean",          "q_mean",               "line"),
        ("Q-value Std",           "q_std",                "line"),
        ("Soft Value (V_k)",      "soft_value",           "line"),
        # Row 3
        ("Policy Entropy",        "policy_entropy",       "line"),
        ("Entropy / Max Entropy", "policy_entropy_ratio", "line"),
        ("Action Distribution",   "selected_action",      "hist"),
        ("Q Min / Max",           "q_min",                "line"),
        # Row 4
        ("Step Time (s)",         "step_time",            "line"),
        ("Q Grad Norm",          "q_grad_norm",          "line"),
        ("LM Grad Norm",         "lm_grad_norm",         "line"),
        ("Q Max",                "q_max",                "line"),
    ]

    def __init__(self, output_path: str = "dashboard.png"):
        self.output_path = output_path

    def render(self, log_paths: List[str]):
        """
        Read log file(s) and render the dashboard.
        
        Multiple log files are overlaid on the same axes (for comparing methods).
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(4, 4, figsize=(22, 16))
        fig.suptitle("Autodidact Training Dashboard", fontsize=16, fontweight="bold", y=0.98)

        # Collect data from all log files
        all_runs = []
        for path in log_paths:
            if not os.path.exists(path):
                continue
            config, metrics, evals = _read_log(path)
            method = config.get("method_name", os.path.basename(path).rsplit(".", 1)[0])
            # Also try the top-level method field from header
            with open(path, "r") as f:
                first = json.loads(f.readline())
                method = first.get("method", method)
            all_runs.append((method, metrics, evals))

        if not all_runs:
            plt.close(fig)
            return

        colors = plt.cm.tab10(np.linspace(0, 1, max(len(all_runs), 1)))

        for panel_idx, (title, key, ptype) in enumerate(self.PANELS):
            row, col = divmod(panel_idx, 4)
            ax = axes[row][col]
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)

            for run_idx, (method, metrics, evals) in enumerate(all_runs):
                color = colors[run_idx]
                label = method

                if ptype == "eval_line":
                    # Plot from eval entries
                    steps = [e["step"] for e in evals if key in e]
                    vals = [e[key] for e in evals if key in e]
                    if steps:
                        ax.plot(steps, vals, "-o", color=color, label=label, markersize=3, linewidth=1.2)
                elif ptype == "hist":
                    # Histogram of action selections
                    vals = [m[key] for m in metrics if key in m]
                    if vals:
                        ax.hist(vals, bins=range(max(vals) + 2), color=color, alpha=0.6, label=label, edgecolor="black", linewidth=0.5)
                    ax.set_xlabel("Action index", fontsize=8)
                elif ptype == "semilogy":
                    steps = [m["step"] for m in metrics if key in m]
                    vals = [m[key] for m in metrics if key in m]
                    if steps:
                        # Clamp to positive for log scale
                        vals_pos = [max(v, 1e-12) for v in vals]
                        ax.semilogy(steps, vals_pos, color=color, label=label, linewidth=1.0, alpha=0.8)
                elif ptype == "topic_sim":
                    # Topic similarity: best-Q, selected, and pool mean
                    best_steps = [m["step"] for m in metrics if "best_q_topic_sim" in m]
                    best_vals = [m["best_q_topic_sim"] for m in metrics if "best_q_topic_sim" in m]
                    sel_vals = [m["selected_topic_sim"] for m in metrics if "selected_topic_sim" in m]
                    mean_vals = [m["mean_topic_sim"] for m in metrics if "mean_topic_sim" in m]
                    if best_vals:
                        ax.plot(best_steps, best_vals, color="red", alpha=0.3, linewidth=0.5)
                        ax.plot(best_steps[:len(sel_vals)], sel_vals, color="orange", alpha=0.3, linewidth=0.5)
                        if mean_vals:
                            ax.plot(best_steps[:len(mean_vals)], mean_vals, color="gray", alpha=0.5, linewidth=0.8, label="Pool mean")
                        # Smoothed lines
                        sw = 10
                        if len(best_vals) >= sw:
                            kernel = np.ones(sw) / sw
                            ax.plot(best_steps[sw-1:], np.convolve(best_vals, kernel, mode="valid"),
                                    color="red", linewidth=1.8, label="Best-Q")
                        if len(sel_vals) >= sw:
                            kernel = np.ones(sw) / sw
                            ax.plot(best_steps[sw-1:len(sel_vals)], np.convolve(sel_vals, kernel, mode="valid"),
                                    color="orange", linewidth=1.8, label="Selected")
                        ax.set_ylim(-0.1, 1.05)
                    ax.legend(fontsize=7, loc="best")
                else:  # "line"
                    steps = [m["step"] for m in metrics if key in m]
                    vals = [m[key] for m in metrics if key in m]
                    if steps:
                        ax.plot(steps, vals, color=color, label=label, linewidth=1.0, alpha=0.8)

                if ptype != "hist":
                    ax.set_xlabel("Step", fontsize=8)

            if len(all_runs) > 1 and ptype != "topic_sim":
                ax.legend(fontsize=7, loc="best")

        fig.tight_layout(rect=[0, 0, 1, 0.96])

        # Atomic write: write to temp, then rename
        tmp_path = self.output_path + ".tmp.png"
        fig.savefig(tmp_path, dpi=120, bbox_inches="tight")
        os.replace(tmp_path, self.output_path)
        plt.close(fig)


class LangevinRAGDashboard:
    """
    Renders a 5x4 (20-panel) dashboard tailored for the Langevin-RAG pipeline.

    Panels (row-major):
      Row 1 — Training quality:
        Reward               | Topic Similarity      | LM Loss              | Eval Perplexity
      Row 2 — Q-learning:
        TD Loss              | SGLD Q vs Random Q    | SGLD Q Gain          | Soft Value
      Row 3 — SGLD health:
        SGLD Grad Norm       | Grad Clip Fraction    | Embed Cosine Sim     | Snap Cosine Sim
      Row 4 — RAG + diversity:
        RAG Top-1 Score      | Num Retrieved         | Token Jaccard Sim    | SGLD Q Best
      Row 5 — Timing + resources:
        Time Breakdown        | Time per Step         | GPU Peak Memory      | LM / Q Grad Norms
    """

    def __init__(self, output_path: str = "dashboard.png"):
        self.output_path = output_path

    def render(self, log_paths: List[str]):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(5, 4, figsize=(24, 22))
        fig.suptitle("Langevin-RAG Training Dashboard", fontsize=16, fontweight="bold", y=0.98)

        all_runs = []
        first_config = {}
        for path in log_paths:
            if not os.path.exists(path):
                continue
            config, metrics, evals = _read_log(path)
            if not first_config:
                first_config = config
            method = config.get("method_name", os.path.basename(path).rsplit(".", 1)[0])
            with open(path, "r") as f:
                first = json.loads(f.readline())
                method = first.get("method", method)
            all_runs.append((method, metrics, evals))

        if not all_runs:
            plt.close(fig)
            return

        colors = plt.cm.tab10(np.linspace(0, 1, max(len(all_runs), 1)))

        def _smooth(vals, window=20):
            if len(vals) < window:
                return vals
            kernel = np.ones(window) / window
            return np.convolve(vals, kernel, mode="valid").tolist()

        def _plot_line(ax, metrics, key, color, label, alpha=0.3, smooth_alpha=1.0, window=20):
            steps = [m["step"] for m in metrics if key in m]
            vals = [m[key] for m in metrics if key in m]
            if not steps:
                return
            ax.plot(steps, vals, color=color, alpha=alpha, linewidth=0.5)
            if len(vals) >= window:
                smoothed = _smooth(vals, window)
                ax.plot(steps[window - 1:], smoothed, color=color, linewidth=1.8, alpha=smooth_alpha, label=label)
            else:
                ax.plot(steps, vals, color=color, linewidth=1.2, alpha=smooth_alpha, label=label)

        for run_idx, (method, metrics, evals) in enumerate(all_runs):
            c = colors[run_idx]
            steps = [m["step"] for m in metrics]

            # ===== Row 1: Training quality =====

            # (0,0) Reward
            ax = axes[0][0]
            _plot_line(ax, metrics, "reward", c, method)

            # (0,1) Topic Similarity (best-Q query + retrieved vs held-out)
            ax = axes[0][1]
            best_q_sims = [m["best_q_topic_sim"] for m in metrics if "best_q_topic_sim" in m]
            mean_q_sims = [m["mean_query_topic_sim"] for m in metrics if "mean_query_topic_sim" in m]
            best_r_sims = [m["best_retrieved_topic_sim"] for m in metrics if "best_retrieved_topic_sim" in m]
            mean_r_sims = [m["mean_retrieved_topic_sim"] for m in metrics if "mean_retrieved_topic_sim" in m]
            sim_steps = [m["step"] for m in metrics if "best_q_topic_sim" in m]
            if best_q_sims:
                ax.plot(sim_steps, best_q_sims, color="red", alpha=0.3, linewidth=0.5)
                if len(best_q_sims) >= 20:
                    ax.plot(sim_steps[19:], _smooth(best_q_sims, 20), color="red", linewidth=1.8, label="Best-Q query")
            if best_r_sims:
                ax.plot(sim_steps[:len(best_r_sims)], best_r_sims, color="green", alpha=0.3, linewidth=0.5)
                if len(best_r_sims) >= 20:
                    ax.plot(sim_steps[19:len(best_r_sims)], _smooth(best_r_sims, 20), color="green", linewidth=1.8, label="Best retrieved")
            if mean_r_sims:
                ax.plot(sim_steps[:len(mean_r_sims)], mean_r_sims, color="gray", alpha=0.5, linewidth=0.8, label="Mean retrieved")
            ax.set_ylim(-0.1, 1.05)
            ax.legend(fontsize=6, loc="best")

            # (0,2) LM Loss
            ax = axes[0][2]
            _plot_line(ax, metrics, "lm_loss", c, method)

            # (0,3) Eval Perplexity
            ax = axes[0][3]
            eval_steps = [e["step"] for e in evals if "eval_perplexity" in e]
            eval_ppls = [e["eval_perplexity"] for e in evals if "eval_perplexity" in e]
            if eval_steps:
                ax.plot(eval_steps, eval_ppls, "-o", color=c, markersize=4, linewidth=1.5, label=method)

            # ===== Row 2: Q-learning =====

            # (1,0) TD Loss
            ax = axes[1][0]
            td_steps = [m["step"] for m in metrics if "td_loss" in m]
            td_vals = [max(m["td_loss"], 1e-12) for m in metrics if "td_loss" in m]
            if td_steps:
                ax.semilogy(td_steps, td_vals, color=c, alpha=0.4, linewidth=0.5)
                if len(td_vals) >= 20:
                    smoothed = _smooth([math.log(v) for v in td_vals], 20)
                    ax.semilogy(td_steps[19:], [math.exp(v) for v in smoothed], color=c, linewidth=1.8, label=method)

            # (1,1) SGLD Q vs Random Q (the key unbiased comparison)
            ax = axes[1][1]
            sgld_q = [m["sgld_q_final_mean"] for m in metrics if "sgld_q_final_mean" in m]
            rand_q = [m["q_random_mean"] for m in metrics if "q_random_mean" in m]
            if sgld_q:
                ax.plot(steps[:len(sgld_q)], sgld_q, color="red", alpha=0.3, linewidth=0.5)
                if len(sgld_q) >= 20:
                    ax.plot(steps[19:len(sgld_q)], _smooth(sgld_q, 20), color="red", linewidth=1.8, label="SGLD Q")
            if rand_q:
                ax.plot(steps[:len(rand_q)], rand_q, color="blue", alpha=0.3, linewidth=0.5)
                if len(rand_q) >= 20:
                    ax.plot(steps[19:len(rand_q)], _smooth(rand_q, 20), color="blue", linewidth=1.8, label="Random Q")

            # (1,2) SGLD Q Gain (final - init per step)
            ax = axes[1][2]
            _plot_line(ax, metrics, "sgld_q_gain", c, method)
            ax.axhline(y=0, color="black", linestyle="--", alpha=0.4)

            # (1,3) Soft Value
            ax = axes[1][3]
            _plot_line(ax, metrics, "soft_value", c, method)

            # ===== Row 3: SGLD health =====

            # (2,0) SGLD Grad Norm
            ax = axes[2][0]
            _plot_line(ax, metrics, "sgld_grad_norm_mean", c, method)

            # (2,1) Grad Clip Fraction
            ax = axes[2][1]
            _plot_line(ax, metrics, "sgld_grad_clip_frac", c, method)
            ax.set_ylim(-0.05, 1.05)

            # (2,2) Embedding Pairwise Cosine Similarity (lower = more diverse)
            ax = axes[2][2]
            _plot_line(ax, metrics, "sgld_embed_cosine_mean", c, method)

            # (2,3) SGLD Snap Cosine Similarity (embedding -> nearest token)
            ax = axes[2][3]
            _plot_line(ax, metrics, "sgld_snap_cosine_mean", c, method)

            # ===== Row 4: RAG + diversity =====

            # (3,0) RAG Top-1 Score
            ax = axes[3][0]
            _plot_line(ax, metrics, "rag_avg_top1_score", c, method)

            # (3,1) Num Retrieved (unique)
            ax = axes[3][1]
            _plot_line(ax, metrics, "num_retrieved", c, method, alpha=0.6, smooth_alpha=1.0, window=1)

            # (3,2) Token Jaccard Similarity (lower = more diverse)
            ax = axes[3][2]
            _plot_line(ax, metrics, "sgld_token_jaccard_mean", c, method)

            # (3,3) SGLD Q Best
            ax = axes[3][3]
            _plot_line(ax, metrics, "sgld_q_best", c, method)

            # ===== Row 5: Timing + resources =====

            # (4,0) Time Breakdown (stacked area)
            ax = axes[4][0]
            t_sgld = [m.get("time_sgld_s", 0) for m in metrics]
            t_rag = [m.get("time_rag_s", 0) for m in metrics]
            t_lm = [m.get("time_lm_s", 0) for m in metrics]
            t_rew = [m.get("time_reward_s", 0) for m in metrics]
            if steps and t_sgld:
                n = min(len(steps), len(t_sgld), len(t_rag), len(t_lm), len(t_rew))
                ax.stackplot(
                    steps[:n],
                    t_sgld[:n], t_rag[:n], t_lm[:n], t_rew[:n],
                    labels=["SGLD", "RAG", "LM Train", "Reward"],
                    colors=["#e74c3c", "#3498db", "#2ecc71", "#f39c12"],
                    alpha=0.7,
                )

            # (4,1) Step Time
            ax = axes[4][1]
            _plot_line(ax, metrics, "step_time", c, method)

            # (4,2) GPU Peak Memory
            ax = axes[4][2]
            _plot_line(ax, metrics, "gpu_peak_mem_gb", c, method, alpha=0.6, smooth_alpha=1.0, window=1)
            ax.set_ylabel("GB", fontsize=8)

            # (4,3) LM + Q Grad Norms
            ax = axes[4][3]
            lm_gn = [m["lm_grad_norm"] for m in metrics if "lm_grad_norm" in m]
            q_gn = [m["q_grad_norm"] for m in metrics if "q_grad_norm" in m]
            if lm_gn:
                ax.plot(steps[:len(lm_gn)], lm_gn, color="green", alpha=0.3, linewidth=0.5)
                if len(lm_gn) >= 20:
                    ax.plot(steps[19:len(lm_gn)], _smooth(lm_gn, 20), color="green", linewidth=1.5, label="LM grad")
            if q_gn:
                ax.plot(steps[:len(q_gn)], q_gn, color="purple", alpha=0.3, linewidth=0.5)
                if len(q_gn) >= 20:
                    ax.plot(steps[19:len(q_gn)], _smooth(q_gn, 20), color="purple", linewidth=1.5, label="Q grad")

        # --- Titles and formatting ---
        titles = [
            # Row 1
            "Reward (r_k)", "Topic Similarity to Held-Out", "LM Loss", "Eval Perplexity",
            # Row 2
            "TD Loss", "SGLD Q vs Random Q", "SGLD Q Gain (final-init)", "Soft Value (V_k)",
            # Row 3
            "SGLD Grad Norm", "Grad Clip Fraction", "Embed Cosine Sim (lower=diverse)", "SGLD Snap Cosine Sim",
            # Row 4
            "RAG Top-1 Score", "Num Retrieved", "Token Jaccard (lower=diverse)", "SGLD Best Q",
            # Row 5
            "Time Breakdown (s)", "Step Time (s)", "GPU Peak Memory (GB)", "LM / Q Grad Norms",
        ]
        for idx, title in enumerate(titles):
            r, c = divmod(idx, 4)
            ax = axes[r][c]
            ax.set_title(title, fontsize=9, fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)
            if idx not in (8,):  # skip stacked area x-label
                ax.set_xlabel("Step", fontsize=7)
            if ax.get_legend_handles_labels()[1]:
                ax.legend(fontsize=7, loc="best")

        # --- Reference lines: expected max cosine sim for random vectors ---
        # E[max cos(q, v_i)] ~ sqrt(2 * ln(N) / d) for N unit vectors in R^d
        random_max_cos_rag = first_config.get("random_max_cos_rag")
        random_max_cos_snap = first_config.get("random_max_cos_snap")
        rag_embed_dim = first_config.get("rag_embed_dim")
        rag_index_size = first_config.get("rag_index_size")
        wte_hidden_dim = first_config.get("wte_hidden_dim")
        wte_vocab_size = first_config.get("wte_vocab_size")

        if random_max_cos_rag is not None:
            ax = axes[3][0]  # RAG Top-1 Score
            ax.axhline(
                y=random_max_cos_rag, color="gray", linestyle="--", alpha=0.7, linewidth=1.2,
                label=f"Random baseline (d={rag_embed_dim}, N={rag_index_size}): {random_max_cos_rag:.3f}",
            )
            ax.legend(fontsize=6, loc="best")

        if random_max_cos_snap is not None:
            ax = axes[2][3]  # SGLD Snap Cosine Sim
            ax.axhline(
                y=random_max_cos_snap, color="gray", linestyle="--", alpha=0.7, linewidth=1.2,
                label=f"Random baseline (d={wte_hidden_dim}, N={wte_vocab_size}): {random_max_cos_snap:.3f}",
            )
            ax.legend(fontsize=6, loc="best")

        fig.tight_layout(rect=[0, 0, 1, 0.96])

        tmp_path = self.output_path + ".tmp.png"
        fig.savefig(tmp_path, dpi=120, bbox_inches="tight")
        os.replace(tmp_path, self.output_path)
        plt.close(fig)
