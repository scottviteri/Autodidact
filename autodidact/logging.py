"""
Logging and live dashboard for Autodidact training.

Provides:
- MetricsLogger: Writes every metric to a timestamped JSONL log file.
- DashboardPlotter: Periodically renders a 4x4 matplotlib figure from the log.
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
    
    Every line contains the full metric dict for that step, plus a wall-clock
    timestamp, so we can reconstruct any plot after the fact.
    """

    def __init__(self, method_name: str, log_dir: str = "logs", config: Optional[Dict[str, Any]] = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.method_name = method_name
        self.log_file = self.log_dir / f"{timestamp}_{method_name}.jsonl"

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
      Row 1: Reward (r_k)          | Avg Reward (rho)        | LM Loss              | Eval Perplexity
      Row 2: TD Loss               | Q-value Mean            | Q-value Std           | Soft Value (V_k)
      Row 3: Policy Entropy        | Entropy Ratio           | Selected Action Hist  | Reward Delta (r_k - rho)
      Row 4: Step Time (s)         | Cumulative Reward       | LM Grad Norm          | Q Grad Norm
    
    Saves to a PNG that can be viewed in any image viewer / IDE preview.
    The file is atomically replaced so viewers see a consistent image.
    """

    # Panel layout: (row, col) -> (title, y_key, plot_type)
    # plot_type: "line", "semilogy", "hist", "scatter"
    PANELS = [
        # Row 1
        ("Reward (r_k)",           "reward",               "line"),
        ("Avg Reward (rho)",       "rho",                  "line"),
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
        ("Reward - rho",          "reward_minus_rho",     "line"),
        # Row 4
        ("Step Time (s)",         "step_time",            "line"),
        ("Cumulative Reward",     "cumulative_reward",    "line"),
        ("LM Grad Norm",         "lm_grad_norm",         "line"),
        ("Q Grad Norm",          "q_grad_norm",          "line"),
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
                else:  # "line"
                    steps = [m["step"] for m in metrics if key in m]
                    vals = [m[key] for m in metrics if key in m]
                    if steps:
                        ax.plot(steps, vals, color=color, label=label, linewidth=1.0, alpha=0.8)

                if ptype != "hist":
                    ax.set_xlabel("Step", fontsize=8)

            if len(all_runs) > 1:
                ax.legend(fontsize=7, loc="best")

        fig.tight_layout(rect=[0, 0, 1, 0.96])

        # Atomic write: write to temp, then rename
        tmp_path = self.output_path + ".tmp.png"
        fig.savefig(tmp_path, dpi=120, bbox_inches="tight")
        os.replace(tmp_path, self.output_path)
        plt.close(fig)

