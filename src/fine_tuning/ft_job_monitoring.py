from __future__ import annotations

import json
import os
import re
import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import matplotlib.pyplot as plt

# seaborn è opzionale, non specificare colori come da best practice
try:
    import seaborn as sns  # type: ignore
    _HAS_SEABORN = True
except ModuleNotFoundError:
    _HAS_SEABORN = False

# ---------------------------------------------------------------------------
# watcher.py – polling + raw collection
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class _RawJobArtifacts:
    """Container for everything fetched from the API."""
    job: Dict[str, Any]
    events: List[Dict[str, Any]]
    checkpoints: List[Dict[str, Any]]


def _poll_job_until_done(client, job_id: str, poll_secs: int = 30) -> Dict[str, Any]:
    """Poll *job_id* ogni *poll_secs* finché non è in stato terminale."""
    terminal = {"succeeded", "failed", "cancelled"}
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id).model_dump()
        status = job.get("status")
        logging.info("FT-job %s status → %s", job_id, status)
        if status in terminal:
            return job
        time.sleep(poll_secs)


def _collect_artifacts(client, job_id: str) -> _RawJobArtifacts:
    """Retrieve job, events and checkpoints una volta che il job è terminato."""
    job = client.fine_tuning.jobs.retrieve(job_id).model_dump()
    events = [e.model_dump() for e in client.fine_tuning.jobs.list_events(
        fine_tuning_job_id=job_id, limit=1000).data]
    ckpts = [c.model_dump() for c in client.fine_tuning.jobs.list_checkpoints(
        fine_tuning_job_id=job_id, limit=1000).data]
    return _RawJobArtifacts(job, events, ckpts)

# ---------------------------------------------------------------------------
# extractor.py – from raw → pandas.DataFrame
# ---------------------------------------------------------------------------
def _events_df(events: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(events)

def _checkpoints_df(ckpts: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(ckpts)

def _metrics_df(events: List[Dict[str, Any]]) -> pd.DataFrame:
    """Extract *train/valid* loss metrics from event list into tidy frame."""
    records: List[Dict[str, Any]] = []
    for evt in events:
        if evt.get("type") == "metrics":
            data = evt.get("data", {})
            step = data.get("step")
            if step is None:
                continue
            records.append({
                "step": step,
                "train_loss": data.get("train_loss"),
                "valid_loss": data.get("valid_loss"),
                "full_valid_loss": data.get("full_valid_loss"),
            })
    return pd.DataFrame(records).sort_values("step", ascending=True)

# ---------------------------------------------------------------------------
# visualizer.py – pure plotting helpers
# ---------------------------------------------------------------------------
def _init_style():
    if _HAS_SEABORN:
        sns.set_theme(style="whitegrid")  # usa solo default

def plot_training_curve(df: pd.DataFrame, *, ax: plt.Axes | None = None) -> plt.Figure:
    """Return a Matplotlib *Figure* with train/valid loss vs. step."""
    _init_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    if df.empty:
        ax.text(0.5, 0.5, "No metrics", ha="center", va="center", fontsize=14)
        return fig

    ax.plot(df["step"], df["train_loss"], "-o", label="train_loss")
    if df["valid_loss"].notna().any():
        ax.plot(df["step"], df["valid_loss"], "-o", label="valid_loss")
    if df["full_valid_loss"].notna().any():
        ax.plot(df["step"], df["full_valid_loss"], "-o", label="full_valid_loss")

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.set_title("Training losses vs step")
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.4)
    fig.tight_layout()
    return fig

# ---------------------------------------------------------------------------
# reporter.py – disk I/O + JSON / CSV / PNG
# ---------------------------------------------------------------------------
def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _write_json(obj: Any, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))

def _write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)

def _write_png(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150)
    plt.close(fig)

# ---------------------------------------------------------------------------
# public façade
# ---------------------------------------------------------------------------
def monitor_and_report(
    *,
    client,
    job_id: str,
    out_dir: str | Path = "result/ft_reports",
    poll_secs: int = 60,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Poll a fine-tuning *job_id* until completion, then export a report.
    Filenames prefissate con il nome del modello fine-tuned."""
    logging.info("Starting monitor for FT-job %s", job_id)

    # 1) Wait for job to finish
    _poll_job_until_done(client, job_id, poll_secs)

    # 2) Fetch everything
    raw = _collect_artifacts(client, job_id)

    # 3) Convert to DataFrame
    df_events = _events_df(raw.events)
    df_ckpts  = _checkpoints_df(raw.checkpoints)
    df_metrics = _metrics_df(raw.events)

    # 4) Determine model name and sanitize per filesystem
    model_name = raw.job.get("model", job_id)
    safe_name = re.sub(r"[^A-Za-z0-9_-]", "_", model_name)

    # 5) Prepare output directory named dal modello
    odir = _ensure_dir(Path(out_dir) / safe_name)

    # 6) Write files con prefisso model_name
    _write_json(raw.job,      odir / f"job_{safe_name}.json")
    _write_csv(df_events,     odir / f"events_{safe_name}.csv")
    _write_csv(df_ckpts,      odir / f"checkpoints_{safe_name}.csv")
    _write_csv(df_metrics,    odir / f"metrics_{safe_name}.csv")

    # 7) Plot & save curve
    fig = plot_training_curve(df_metrics)
    _write_png(fig,           odir / f"training_curve_{safe_name}.png")

    logging.info("Report for model %s saved to %s", model_name, odir)
    return df_events, df_ckpts, df_metrics
