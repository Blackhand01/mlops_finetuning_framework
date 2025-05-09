# src/pipeline_automatic.py
"""
Automatic pipeline: upload dataset → fine-tune (wait+diagnosis) → evaluation (diagnosis).
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from evaluation.openai_evaluation_manager import EvaluationManager
from file_management.openai_file_manager import OpenAIFileManager
from fine_tuning.openai_fine_tuning_manager import FineTuningManager
from fine_tuning.ft_job_monitoring import monitor_and_report

# --------------------------------------------------------------------------- #
# Path helpers                                                                #
# --------------------------------------------------------------------------- #

def parse_base_dir(path: str) -> Tuple[str, str, str]:
    """
    Extract lang_pair and method from any path containing
    'lang_pairs/<lang_pair>/3_fineTuning/<method>'.
    Returns (lang_pair, method, resolved_path).
    """
    p = Path(path).resolve()
    parts: List[str] = list(p.parts)
    try:
        lp_idx = parts.index("lang_pairs") + 1
        lang_pair = parts[lp_idx]
        method    = p.name
        return lang_pair, method, str(p)
    except (ValueError, IndexError):
        raise RuntimeError(f"Cannot extract lang_pair/method from: {path}")

def discover_latest_version_dir(root: str) -> str:
    """
    Scan root for subfolders named 'v<N>' and return the one with highest N.
    """
    root_p = Path(root)
    versions = [
        (int(m.group(1)), d)
        for d in root_p.iterdir()
        if d.is_dir() and (m := re.match(r"v(\d+)$", d.name))
    ]
    if not versions:
        raise RuntimeError(f"No 'v*' subfolders in {root}")
    _, latest = sorted(versions)[-1]
    return str(latest)

_PATTERNS = {
    "train": re.compile(r"(?i).*?(train|training).*\.jsonl$"),
    "valid": re.compile(r"(?i).*?(valid|validation).*\.jsonl$"),
    "eval":  re.compile(r"(?i).*?(eval|evaluation).*\.jsonl$"),
}

def collect_dataset_files(version_dir: str) -> Dict[str, str]:
    """
    Find exactly one .jsonl per split inside version_dir using PATTERNS.
    Raises if missing or duplicated.
    """
    mapping: Dict[str, str] = {}
    entries = os.listdir(version_dir)
    for typ, pattern in _PATTERNS.items():
        matches = [f for f in entries if pattern.match(f)]
        if not matches:
            raise RuntimeError(f"Missing {typ}.*.jsonl in {version_dir}")
        if len(matches) > 1:
            raise RuntimeError(f"Multiple {typ} files in {version_dir}: {matches}")
        mapping[typ] = os.path.join(version_dir, matches[0])
    return mapping

# --------------------------------------------------------------------------- #
# Main pipeline                                                               #
# --------------------------------------------------------------------------- #

def run_automatic_pipeline(config: Dict[str, Any], client) -> None:
    # --- dataset discovery ---
    base_dir = config["dataset"]["base_dir"]
    lang_pair, method, root = parse_base_dir(base_dir)
    print(f"📂 lang_pair={lang_pair}, method={method}, root={root}")

    version_dir = discover_latest_version_dir(root)
    version = Path(version_dir).name
    print(f"🔢 Latest version: {version} (dir: {version_dir})")

    ds_files = collect_dataset_files(version_dir)
    print("🗄 Found files:", ds_files)

    # --- upload files ---
    fm = OpenAIFileManager(client)
    uploaded: Dict[str, str] = {}
    for typ, path in ds_files.items():
        resp = fm.upload_file(path=path, purpose="fine-tune", check_jsonl=True)
        fid = resp[0]["id"] if isinstance(resp, list) else resp["id"]
        uploaded[typ] = fid
        print(f"✅ {typ} uploaded → file_id={fid}")

    # --- launch fine-tuning job ---
    ft_cfg = config["fine_tuning"]
    suffix = ft_cfg.get(
        "suffix_template",
        "{lang_pair}-translator-{version}-{method}"
    ).format(
        lang_pair=lang_pair.upper(),
        version=version,
        method=method
    )

    ftm = FineTuningManager(client)
    job = ftm.create_fine_tuning_job(
        model=ft_cfg.get("base_model", "gpt-4o-2024-08-06"),
        training_file=uploaded["train"],
        validation_file=uploaded["valid"],
        suffix=suffix,
        method={ "type": method, method: {"hyperparameters": ft_cfg["hyperparameters"]} },
        metadata={ "source": "auto", "lang_pair": lang_pair, "version": version },
        seed=ft_cfg["hyperparameters"].get("seed")
    )
    jid = job.get("id", "N/A")
    status = job.get("status", "unknown")
    print(f"🤖 Fine-tune started: ID={jid} | suffix={suffix} | status={status}")

    # --- wait for fine-tuning to finish ---
    terminal_states = {"succeeded", "failed", "cancelled"}
    print("⌛ Waiting for fine-tuning to complete…")
    while status not in terminal_states:
        time.sleep(120)
        job = ftm.retrieve_fine_tuning_job(jid)
        status = job.get("status", "unknown")
        print(f"⚙️  Status: {status}")
    print(f"✅ Fine-tuning completed with status='{status}'")

    # --- diagnose fine-tuning failure ---
    if status != "succeeded":
        print("❌ FINE-TUNING FAILED – launching diagnosis:")
        print(json.dumps(job, indent=2, ensure_ascii=False))

        # events
        try:
            events = ftm.list_fine_tuning_events(jid)
            print(f"⚠️ Events ({len(events)}):")
            for e in events:
                ts  = e.get("timestamp")
                lvl = e.get("level")
                msg = e.get("message")
                print(f"  • [{ts}] {lvl} — {msg}")
        except Exception as e:
            print("Error retrieving events:", e)

        # checkpoints
        try:
            ckpts = ftm.list_fine_tuning_checkpoints(jid)
            print(f"⚠️ Checkpoints ({len(ckpts)}):")
            for c in ckpts:
                print(f"  • ID={c.get('id')} status={c.get('status')}")
        except Exception as e:
            print("Error retrieving checkpoints:", e)

        return
    
    # --- monitoring / report ---
    monitor_and_report(client=client, job_id=jid)

    # --- evaluation phase ---
    ev_cfg = config["evaluation"]
    if not ev_cfg.get("enable", False):
        print("⚠️ Evaluation disabled.")
        return

    # load evaluation configs (data_source_config + testing_criteria)
    try:
        with open(ev_cfg["data_source_config_path"], encoding="utf-8") as f:
            data_src_cfg = json.load(f)
        with open(ev_cfg["testing_criteria_path"], encoding="utf-8") as f:
            test_crit = json.load(f)
    except FileNotFoundError as e:
        print(f"⚠️ Skipping evaluation: missing file {e.filename}")
        return

    evm = EvaluationManager(client)
    try:
        ev = evm.create_evaluation(
            name=f"Eval {lang_pair} {version}",
            data_source_config=data_src_cfg,
            testing_criteria=test_crit,
            metadata={ "version": version, "lang_pair": lang_pair },
        )
        ev_id = ev.get("id", "N/A")
        print(f"🧪 Evaluation created: ID={ev_id}")
    except Exception as e:
        print("❌ Error creating evaluation:", e)
        return

    # --- prepare and run evaluation ---
    # 1) inject fine-tuned model into run template
    fine_tuned_model = job.get("model")
    run_cfg_path = Path(ev_cfg["data_source_run_path"])
    try:
        with run_cfg_path.open(encoding="utf-8") as f:
            run_cfg = json.load(f)
    except FileNotFoundError:
        print(f"⚠️ Skipping eval run: run config not found at {ev_cfg['data_source_run_path']}")
        return

    # Update template with actual model name and eval file_id
    run_cfg["model"] = fine_tuned_model
    run_cfg["source"] = {"type": "file_id", "id": uploaded["eval"]}

    try:
        run = evm.create_evaluation_run(ev_id, run_cfg)
        print(f"▶️ Eval run started: model={fine_tuned_model} | ID={run.get('id')}")
    except Exception as e:
        print("❌ Error starting evaluation run:", e)
        return

    # diagnose evaluation outputs
    try:
        items = evm.list_output_items(ev_id, run.get("id"))
        print(f"📊 Output items ({len(items)}):")
        for item in items[:5]:
            print(json.dumps(item, indent=2, ensure_ascii=False))
        if len(items) > 5:
            print("… (truncated)")
    except Exception as e:
        print("❌ Error retrieving output items:", e)

    print("✅ Automatic pipeline done.")
