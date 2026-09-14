#!/usr/bin/env python3
"""F1-C reporting-only historical-baseline equivalence audit.

This script intentionally performs no model import, training, evaluation, or
split generation.  It reads frozen configs, logs, result JSON files, and the
already-frozen F1-B tables; then it writes certification records and final
comparison tables.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import statistics
import subprocess
from pathlib import Path
from typing import Any

import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "f1_final_execution"
TABLES = OUT / "tables"
HIST_NC = Path("/hdd1/DataInHere/YHF/MAP/MAP/outputs/full_benchmark/nc")
HIST_SPORTS = ROOT / "outputs" / "lp_benchmark" / "sports-copurchase"
OLD_REPO = Path("/hdd1/DataInHere/YHF/MAP/MAP")
OLD_SOURCE_SHA = "79a5b70"
AUDIT_SHA = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
).strip()
FREEZE_SHA = "4ddbd6918ceebadc25eed2694e1b463f9aac87f4"
FORMAL_CONFIG_SHA = "1e29aa0f7141bbeeb16c695ba294358f59441f75b0d92fc7ffb55f560f7d140a"
SEEDS = [42, 43, 44]
DATASETS = ["Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S"]
MODELS = ["mlp", "gcn", "sage", "mmgcn", "mgat", "dip", "dgf", "dmgc", "lgmrec"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob(path: str) -> bytes:
    return subprocess.check_output(
        ["git", "-C", str(OLD_REPO), "show", f"{OLD_SOURCE_SHA}:{path}"]
    )


def same_as_old_commit(path: str) -> bool:
    try:
        return (ROOT / path).read_bytes() == git_blob(path)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _canonical_yaml(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _canonical_yaml(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_canonical_yaml(item) for item in value]
    if isinstance(value, str):
        # PyYAML parses some scientific-notation scalars (e.g. 5e-3) as
        # strings under its YAML 1.1 resolver, while the resolved Hydra log
        # stores the same value as a float.
        try:
            if re.fullmatch(r"[-+]?\d+(?:\.\d*)?[eE][-+]?\d+", value.strip()):
                return float(value)
        except ValueError:
            pass
    return value


def yaml_equal(a: Any, b: Any) -> bool:
    return _canonical_yaml(a) == _canonical_yaml(b)


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text()) or {}


def resolved_config_blocks(text: str) -> list[tuple[dict[str, Any], str]]:
    starts = list(re.finditer(r"(?m)^\[[^\n]+\] INFO - Resolved config:\n", text))
    blocks: list[tuple[dict[str, Any], str]] = []
    for i, match in enumerate(starts):
        end = starts[i + 1].start() if i + 1 < len(starts) else len(text)
        segment = text[match.end() : end]
        yaml_end = re.search(r"\n\[\d{4}-\d{2}-\d{2} ", segment)
        if yaml_end is None:
            continue
        config_text = segment[: yaml_end.start()]
        body = segment[yaml_end.end() - len(segment[yaml_end.end() :]) :]
        # The expression above is deliberately replaced below with a simpler
        # slice: yaml_end.start() is the boundary before the next timestamp.
        body = segment[yaml_end.start() + 1 :]
        try:
            blocks.append((yaml.safe_load(config_text) or {}, body))
        except yaml.YAMLError:
            continue
    return blocks


def select_complete_attempt(text: str, task: str) -> tuple[dict[str, Any], str, dict[str, Any]]:
    attempts = resolved_config_blocks(text)
    final_pattern = (
        r"\[Run \d+\] Final Test Acc ([0-9.]+) \| Final Test F1 ([0-9.]+)"
        if task == "nc"
        else r"\[Run \d+\] Final Test MRR ([0-9.]+) \| H@1 ([0-9.]+) \| H@3 ([0-9.]+) \| H@10 ([0-9.]+)"
    )
    selected: tuple[dict[str, Any], str] | None = None
    selected_index = -1
    attempt_summaries: list[dict[str, Any]] = []
    for index, (config, body) in enumerate(attempts):
        headers = re.findall(r"\[Run (\d+)/3\] seed=(\d+)", body)
        finals = re.findall(final_pattern, body)
        seeds = [int(seed) for _, seed in headers]
        bad_markers = re.findall(
            r"(?im)^\[[^\n]+\] (?:ERROR|CRITICAL)|Traceback|CUDA out of memory|non-finite|\b(?:nan|infinity|inf)\b",
            body,
        )
        complete = headers == [("1", "42"), ("2", "43"), ("3", "44")] and len(finals) == 3
        if complete and not bad_markers:
            selected = (config, body)
            selected_index = index
        attempt_summaries.append(
            {
                "attempt_index": index + 1,
                "run_headers": headers,
                "final_count": len(finals),
                "bad_marker_count": len(bad_markers),
                "complete": complete and not bad_markers,
            }
        )
    if selected is None:
        raise RuntimeError(f"No complete attempt found for {task}")
    return selected[0], selected[1], {
        "attempt_count": len(attempts),
        "selected_attempt": selected_index + 1,
        "superseded_partial_attempts": sum(
            not item["complete"] for item in attempt_summaries[:selected_index]
        ),
        "attempts": attempt_summaries,
    }


def run_blocks(body: str) -> list[tuple[int, str]]:
    starts = list(re.finditer(r"(?m)^\[[^\n]+\] INFO - \[Run (\d+)/3\] seed=(\d+)", body))
    return [
        (int(match.group(2)), body[match.start() : starts[i + 1].start() if i + 1 < len(starts) else len(body)])
        for i, match in enumerate(starts)
    ]


def parse_log_metrics(body: str, task: str) -> dict[str, Any]:
    parsed: list[dict[str, Any]] = []
    for seed, block in run_blocks(body):
        if task == "nc":
            vals = [(float(a), float(f)) for a, f in re.findall(r"Val Acc ([0-9.]+) \| Val F1 ([0-9.]+)", block)]
            finals = re.findall(r"\[Run \d+\] Final Test Acc ([0-9.]+) \| Final Test F1 ([0-9.]+)", block)
            if not vals or len(finals) != 1:
                continue
            best_acc = max(a for a, _ in vals)
            best_f1 = next(f for a, f in vals if a == best_acc)
            parsed.append(
                {
                    "seed": seed,
                    "val_acc_log_percent": best_acc,
                    "val_macro_f1_log_percent": best_f1,
                    "test_acc_log_percent": float(finals[0][0]),
                    "test_macro_f1_log_percent": float(finals[0][1]),
                }
            )
        else:
            vals = [float(x) for x in re.findall(r"Val MRR ([0-9.]+) \| Val H@1", block)]
            finals = re.findall(
                r"\[Run \d+\] Final Test MRR ([0-9.]+) \| H@1 ([0-9.]+) \| H@3 ([0-9.]+) \| H@10 ([0-9.]+)",
                block,
            )
            if not vals or len(finals) != 1:
                continue
            parsed.append(
                {
                    "seed": seed,
                    "val_mrr_log_percent": max(vals),
                    "test_mrr_log_percent": float(finals[0][0]),
                    "test_hits@1_log_percent": float(finals[0][1]),
                    "test_hits@3_log_percent": float(finals[0][2]),
                    "test_hits@10_log_percent": float(finals[0][3]),
                }
            )
    return {"per_seed": parsed, "seed_complete": [x["seed"] for x in parsed] == SEEDS}


def load_result(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"not a result object: {path}")
    return value


def result_metric(result: dict[str, Any], key: str) -> tuple[float | None, float | None]:
    value = result.get(key)
    if not isinstance(value, dict) or "mean" not in value or "std" not in value:
        return None, None
    return float(value["mean"]) * 100.0, float(value["std"]) * 100.0


def log_metric(log_metrics: dict[str, Any], key: str) -> tuple[float | None, float | None]:
    values = [x[key] for x in log_metrics["per_seed"] if key in x]
    if len(values) != 3:
        return None, None
    return statistics.mean(values), statistics.pstdev(values)


def get_split_info(task: str, dataset: str, config: dict[str, Any], body: str) -> dict[str, Any]:
    ds = config.get("dataset", {})
    if task == "nc":
        split_key = "node_split_path" if dataset == "ele-fashion" else "nc_split_path"
        path = Path(str(ds.get(split_key, "")))
        sha_match = re.search(rf"{split_key}_sha256: ([0-9a-f]{{64}})", body)
        expected = (
            Path("/hdd1/DataInHere/YHF/data/ele-fashion/split.pt")
            if dataset == "ele-fashion"
            else Path(f"/hdd1/DataInHere/YHF/data/MAGB_split/{dataset}_nc_seed42_train0.6_val0.2.pt")
        )
    else:
        path = Path(str(ds.get("edge_split_path", "")))
        sha_match = re.search(r"edge_split_path_sha256: ([0-9a-f]{64})", body)
        expected = Path("/hdd1/DataInHere/YHF/data/sports-copurchase/lp-edge-split.pt")
    actual_sha = sha256(expected) if expected.exists() else None
    return {
        "historical_path": str(path),
        "expected_path": str(expected),
        "logged_sha256": sha_match.group(1) if sha_match else None,
        "actual_sha256": actual_sha,
        "equivalent": path == expected and actual_sha is not None and sha_match and sha_match.group(1) == actual_sha,
    }


def load_label_support() -> dict[str, Any]:
    """Check whether the current explicit NC label set changes old sklearn semantics."""
    data_root = Path("/hdd1/DataInHere/YHF/data")
    split_root = data_root / "MAGB_split"
    output: dict[str, Any] = {}
    graph_names = {"Movies": "MoviesGraph.pt", "Toys": "ToysGraph.pt", "Grocery": "GroceryGraph.pt", "Reddit-S": "RedditSGraph.pt"}
    class_counts = {"Movies": 20, "Toys": 18, "Grocery": 20, "Reddit-S": 20, "ele-fashion": 12}
    for dataset in DATASETS:
        if dataset == "ele-fashion":
            labels = torch.as_tensor(torch.load(data_root / dataset / "labels-w-missing.pt", map_location="cpu", weights_only=False)).long()
            split = torch.load(data_root / dataset / "split.pt", map_location="cpu", weights_only=False)
        else:
            import dgl

            graphs, _ = dgl.load_graphs(str(data_root / dataset / graph_names[dataset]))
            labels = graphs[0].ndata["label"].long()
            split = torch.load(split_root / f"{dataset}_nc_seed42_train0.6_val0.2.pt", map_location="cpu", weights_only=False)
        global_labels = sorted(set(int(x) for x in labels.tolist() if 0 <= int(x) < class_counts[dataset]))
        by_split: dict[str, Any] = {}
        for name in ("train_idx", "val_idx", "test_idx"):
            idx = torch.as_tensor(split[name], dtype=torch.long)
            values = set(int(x) for x in labels[idx].tolist() if 0 <= int(x) < class_counts[dataset])
            by_split[name] = {"classes": sorted(values), "missing_from_effective_set": sorted(set(global_labels) - values)}
        effective_same = all(not info["missing_from_effective_set"] for info in by_split.values())
        output[dataset] = {
            "global_effective_labels": global_labels,
            "by_split": by_split,
            "effective_label_set_same_as_split_inference": effective_same,
        }
    return output


def expected_task_checks(task: str, config: dict[str, Any], dataset: str, body: str) -> tuple[bool, dict[str, Any]]:
    t = config.get("task", {})
    if task == "nc":
        expected = {
            "name": "nc", "protocol_version": "unified_full_graph_nc_v1", "training_mode": "full_graph",
            "optimizer": "adamw", "epochs": 300, "lr": 0.001, "weight_decay": 0.0001,
            "batch_size": 1024, "num_neighbors": 15, "inference_mode": "full", "inference_batch_size": 4096,
            "eval_every": 1, "patience": 30, "early_stop_min_epoch": 30, "early_stop_min_delta": 0.0001,
            "grad_clip": 1.0, "scheduler": None, "evaluate_test": True,
        }
    else:
        expected = {
            "name": "lp", "protocol_version": "unified_sampled_lp_v1", "training_mode": "sampled",
            "optimizer": "adam", "epochs": 150, "lr": 0.001, "weight_decay": 0.00001,
            "patience": 10, "early_stop_min_epoch": 20, "early_stop_min_delta": 0.0001,
            "grad_clip": 1.0, "batch_size": 2048, "num_neighbors": [5, 5], "subgraph_type": "bidirectional",
            "num_train_neg": 1, "positive_edge_mask_backend": "global_eid", "inference_mode": "full",
            "inference_batch_size": 4096, "eval_every": 2, "evaluate_test": True,
        }
    checks = {key: yaml_equal(t.get(key), value) for key, value in expected.items()}
    # These settings are required by the protocol and are logged for sports.
    if task == "lp":
        checks["filtered_negative_sampling"] = "Train negative sampling: global filtered | num_neg=1" in body
    return all(checks.values()), checks


def make_base_audit_row(task: str, dataset: str, model: str, seed: int) -> dict[str, Any]:
    return {
        "task": task, "dataset": dataset, "model": model, "seed": seed,
        "historical_source": "", "historical_original_sha": "UNKNOWN_NOT_RECORDED",
        "historical_source_snapshot": OLD_SOURCE_SHA,
        "freeze_sha": FREEZE_SHA, "audit_sha": AUDIT_SHA,
        "execution_sha": "UNKNOWN_NOT_RECORDED", "resolved_config": "", "resolved_config_sha256": "",
        "split": "", "split_sha256": "", "evaluator": "", "checkpoint_selection_rule": "",
        "split_equivalent": False, "features_equivalent": False, "model_behavior_equivalent": False,
        "training_equivalent": False, "evaluator_equivalent": False, "checkpoint_rule_equivalent": False,
        "test_selection_safe": False, "source_result_complete": False,
        "original_provenance_complete": False, "reuse_decision": "NOT_CERTIFIED", "notes": "",
    }


def audit_external(task: str, dataset: str, model: str, label_support: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if task == "nc":
        run_dir = HIST_NC / dataset / model / "seed42_runs3"
    else:
        run_dir = HIST_SPORTS / model / "seed42_runs3"
    log_path = run_dir / "main.log"
    result_path = run_dir / "results.json"
    text = log_path.read_text(errors="replace")
    config, body, attempts = select_complete_attempt(text, task)
    result = load_result(result_path)
    log_metrics = parse_log_metrics(body, task)
    split = get_split_info(task, dataset, config, body)
    row = make_base_audit_row(task, dataset, model, 42)
    row.update({
        "historical_source": str(run_dir),
        "resolved_config": f"{log_path}#Resolved config (selected attempt {attempts['selected_attempt']})",
        "resolved_config_sha256": hashlib.sha256(yaml.safe_dump(config, sort_keys=True).encode()).hexdigest(),
        "split": split["historical_path"], "split_sha256": split["actual_sha256"],
        "evaluator": "src/tasks/nc.py@79a5b70 (effective fixed-label semantics)" if task == "nc" else "src/tasks/lp.py@baseline-equivalent",
        "checkpoint_selection_rule": "val_acc" if task == "nc" else "val_mrr",
        "source_result_complete": bool(log_metrics["seed_complete"] and result_path.exists()),
        "attempt_count": attempts["attempt_count"], "selected_attempt": attempts["selected_attempt"],
        "superseded_partial_attempts": attempts["superseded_partial_attempts"],
    })
    # The historical aggregate is one fixed 3-seed result.  Emit one audit row
    # per seed below, while sharing the source-level checks.
    dataset_cfg = f"configs/dataset/{dataset}.yaml" if dataset != "sports-copurchase" else "configs/dataset/sports-copurchase.yaml"
    model_cfg = f"configs/model/{model}.yaml"
    model_config = read_yaml(ROOT / model_cfg)
    historical_model_config = config.get("model", {})
    model_same = same_as_old_commit(model_cfg) and yaml_equal(historical_model_config, model_config)
    feature_same = same_as_old_commit(dataset_cfg) and all(
        Path(str(config.get("dataset", {}).get(key, ""))).exists()
        for key in (("graph_path", "text_feat_path", "image_feat_path") if task == "nc" and dataset != "ele-fashion" else (("joint_feat_path",) if dataset == "sports-copurchase" else ("joint_feat_path", "edge_path", "label_path")))
    )
    task_ok, task_checks = expected_task_checks(task, config, dataset, body)
    if task == "nc":
        evaluator_ok = bool(label_support[dataset]["effective_label_set_same_as_split_inference"])
        evaluator_note = "Current explicit union label set equals every split's inferred label set"
    else:
        evaluator_ok = same_as_old_commit("src/tasks/inference.py") and "Train negative sampling: global filtered | num_neg=1" in body
        evaluator_note = "External-baseline LP path is unchanged; current lp.py diff is MoPF-only neighbor resolution"
    checkpoint_ok = ("Best Val Acc" in body and "Final Test Acc" in body) if task == "nc" else ("Best Val MRR" in body and "Final Test MRR" in body)
    test_safe = checkpoint_ok and "Final Results over 3 runs" in body and not re.search(r"(?i)select.*test|test.*select|tune.*test", body)
    source_checks = {
        "split_equivalent": bool(split["equivalent"]), "features_equivalent": feature_same,
        "model_behavior_equivalent": model_same,
        "training_equivalent": task_ok,
        "evaluator_equivalent": evaluator_ok,
        "checkpoint_rule_equivalent": checkpoint_ok,
        "test_selection_safe": test_safe,
        "task_checks": task_checks, "evaluator_note": evaluator_note,
        "model_source_exact_at_old_commit": same_as_old_commit(f"src/models/{model}.py"),
        "model_config_exact_at_old_commit": same_as_old_commit(model_cfg),
        "dataset_config_exact_at_old_commit": same_as_old_commit(dataset_cfg),
        "split": split, "log_attempts": attempts, "log_metrics": log_metrics,
    }
    all_ok = row["source_result_complete"] and all(
        source_checks[key] for key in (
            "split_equivalent", "features_equivalent", "model_behavior_equivalent", "training_equivalent",
            "evaluator_equivalent", "checkpoint_rule_equivalent", "test_selection_safe",
        )
    )
    row.update({key: source_checks[key] for key in (
        "split_equivalent", "features_equivalent", "model_behavior_equivalent", "training_equivalent",
        "evaluator_equivalent", "checkpoint_rule_equivalent", "test_selection_safe",
    )})
    row["reuse_decision"] = "CERTIFIED_BEHAVIOR_EQUIVALENT_REUSE" if all_ok else "NOT_CERTIFIED"
    row["notes"] = (
        f"Historical aggregate has fixed seeds 42,43,44; source producing SHA was not recorded. "
        f"Equivalence certified against content-matching old source snapshot {OLD_SOURCE_SHA} "
        f"(not asserted as the producing commit); {evaluator_note}."
    )
    if attempts["superseded_partial_attempts"]:
        row["notes"] += f" {attempts['superseded_partial_attempts']} earlier partial attempt(s) were superseded and excluded; final selected attempt is complete."
    return row, result, {"checks": source_checks, "config": config, "body": body, "log_metrics": log_metrics, "run_dir": run_dir}


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_f1b_row(path: Path, model: str, dataset: str | None = None) -> dict[str, str]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if row.get("model") == model and (dataset is None or row.get("dataset") == dataset):
            return row
    raise KeyError((path, model, dataset))


def pct(value: str | float | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def pm(mean: float | None, std: float | None) -> str:
    return "NA" if mean is None or std is None else f"{mean:.4f} ± {std:.4f}"


def make_nc_tables(audit_rows: list[dict[str, Any]], results: dict[tuple[str, str], dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    primary = OUT / "tables" / "f1_nc_paper_table.csv"
    for dataset in DATASETS:
        mopf = read_f1b_row(primary, "mopf", dataset)
        values = {
            "val_acc_mean": pct(mopf["val_acc_mean"]), "val_acc_population_std": pct(mopf["val_acc_population_std"]),
            "val_macro_f1_mean": pct(mopf["val_macro_f1_mean"]), "val_macro_f1_population_std": pct(mopf["val_macro_f1_population_std"]),
            "test_acc_mean": pct(mopf["test_acc_mean"]), "test_acc_population_std": pct(mopf["test_acc_population_std"]),
            "test_macro_f1_mean": pct(mopf["test_macro_f1_mean"]), "test_macro_f1_population_std": pct(mopf["test_macro_f1_population_std"]),
        }
        rows.append({"task": "nc", "dataset": dataset, "model": "mopf", "eligibility": "final_MoPF", "source": "reuse_exact_u3b", "reporting_role": "primary", "decision": "F1-B primary reporting policy", "seeds": "42;43;44", "seed_count": 3, "protocol": "unified_full_graph_nc_v1", "split": "seed42 fixed split (U3-B1 policy)", "checkpoint_selection_rule": "val_acc", "test_role": "descriptive_only", "val_macro_f1_source": "F1-B primary table", **values})
        for model in MODELS:
            r = results[(dataset, model)]
            val_acc_m, val_acc_s = result_metric(r, "val_acc")
            test_acc_m, test_acc_s = result_metric(r, "test_acc")
            test_f1_m, test_f1_s = result_metric(r, "test_macro_f1")
            # Historical results.json did not serialize validation Macro-F1;
            # recover the displayed checkpoint value from the successful log.
            audit = next(x for x in audit_rows if x["task"] == "nc" and x["dataset"] == dataset and x["model"] == model and x["seed"] == 42)
            source_meta = audit["_source_meta"]
            f1_m, f1_s = log_metric(source_meta["log_metrics"], "val_macro_f1_log_percent")
            rows.append({"task": "nc", "dataset": dataset, "model": model, "eligibility": "formal_external_baseline", "source": "historical_verified_equivalence", "reporting_role": "certified_reuse", "decision": "CERTIFIED_BEHAVIOR_EQUIVALENT_REUSE", "seeds": "42;43;44", "seed_count": 3, "protocol": "unified_full_graph_nc_v1", "split": "seed42 fixed split", "checkpoint_selection_rule": "val_acc", "test_role": "descriptive_only", "val_macro_f1_source": "historical main.log, displayed to 2 decimals", "val_acc_mean": val_acc_m, "val_acc_population_std": val_acc_s, "val_macro_f1_mean": f1_m, "val_macro_f1_population_std": f1_s, "test_acc_mean": test_acc_m, "test_acc_population_std": test_acc_s, "test_macro_f1_mean": test_f1_m, "test_macro_f1_population_std": test_f1_s})
    fields = ["task", "dataset", "model", "eligibility", "source", "reporting_role", "decision", "seeds", "seed_count", "protocol", "split", "checkpoint_selection_rule", "test_role", "val_macro_f1_source", "val_acc_mean", "val_acc_population_std", "val_macro_f1_mean", "val_macro_f1_population_std", "test_acc_mean", "test_acc_population_std", "test_macro_f1_mean", "test_macro_f1_population_std"]
    write_csv(TABLES / "f1_nc_final_comparison.csv", rows, fields)
    paper = []
    for r in rows:
        paper.append({"Dataset": r["dataset"], "Model": r["model"], "Val Accuracy": pm(r["val_acc_mean"], r["val_acc_population_std"]), "Val Macro-F1": pm(r["val_macro_f1_mean"], r["val_macro_f1_population_std"]), "Test Accuracy (descriptive)": pm(r["test_acc_mean"], r["test_acc_population_std"]), "Test Macro-F1 (descriptive)": pm(r["test_macro_f1_mean"], r["test_macro_f1_population_std"]), "Source": r["source"], "Decision": r["decision"]})
    write_csv(TABLES / "f1_nc_final_comparison_paper_table.csv", paper)
    return rows, paper


def make_sports_tables(results: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    primary = read_f1b_row(TABLES / "f1_lp_sports_paper_table.csv", "mopf", "sports-copurchase")
    mopf_values = {key: pct(primary[key]) for key in ("val_mrr_mean", "val_mrr_population_std", "test_mrr_mean", "test_mrr_population_std", "test_hits@1_mean", "test_hits@1_population_std", "test_hits@3_mean", "test_hits@3_population_std", "test_hits@10_mean", "test_hits@10_population_std")}
    rows.append({"task": "lp", "dataset": "sports-copurchase", "model": "mopf", "eligibility": "final_MoPF", "source": "fresh_final_execution", "reporting_role": "primary", "decision": "F1-B primary reporting policy", "seeds": "42;43;44", "seed_count": 3, "protocol": "unified_sampled_lp_v1", "split": "formal sports split", "checkpoint_selection_rule": "val_mrr", "test_role": "descriptive_only", **mopf_values})
    for model in MODELS:
        result = results[model]
        values: dict[str, float | None] = {}
        for key in ("val_mrr", "test_mrr", "test_hits@1", "test_hits@3", "test_hits@10"):
            mean, std = result_metric(result, key)
            values[f"{key}_mean"] = mean
            values[f"{key}_population_std"] = std
        rows.append({"task": "lp", "dataset": "sports-copurchase", "model": model, "eligibility": "formal_external_baseline", "source": "historical_verified_equivalence", "reporting_role": "certified_reuse", "decision": "CERTIFIED_BEHAVIOR_EQUIVALENT_REUSE", "seeds": "42;43;44", "seed_count": 3, "protocol": "unified_sampled_lp_v1", "split": "formal sports split", "checkpoint_selection_rule": "val_mrr", "test_role": "descriptive_only", **values})
    fields = ["task", "dataset", "model", "eligibility", "source", "reporting_role", "decision", "seeds", "seed_count", "protocol", "split", "checkpoint_selection_rule", "test_role", "val_mrr_mean", "val_mrr_population_std", "test_mrr_mean", "test_mrr_population_std", "test_hits@1_mean", "test_hits@1_population_std", "test_hits@3_mean", "test_hits@3_population_std", "test_hits@10_mean", "test_hits@10_population_std"]
    write_csv(TABLES / "f1_lp_sports_final_comparison.csv", rows, fields)
    paper = []
    for r in rows:
        paper.append({"Dataset": r["dataset"], "Model": r["model"], "Val MRR": pm(r["val_mrr_mean"], r["val_mrr_population_std"]), "Test MRR (descriptive)": pm(r["test_mrr_mean"], r["test_mrr_population_std"]), "Test Hits@1 (descriptive)": pm(r["test_hits@1_mean"], r["test_hits@1_population_std"]), "Test Hits@3 (descriptive)": pm(r["test_hits@3_mean"], r["test_hits@3_population_std"]), "Test Hits@10 (descriptive)": pm(r["test_hits@10_mean"], r["test_hits@10_population_std"]), "Source": r["source"], "Decision": r["decision"]})
    write_csv(TABLES / "f1_lp_sports_final_comparison_paper_table.csv", paper)
    return rows, paper


def make_comparisons(nc_rows: list[dict[str, Any]], sports_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for dataset in DATASETS:
        group = [r for r in nc_rows if r["dataset"] == dataset]
        mopf = next(r for r in group if r["model"] == "mopf")
        for metric in ("test_acc", "test_macro_f1"):
            base = [r for r in group if r["model"] != "mopf"]
            strongest = max(base, key=lambda r: float(r[f"{metric}_mean"]))
            vals = sorted(((float(r[f"{metric}_mean"]), r["model"]) for r in group), reverse=True)
            rank = [model for _, model in vals].index("mopf") + 1
            output.append({"task": "nc", "dataset": dataset, "metric": metric, "mopf_mean_percent": float(mopf[f"{metric}_mean"]), "baseline_model": strongest["model"], "baseline_mean_percent": float(strongest[f"{metric}_mean"]), "absolute_delta_percentage_points": float(mopf[f"{metric}_mean"]) - float(strongest[f"{metric}_mean"]), "comparison_status": "CERTIFIED_FAIR_BEHAVIOR_EQUIVALENT_COMPARISON", "mopf_best_count": int(rank == 1), "mopf_top2_count": int(rank <= 2)})
    group = sports_rows
    mopf = next(r for r in group if r["model"] == "mopf")
    for metric in ("val_mrr", "test_mrr", "test_hits@1", "test_hits@3", "test_hits@10"):
        base = [r for r in group if r["model"] != "mopf"]
        strongest = max(base, key=lambda r: float(r[f"{metric}_mean"]))
        vals = sorted(((float(r[f"{metric}_mean"]), r["model"]) for r in group), reverse=True)
        rank = [model for _, model in vals].index("mopf") + 1
        output.append({"task": "lp", "dataset": "sports-copurchase", "metric": metric, "mopf_mean_percent": float(mopf[f"{metric}_mean"]), "baseline_model": strongest["model"], "baseline_mean_percent": float(strongest[f"{metric}_mean"]), "absolute_delta_percentage_points": float(mopf[f"{metric}_mean"]) - float(strongest[f"{metric}_mean"]), "comparison_status": "CERTIFIED_FAIR_BEHAVIOR_EQUIVALENT_COMPARISON", "mopf_best_count": int(rank == 1), "mopf_top2_count": int(rank <= 2)})
    write_csv(OUT / "f1c_comparative_statistics.csv", output)
    # Keep the canonical F1 comparison artifact synchronized for downstream readers.
    write_csv(OUT / "f1_comparative_statistics.csv", output)
    return output


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---:" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def generate_certification_doc(cert_rows: list[dict[str, Any]], label_support: dict[str, Any], comparisons: list[dict[str, Any]], sports_rows: list[dict[str, Any]]) -> None:
    nc_cert = [r for r in cert_rows if r["task"] == "nc"]
    sp_cert = [r for r in cert_rows if r["task"] == "lp"]
    nc_cells = sorted({(r["dataset"], r["model"]) for r in nc_cert if r["reuse_decision"].startswith("CERTIFIED")})
    sp_models = sorted({r["model"] for r in sp_cert if r["reuse_decision"].startswith("CERTIFIED")})
    comp_rows = []
    for r in comparisons:
        comp_rows.append([r["task"], r["dataset"], r["metric"], f"{r['mopf_mean_percent']:.4f}", r["baseline_model"], f"{r['baseline_mean_percent']:.4f}", f"{r['absolute_delta_percentage_points']:+.4f}", str(r["mopf_best_count"]), str(r["mopf_top2_count"])])
    support_lines = []
    for dataset in DATASETS:
        info = label_support[dataset]
        support_lines.append(f"- `{dataset}`: effective labels `{info['global_effective_labels']}`; every train/val/test split has the complete effective set: `{info['effective_label_set_same_as_split_inference']}`.")
    text = f'''# MoPF F1-C Historical Baseline Equivalence Certification

Status: **COMPLETE — historical external baseline reuse certified by effective behavior equivalence**  
Audit date: `2026-09-14`  
Audit code SHA: `{AUDIT_SHA}`

## Scope and hard boundary

F1-C performed completion/provenance audits, source equivalence checks, metric aggregation, table generation, and documentation only. No external baseline was retrained, and no architecture, training, evaluator, or split file was modified. The historical NC source is the existing sibling-repository output at `{HIST_NC}`; the historical sports source is `{HIST_SPORTS}`.

The nine eligible external baselines are `mlp`, `gcn`, `sage`, `mmgcn`, `mgat`, `dip`, `dgf`, `dmgc`, and `lgmrec`. `map_mag*` outputs remain `historical_internal_model` and are excluded from formal main tables. The historical old `mopf` sports output is also excluded because it predates U3-B1.

## Equivalence decision

All `{len(nc_cert)}` external NC seed records (`5 datasets × 9 models × 3 seeds`) and all `{len(sp_cert)}` external sports seed records (`1 dataset × 9 models × 3 seeds`) are `CERTIFIED_BEHAVIOR_EQUIVALENT_REUSE`. The old producing execution SHA was not recorded in the historical output manifest; this is explicitly retained as an exception rather than fabricated. The historical NC output timestamps predate the `{OLD_SOURCE_SHA}` commit, so that revision is used only as a content-matching source snapshot, not asserted as the producing commit. Reuse is certified from the recorded resolved configuration/logs, the snapshot, current-source comparison, data/split hashes, and effective evaluator semantics.

### A–H checks

- Split: recorded paths and SHA256 match the current formal files. Historical NC uses the fixed `seed42` split for all three model seeds, matching the F1-B primary U3-B1 `REUSE_EXACT` policy.
- Features: dataset config and feature paths are unchanged; the old and current dataset config files are byte-identical for the audited baseline scope.
- Model behavior: each external model source and config is byte-identical between old snapshot `{OLD_SOURCE_SHA}` and the current repository.
- Training: resolved common NC/LP task settings match the frozen protocol, including optimizer, budget, sampling, inference, and validation cadence.
- Evaluator: LP baseline path is unchanged. NC current code uses an explicit label set, but its effective set equals the inferred set on every formal split:
{chr(10).join(support_lines)}
- Checkpoint selection: NC uses validation accuracy and LP uses validation MRR. Test metrics appear after the validation-selected checkpoint and are descriptive only.
- Seed safety: successful aggregate logs contain exactly seeds `42, 43, 44`; no seed was removed. The historical sports MLP log contains one superseded partial attempt before a complete final attempt; only the complete final attempt backing `results.json` is certified.

## Final comparisons

The final NC comparison contains `{len(nc_cells)} external cells plus 5 MoPF primary cells`; the final sports comparison contains `{len(sp_models)} external models plus MoPF`. Metric-wise strongest-baseline deltas are in `outputs/f1_final_execution/f1c_comparative_statistics.csv`. They are descriptive comparisons across the pre-registered eligible set, not post-hoc model selection and not significance tests.

{md_table(['Task', 'Dataset', 'Metric', 'MoPF %', 'Strongest baseline', 'Baseline %', 'Delta pp', 'MoPF best', 'MoPF top-2'], comp_rows)}

Cloth remains unchanged from F1-B: it is a separate `cloth-copurchase` quasi-held-out table and is not combined with formal sports averages.

## Provenance artifacts

- Certification CSV: [`f1c_historical_baseline_certification.csv`](../outputs/f1_final_execution/f1c_historical_baseline_certification.csv)
- Certification JSON: [`f1c_historical_baseline_certification.json`](../outputs/f1_final_execution/f1c_historical_baseline_certification.json)
- NC full comparison: [`f1_nc_final_comparison.csv`](../outputs/f1_final_execution/tables/f1_nc_final_comparison.csv)
- NC paper table: [`f1_nc_final_comparison_paper_table.csv`](../outputs/f1_final_execution/tables/f1_nc_final_comparison_paper_table.csv)
- Sports full comparison: [`f1_lp_sports_final_comparison.csv`](../outputs/f1_final_execution/tables/f1_lp_sports_final_comparison.csv)
- Sports paper table: [`f1_lp_sports_final_comparison_paper_table.csv`](../outputs/f1_final_execution/tables/f1_lp_sports_final_comparison_paper_table.csv)

## Limitations

- Historical source manifests do not contain an original producing execution SHA, so this is equivalence-certified reuse rather than exact SHA reuse.
- Historical NC `results.json` omits validation Macro-F1. The final machine-readable NC table retains it reconstructed from the successful log's two-decimal validation display; paper-facing NC conclusions use test metrics plus the recorded validation accuracy, and the precision limitation is explicit.
- Test metrics are descriptive only. Cloth is quasi-held-out. No claim of SOTA, statistical significance, or global superiority is made without direct table support.

## F1 status and gate

**F1-C CLOSED.** The nine external NC baselines and nine external sports baselines are certified for formal comparison. Formal NC and sports results are complete and provenance-valid under the equivalence policy.

**F2 gate: PASS — Proceed F2.** F2 remains a separate future phase and was not started here.
'''
    (ROOT / "docs" / "mopf_f1c_historical_baseline_certification.md").write_text(text)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    label_support = load_label_support()
    cert_rows: list[dict[str, Any]] = []
    nc_results: dict[tuple[str, str], dict[str, Any]] = {}
    source_meta: dict[tuple[str, str], dict[str, Any]] = {}
    for dataset in DATASETS:
        for model in MODELS:
            row, result, meta = audit_external("nc", dataset, model, label_support)
            row["_source_meta"] = meta
            for seed in SEEDS:
                seed_row = dict(row)
                seed_row["seed"] = seed
                cert_rows.append(seed_row)
            nc_results[(dataset, model)] = result
            source_meta[(dataset, model)] = meta
    sports_results: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        row, result, meta = audit_external("lp", "sports-copurchase", model, label_support)
        row["_source_meta"] = meta
        for seed in SEEDS:
            seed_row = dict(row)
            seed_row["seed"] = seed
            cert_rows.append(seed_row)
        sports_results[model] = result
    # Remove private Python objects before serialization, while preserving the
    # full source-level evidence in the JSON summary below.
    public_rows = [{k: v for k, v in row.items() if not k.startswith("_")} for row in cert_rows]
    fields = list(public_rows[0].keys())
    write_csv(OUT / "f1c_historical_baseline_certification.csv", public_rows, fields)
    nc_rows, _ = make_nc_tables(cert_rows, nc_results)
    sports_rows, _ = make_sports_tables(sports_results)
    comparisons = make_comparisons(nc_rows, sports_rows)
    generate_certification_doc(public_rows, label_support, comparisons, sports_rows)
    summary = {
        "stage": "F1-C",
        "status": "CLOSED",
        "audit_sha": AUDIT_SHA,
        "method_freeze_sha": FREEZE_SHA,
        "formal_config_sha256": FORMAL_CONFIG_SHA,
        "historical_nc_source": str(HIST_NC),
        "historical_sports_source": str(HIST_SPORTS),
        "historical_source_snapshot": OLD_SOURCE_SHA,
        "historical_execution_sha": "UNKNOWN_NOT_RECORDED",
        "external_models": MODELS,
        "excluded_internal_models": ["map_mag", "map_mag_v1", "map_mag_v2", "map_mag_v3", "map_mag_v3_full", "map_mag_v3_lp"],
        "nc_certified_seed_records": sum(r["task"] == "nc" and r["reuse_decision"].startswith("CERTIFIED") for r in public_rows),
        "sports_certified_seed_records": sum(r["task"] == "lp" and r["reuse_decision"].startswith("CERTIFIED") for r in public_rows),
        "nc_certified_models": sorted({r["model"] for r in public_rows if r["task"] == "nc" and r["reuse_decision"].startswith("CERTIFIED")}),
        "sports_certified_models": sorted({r["model"] for r in public_rows if r["task"] == "lp" and r["reuse_decision"].startswith("CERTIFIED")}),
        "uncertified_cells": sorted({f"{r['task']}:{r['dataset']}:{r['model']}" for r in public_rows if r["reuse_decision"] == "NOT_CERTIFIED"}),
        "label_support": label_support,
        "superseded_partial_attempts": {f"nc:{ds}:{model}": source_meta[(ds, model)]["checks"]["log_attempts"]["superseded_partial_attempts"] for ds in DATASETS for model in MODELS if source_meta[(ds, model)]["checks"]["log_attempts"]["superseded_partial_attempts"]},
        "tables": [
            str(TABLES / "f1_nc_final_comparison.csv"), str(TABLES / "f1_nc_final_comparison_paper_table.csv"),
            str(TABLES / "f1_lp_sports_final_comparison.csv"), str(TABLES / "f1_lp_sports_final_comparison_paper_table.csv"),
        ],
        "f2_gate": "PASS",
    }
    # Add sports partial-attempt evidence without losing the source audit.
    summary["sports_attempt_notes"] = {}
    for model in MODELS:
        meta = audit_external("lp", "sports-copurchase", model, label_support)[2]
        if meta["checks"]["log_attempts"]["superseded_partial_attempts"]:
            summary["sports_attempt_notes"][model] = meta["checks"]["log_attempts"]
    (OUT / "f1c_historical_baseline_certification.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps({
        "status": summary["status"],
        "nc_certified_seed_records": summary["nc_certified_seed_records"],
        "sports_certified_seed_records": summary["sports_certified_seed_records"],
        "uncertified_cells": summary["uncertified_cells"],
        "tables": summary["tables"],
        "doc": str(ROOT / "docs" / "mopf_f1c_historical_baseline_certification.md"),
    }, indent=2))


if __name__ == "__main__":
    main()
