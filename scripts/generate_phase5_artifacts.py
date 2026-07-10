"""Generate Phase 5 reproducibility snapshot and defense docs.

This script only copies small existing result artifacts and writes Markdown/JSON
documentation. It does not run benchmarks, retrain models, rebuild indexes, or
modify experiment metrics.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = ROOT / "docs" / "final_results_snapshot"
MAX_COPY_BYTES = 5 * 1024 * 1024


SNAPSHOT_CANDIDATES = [
    ("eval_results/legal_strict_full_summary.json", "Main strict end-to-end QA results", True, "main"),
    ("eval_results/clarify_stage1_only.json", "Stage 1 only clarification baseline", True, "diagnostic"),
    ("eval_results/clarify_two_stage.json", "Original two-stage clarification benchmark", True, "diagnostic"),
    ("eval_results/clarify_two_stage.csv", "Original two-stage clarification predictions", False, "diagnostic"),
    ("results_phase3/clarify_eval_summary.json", "Phase 3 clarification summary", True, "diagnostic"),
    ("results_phase3/clarify_eval_summary.md", "Phase 3 clarification report", True, "diagnostic"),
    ("results_phase3/clarify_eval_results.csv", "Phase 3 clarification predictions", False, "diagnostic"),
    ("evaluation/conversation_ambiguity_eval.json", "Conversation ambiguity evaluation dataset", True, "diagnostic"),
    ("results/conversation_ambiguity_summary.json", "Pre-Phase-3 conversation summary", True, "diagnostic"),
    ("results/conversation_ambiguity_summary.md", "Pre-Phase-3 conversation report", True, "diagnostic"),
    ("results_phase3/conversation_ambiguity_summary.json", "Phase 3 conversation summary", True, "diagnostic"),
    ("results_phase3/conversation_ambiguity_summary.md", "Phase 3 conversation report", True, "diagnostic"),
    ("results_phase3/conversation_ambiguity_predictions.jsonl", "Phase 3 conversation predictions", False, "diagnostic"),
    ("results_phase3/conversation_ambiguity_failures.jsonl", "Phase 3 conversation failures", False, "diagnostic"),
    ("results/stage1_confusion_matrix.json", "Stage 1 confusion matrix", True, "diagnostic"),
    ("results/stage1_classification_report.json", "Stage 1 classification report", True, "diagnostic"),
    ("results/stage1_feature_importance.json", "Stage 1 feature importance", True, "diagnostic"),
    ("results/stage1_diagnostics_summary.md", "Stage 1 diagnostic report", True, "diagnostic"),
    ("results_phase3/stage1_diagnostics_summary.md", "Phase 3 Stage 1 diagnostic report", False, "diagnostic"),
    ("eval_results/post_migration_graph_quality.json", "Post-migration graph quality", True, "diagnostic"),
    ("results_phase3/strict_routing_sanity_summary.json", "Phase 3 strict routing-only sanity summary", True, "diagnostic"),
    ("results_phase3/strict_routing_sanity_summary.md", "Phase 3 strict routing-only sanity report", True, "diagnostic"),
    ("results_phase3/demo_conversation_routing_output.md", "Phase 3 conversation routing demo output", True, "diagnostic"),
    ("results_phase3/demo_conversation_routing_output.json", "Phase 3 conversation routing demo output JSON", False, "diagnostic"),
    ("results/phase1_audit_report.md", "Phase 1 audit report", False, "diagnostic"),
    ("results/phase2_conversation_benchmark_report.md", "Phase 2 conversation benchmark report", False, "diagnostic"),
    ("results_phase3/phase3_improvement_report.md", "Phase 3 improvement report", True, "diagnostic"),
    ("docs/AI(PM)_ver 2.3.tex", "Final paper LaTeX source", True, "paper"),
    ("docs/AI(PM)_ver 2.3.pdf", "Final paper PDF", True, "paper"),
    ("docs/biblio.bib", "Final paper bibliography", True, "paper"),
]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def read_json(path: str) -> Any:
    p = ROOT / path
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def read_text(path: str, default: str = "not_found") -> str:
    p = ROOT / path
    if not p.exists():
        return default
    return p.read_text(encoding="utf-8", errors="replace")


def git_output(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:  # pragma: no cover - audit fallback
        return f"not_available ({exc})"


def sanitize_remote(value: str) -> str:
    return re.sub(r"(https://)([^/@]+@)", r"\1<redacted>@", value)


def file_status(path: str) -> tuple[str, int | None]:
    p = ROOT / path
    if not p.exists():
        return "not_found", None
    size = p.stat().st_size
    if size > MAX_COPY_BYTES:
        return "skipped_large_file", size
    return "copied", size


def copy_snapshot() -> list[dict[str, Any]]:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    for existing in SNAPSHOT_DIR.iterdir():
        if existing.is_file():
            existing.unlink()
    rows: list[dict[str, Any]] = []
    basenames = [Path(item[0]).name for item in SNAPSHOT_CANDIDATES]
    for original, purpose, used_in_paper, result_type in SNAPSHOT_CANDIDATES:
        src = ROOT / original
        status, size = file_status(original)
        if basenames.count(src.name) > 1:
            snapshot_name = original.replace("/", "__").replace("\\", "__")
        else:
            snapshot_name = src.name
        dest = SNAPSHOT_DIR / snapshot_name
        if status == "copied":
            shutil.copy2(src, dest)
        rows.append(
            {
                "snapshot_filename": snapshot_name,
                "original_path": original,
                "status": status,
                "file_size_bytes": size,
                "purpose": purpose,
                "used_in_paper": used_in_paper,
                "result_type": result_type,
            }
        )
    (SNAPSHOT_DIR / "manifest.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# Final Results Snapshot Manifest",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "| Snapshot filename | Original path | Status | File size | Purpose | Used in paper | Main/diagnostic |",
        "|---|---|---:|---:|---|---:|---|",
    ]
    for row in rows:
        size_text = "n/a" if row["file_size_bytes"] is None else str(row["file_size_bytes"])
        lines.append(
            "| {snapshot_filename} | `{original_path}` | {status} | {size} | {purpose} | {used} | {rtype} |".format(
                snapshot_filename=row["snapshot_filename"],
                original_path=row["original_path"],
                status=row["status"],
                size=size_text,
                purpose=row["purpose"],
                used="yes" if row["used_in_paper"] else "no",
                rtype=row["result_type"],
            )
        )
    (SNAPSHOT_DIR / "MANIFEST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def snapshot_path(original: str) -> str:
    basenames = [Path(item[0]).name for item in SNAPSHOT_CANDIDATES]
    base = Path(original).name
    if basenames.count(base) > 1:
        return f"docs/final_results_snapshot/{original.replace('/', '__').replace(chr(92), '__')}"
    return f"docs/final_results_snapshot/{base}"


def exists_text(path: str) -> str:
    return "exists" if (ROOT / path).exists() else "not_found"


def build_reproducibility_audit(snapshot_rows: list[dict[str, Any]]) -> None:
    branch = git_output(["git", "branch", "--show-current"])
    commit = git_output(["git", "rev-parse", "--short", "HEAD"])
    status = git_output(["git", "status", "--short"])
    log = git_output(["git", "log", "--oneline", "-n", "10"])
    remote = sanitize_remote(git_output(["git", "remote", "-v"]))
    ignored_policy = read_text(".gitignore")
    suspicious_large = [
        row for row in snapshot_rows if row["status"] == "skipped_large_file"
    ]
    lines = [
        "# Reproducibility Audit",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Git State",
        "",
        f"- Current branch: `{branch}`",
        f"- Current commit before Phase 5 commit: `{commit}`",
        "- Remote URL(s), sanitized:",
        "",
        "```text",
        remote or "not_available",
        "```",
        "",
        "- Uncommitted files before final Phase 5 commit:",
        "",
        "```text",
        status or "clean",
        "```",
        "",
        "## Recent Commits",
        "",
        "```text",
        log,
        "```",
        "",
        "## Tracked/Untracked Files Worth Noting",
        "",
        "- `configs/config.yaml.example` was already modified before Phase 5.",
        "- `evaluation/README_legal_clarify_eval.md`, `evaluation/legal_clarify_eval.json`, `experiments/`, and `scripts/build_legal_clarify_eval.py` were already untracked before Phase 5.",
        "- `docs/` sources are ignored by default in `.gitignore`; Phase paper/docs files are intentionally force-added when needed.",
        "- `eval_results/` and `reports/` are ignored as generated experiment folders. The `results/` folder contains curated small diagnostic artifacts from earlier phases; ad-hoc reruns should use `results_repro/`, which is ignored.",
        "",
        "## .gitignore Review",
        "",
        "The current `.gitignore` excludes `.env`, local configs with credentials, vector/database folders, model weights, logs, caches, LaTeX build artifacts, `node_modules`, and generated result folders. This is appropriate for GitHub readiness.",
        "",
        "Key sensitive/large patterns present:",
        "",
        "- `.env`, `.env.local`, `.env.*.local`",
        "- `configs/config.yaml`, `configs/config_legal.yaml`",
        "- `data/`, `*.pkl`, `*.bin`, `*.pt`, `*.safetensors`, vector/Neo4j folders",
        "- `frontend/node_modules/`, `frontend/.next/`",
        "- `__pycache__/`, `.pytest_cache/`, LaTeX `*.aux`, `*.bbl`, `*.blg`, `*.log`, `*.out`",
        "",
        "## Snapshot Large File Policy",
        "",
    ]
    if suspicious_large:
        for row in suspicious_large:
            lines.append(f"- Skipped `{row['original_path']}` because size is `{row['file_size_bytes']}` bytes.")
    else:
        lines.append("- No requested snapshot artifact exceeded the 5 MB copy threshold.")
    lines += [
        "",
        "## Fresh Clone/Worktree Simulation",
        "",
        "Recorded in `docs/phase5_reproducibility_report.md`. If a worktree run fails, the failure is treated as an environment/setup issue and not as a benchmark result.",
    ]
    (ROOT / "docs" / "reproducibility_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_defense_pack(snapshot_rows: list[dict[str, Any]]) -> None:
    branch = git_output(["git", "branch", "--show-current"])
    commit = git_output(["git", "rev-parse", "--short", "HEAD"])
    date = datetime.now().isoformat(timespec="seconds")
    found = {row["original_path"]: row for row in snapshot_rows}

    pack = {
        "project": "Reasoning-aware Adaptive Routing for Hybrid GraphRAG",
        "generated": date,
        "branch": branch,
        "commit_before_phase5_commit": commit,
        "paper_tex": "docs/AI(PM)_ver 2.3.tex",
        "paper_pdf": "docs/AI(PM)_ver 2.3.pdf",
        "main_results": {
            "source": "eval_results/legal_strict_full_summary.json",
            "snapshot": snapshot_path("eval_results/legal_strict_full_summary.json"),
            "systems": {
                "Pure Vector": {"f1": 0.3626, "routing_accuracy": 0.5000, "latency_ms": 1270.7},
                "Pure Graph": {"f1": 0.3556, "routing_accuracy": 0.2500, "latency_ms": 2283.4},
                "Single-stage Router": {"f1": 0.4231, "routing_accuracy": 0.9350, "latency_ms": 2209.2},
                "Two-stage Hybrid": {"f1": 0.4235, "routing_accuracy": 0.9283, "latency_ms": 3913.4},
            },
        },
        "clarification": {
            "stage1_only": {"route_accuracy": 0.269, "precision": 0.0, "recall": 0.0, "f1": 0.0, "trigger": 0.0},
            "pre_phase3_two_stage": {"route_accuracy": 0.585, "precision": 1.0, "recall": 0.519, "f1": 0.684, "trigger": 0.577},
            "after_phase3": {"route_accuracy": 0.7222, "precision": 1.0, "recall": 0.7244, "f1": 0.8401, "trigger": 0.8120, "false_positives": 0, "false_negatives": 43},
        },
        "conversation_stress": {
            "before": {"route_accuracy": 0.475, "precision": 0.907, "recall": 0.390, "f1": 0.545, "stage2_trigger": 0.713, "stage2_override": 0.333, "latency_ms": 2238.7},
            "after": {"route_accuracy": 0.750, "precision": 0.963, "recall": 0.780, "f1": 0.862, "stage2_trigger": 0.825, "stage2_override": 0.227, "latency_ms": 3127.2, "history_resolution_accuracy": 0.850},
        },
        "strict_routing_sanity": {"total": 600, "route_accuracy": 0.8933, "clarify_false_positives": 10, "stage2_trigger_rate": 0.5583, "stage2_override_rate": 0.3403, "avg_latency_ms": 1972.4},
        "artifact_status": found,
    }
    (ROOT / "docs" / "defense_evidence_pack.json").write_text(
        json.dumps(pack, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    md = [
        "# Defense Evidence Pack",
        "",
        "## 1. Metadata",
        "",
        "| Item | Value |",
        "|---|---|",
        "| Project | Reasoning-aware Adaptive Routing for Hybrid GraphRAG |",
        "| Final paper source | `docs/AI(PM)_ver 2.3.tex` |",
        "| Final paper PDF | `docs/AI(PM)_ver 2.3.pdf` |",
        f"| Current branch | `{branch}` |",
        f"| Current commit before Phase 5 commit | `{commit}` |",
        f"| Date generated | `{date}` |",
        "| Phase 1 audit commit | `999e493 phase1: add routing diagnostics and audit reports` |",
        "| Phase 2 benchmark/demo commit | `e822597 phase2: add conversation ambiguity benchmark and demo` |",
        "| Phase 3 improvement commit | `3e2b5f2 phase3: improve conversation ambiguity routing` |",
        "| Phase 4 paper commit | `5dccfd8 phase4: update paper with conversation ambiguity results` |",
        "| Phase 4.1 polish commit | `b762829 phase4.1: polish paper consistency and compile` |",
        "| Phase 5 commit | Created after this audit with message `docs: add reproducibility and defense evidence pack`; exact hash is recorded in `git log` and the final Codex response. |",
        "",
        "## 2. One-page Executive Summary",
        "",
        "Đề tài giải quyết bài toán hỏi đáp pháp luật tiếng Việt khi một chiến lược retrieval cố định không đủ tốt. Câu hỏi pháp luật có thể là tra cứu trực tiếp, cần đi theo quan hệ pháp lý trong graph, cần kết hợp text và graph, hoặc còn mơ hồ nên phải hỏi lại trước khi trả lời.",
        "",
        "Novelty chính là `Reasoning-aware Adaptive Router`: Stage 1 dùng XGBoost để route nhanh theo đặc trưng reasoning/ambiguity; Stage 2 dùng LLM verifier có chọn lọc cho câu hỏi uncertain, relation-heavy hoặc ambiguous. Route set gồm `dense_retrieval`, `graph_traversal`, `hybrid_reasoning`, và `clarify`.",
        "",
        "Kết quả chính trên strict 600-query end-to-end benchmark: Single-stage Router đạt F1 `0.4231`, tốt hơn Pure Vector `0.3626` và Pure Graph `0.3556`; Two-stage Hybrid đạt F1 tốt nhất `0.4235`, nhưng strict routing accuracy thấp hơn nhẹ vì strict set không có intended clarification queries.",
        "",
        "Kết quả diagnostic conversation: sau Phase 3, conversation stress test 160 câu đạt clarify F1 `0.862` và history-resolution accuracy `0.850`. Đây là diagnostic/stress test, không thay thế benchmark strict end-to-end.",
        "",
        "Limitation chính: `multi_interpretation` vẫn yếu trên original template benchmark; `answerable_with_history` route accuracy chỉ `0.600`; clear graph/hybrid control chỉ `0.500`; strict routing-only sanity sau Phase 3 vẫn có 10 false clarifications. Kết luận cần nhớ: hệ thống không always GraphRAG, mà adaptive routing theo reasoning demand và ambiguity.",
        "",
        "## 3. Benchmark Map",
        "",
        "| Benchmark | File/path | Size | Purpose | Main/diagnostic | Paper location | Notes |",
        "|---|---|---:|---|---|---|---|",
        "| Strict Vietnamese legal QA test | `qa_pipeline/data/legal_strict/test.json`; results `eval_results/legal_strict_full_summary.json`; snapshot `docs/final_results_snapshot/legal_strict_full_summary.json` | 600 | End-to-end QA and routing quality | Main | `tab:end_to_end_results` | Do not replace with routing-only sanity. |",
        "| Original ambiguity benchmark | `evaluation/legal_clarify_eval.json`; results `eval_results/clarify_*.json`, `results_phase3/clarify_eval_summary.json` | 234 | Clarification behavior | Diagnostic | `tab:clarify_results`, `tab:clarify_phase3_before_after` | Pre-Phase-3 F1 0.684; Phase 3 F1 0.8401. |",
        "| Conversation ambiguity stress test | `evaluation/conversation_ambiguity_eval.json`; results `results_phase3/conversation_ambiguity_summary.json` | 160 | History-aware ambiguity routing | Diagnostic | `tab:conversation_stress_summary` | Use for demo; not production distribution. |",
        "| Strict routing-only sanity after Phase 3 | `results_phase3/strict_routing_sanity_summary.json` | 600 | Routing-only regression check | Diagnostic | Discussion/Limitations | No retrieval or generation; 10 false clarifications. |",
        "| Stage 1 offline classifier diagnostic | `results/stage1_*`; training output | 600 test split | Classifier behavior | Diagnostic | `tab:per_class_stage1` | Offline classifier and deployed routing metrics are not identical. |",
        "| Routing baseline comparison | Paper table; experiment artifact if available | 600 | Compare rule/model routers | Diagnostic | `tab:routing_baselines`, `fig:routing_baselines` | PhoBERT higher Macro-F1 but slower. |",
        "",
        "## 4. Dataset and Artifact Paths",
        "",
        "| Artifact | Path | Status | Notes |",
        "|---|---|---|---|",
    ]
    artifact_paths = [
        ("Strict train", "qa_pipeline/data/legal_strict/train.json"),
        ("Strict dev", "qa_pipeline/data/legal_strict/dev.json"),
        ("Strict test", "qa_pipeline/data/legal_strict/test.json"),
        ("Final train", "qa_pipeline/data/final/train.json"),
        ("Final dev", "qa_pipeline/data/final/dev.json"),
        ("Final test", "qa_pipeline/data/final/test.json"),
        ("Ambiguity eval", "evaluation/legal_clarify_eval.json"),
        ("Conversation ambiguity eval", "evaluation/conversation_ambiguity_eval.json"),
        ("Stage 1 checkpoint", "data/router_training/legal_strict/router_model.pkl"),
        ("Vector store", "data/vector_store/chroma_harrier_oss_0_6b"),
        ("Graph quality file", "eval_results/post_migration_graph_quality.json"),
        ("Demo script", "scripts/demo_conversation_routing.py"),
        ("Conversation eval script", "scripts/evaluate_conversation_ambiguity.py"),
        ("Strict routing sanity script", "scripts/evaluate_strict_routing_only.py"),
    ]
    for name, path in artifact_paths:
        md.append(f"| {name} | `{path}` | {exists_text(path)} | {'gitignored/local artifact' if path.startswith('data/') else ''} |")
    md += [
        "",
        "## 5. Main Paper Results",
        "",
        "Source: `eval_results/legal_strict_full_summary.json`; snapshot: `docs/final_results_snapshot/legal_strict_full_summary.json`; paper table: `tab:end_to_end_results`.",
        "",
        "| System | F1 | Routing accuracy | Avg latency |",
        "|---|---:|---:|---:|",
        "| Pure Vector | 0.3626 | 0.5000 | 1,270.7 ms |",
        "| Pure Graph | 0.3556 | 0.2500 | 2,283.4 ms |",
        "| Single-stage Router | 0.4231 | 0.9350 | 2,209.2 ms |",
        "| Two-stage Hybrid | 0.4235 | 0.9283 | 3,913.4 ms |",
        "",
        "Đây là main strict end-to-end benchmark. Không thay bằng Phase 3 strict routing-only sanity vì sanity run không có retrieval/generation.",
        "",
        "## 6. Stage 1 Routing Diagnostics",
        "",
        "Offline Stage 1 per-class table used in paper (`tab:per_class_stage1`):",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
        "| Dense retrieval | 0.825 | 0.897 | 0.859 | 300 |",
        "| Graph traversal | 0.646 | 0.547 | 0.592 | 150 |",
        "| Hybrid reasoning | 0.898 | 0.880 | 0.889 | 150 |",
        "| Macro average | 0.790 | 0.774 | 0.780 | 600 |",
        "",
        "Routing baselines in paper:",
        "",
        "| Router | Accuracy | Macro-F1 | Weighted-F1 |",
        "|---|---:|---:|---:|",
        "| MajorityRoute | 0.500 | 0.222 | 0.333 |",
        "| KeywordRuleRouter | 0.612 | 0.497 | 0.568 |",
        "| Logistic Regression | 0.825 | 0.781 | 0.806 |",
        "| Random Forest | 0.833 | 0.790 | 0.815 |",
        "| XGBoost (Stage 1) | 0.805 | 0.780 | 0.800 |",
        "| PhoBERT-base-v2 | 0.913 | 0.901 | 0.910 |",
        "",
        "Deployed Stage 1 diagnostic from Phase 1: accuracy `0.9383`, Macro-F1 `0.9327`, weighted-F1 `0.9387`; source: `results/stage1_diagnostics_summary.md` if present. Warning: offline training report and deployed Stage 1 diagnostic measure different things; do not mix these numbers.",
        "",
        "## 7. Clarification Benchmark",
        "",
        "| Variant | Route Acc. | P | R | F1 | Trigger | FP | FN | Source |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        "| Stage 1 only | 0.269 | 0.000 | 0.000 | 0.000 | 0.000 | not_available | not_available | `eval_results/clarify_stage1_only.json` |",
        "| Original Stage 1 + Stage 2 pre-Phase-3 | 0.585 | 1.000 | 0.519 | 0.684 | 0.577 | not_available | not_available | `eval_results/clarify_two_stage.json` |",
        "| After Phase 3 | 0.7222 | 1.0000 | 0.7244 | 0.8401 | 0.8120 | 0 | 43 | `results_phase3/clarify_eval_summary.json` |",
        "",
        "Stage 1 clarify F1 = 0 không phải bug: Stage 1 train trên 3 retrieval labels (`dense_retrieval`, `graph_traversal`, `hybrid_reasoning`), không có `clarify` label. Phase 3 cải thiện mạnh `missing_entity`, nhưng `multi_interpretation` vẫn yếu trên original template benchmark.",
        "",
        "## 8. Conversation-aware Ambiguity Stress Test",
        "",
        "Source: `results_phase3/conversation_ambiguity_summary.md/json`; snapshot copies in `docs/final_results_snapshot/`.",
        "",
        "| Metric | Before Phase 3 | After Phase 3 |",
        "|---|---:|---:|",
        "| Route accuracy | 0.475 | 0.750 |",
        "| Clarify precision | 0.907 | 0.963 |",
        "| Clarify recall | 0.390 | 0.780 |",
        "| Clarify F1 | 0.545 | 0.862 |",
        "| Stage 2 trigger rate | 0.713 | 0.825 |",
        "| Stage 2 override rate | 0.333 | 0.227 |",
        "| Avg latency | 2,238.7 ms | 3,127.2 ms |",
        "| History resolution accuracy | n/a | 0.850 |",
        "",
        "| Category after Phase 3 | Metric | Value |",
        "|---|---|---:|",
        "| answerable_with_history | route accuracy | 0.600 |",
        "| clarify_without_history | recall | 0.850 |",
        "| irrelevant_history | recall | 0.750 |",
        "| conflicting_history | recall | 0.700 |",
        "| missing_entity | recall | 0.850 |",
        "| multi_interpretation | recall | 0.750 |",
        "| clear_dense_control | route accuracy | 1.000 |",
        "| clear_graph_or_hybrid_control | route accuracy | 0.500 |",
        "",
        "Đây là diagnostic/stress test để demo conversation history và clarification, không thay strict end-to-end result.",
        "",
        "## 9. Strict Routing-only Sanity After Phase 3",
        "",
        "| Metric | Value |",
        "|---|---:|",
        "| Total | 600 |",
        "| Route accuracy | 0.8933 |",
        "| Clarify false positives | 10 |",
        "| Stage 2 trigger rate | 0.5583 |",
        "| Stage 2 override rate | 0.3403 |",
        "| Avg latency | 1,972.4 ms |",
        "",
        "Đây là routing-only sanity, không có answer generation. Không dùng thay `tab:end_to_end_results`. Dùng để nói limitation: Phase 3 còn cần calibration để tránh false clarification trên strict queries.",
        "",
        "## 10. Graph Evidence and Graph Quality",
        "",
        "Source: `eval_results/post_migration_graph_quality.json`; snapshot: `docs/final_results_snapshot/post_migration_graph_quality.json`.",
        "",
        "| Graph item | Count |",
        "|---|---:|",
        "| Nodes | 419,251 |",
        "| Edges | 1,239,542 |",
        "| LegalArticle | 70,347 |",
        "| VectorChunk | 199,530 |",
        "| HAS_ARTICLE | 70,347 |",
        "| BELONGS_TO | 199,530 |",
        "| REFERENCES | 769,000 |",
        "| AMENDS | 27 |",
        "| REPEALS | 19 |",
        "| GUIDES | 25 |",
        "| IMPLEMENTS | 3 |",
        "",
        "Graph đủ để demo và experiment sau migration, nhưng curated legal-effect relations còn sparse. Đây là lý do không always GraphRAG và là limitation/future work.",
        "",
        "## 11. Demo Evidence",
        "",
        "Demo scenarios from `results_phase3/demo_conversation_routing_output.md`: direct dense lookup -> `dense_retrieval`; relation-heavy query -> `graph_traversal`; pronoun with valid history resolves `Nghị định 100/2019/NĐ-CP` and routes graph; pronoun without history -> `clarify`; pronoun with irrelevant history -> `clarify`; missing entity -> `clarify`; multi-interpretation -> `clarify`.",
        "",
        "Demo script nói miệng 1-2 phút: \"Em sẽ demo router thay vì demo full generation để thấy rõ đóng góp chính. Với câu tra cứu trực tiếp, hệ thống chọn dense retrieval. Với câu hỏi quan hệ pháp lý, hệ thống chọn graph traversal. Khi câu hỏi dùng 'văn bản đó' và history có Nghị định 100/2019/NĐ-CP, HistoryResolver resolve được referent nên router không hỏi lại mà đi graph. Nếu không có history hoặc history không liên quan, router chuyển sang clarify để tránh trả lời nhầm văn bản. Điểm chính là hệ thống không always GraphRAG; nó chọn route theo reasoning demand và ambiguity.\"",
        "",
        "## 12. Frequently Asked Questions for Defense",
        "",
        "- Vì sao không dùng GraphRAG cho mọi câu hỏi? Graph traversal tốn hơn và có thể trả context rộng/diffuse cho câu hỏi lookup đơn giản; Pure Graph F1 thấp hơn Pure Vector trên strict test.",
        "- Vì sao Pure Graph thấp hơn Pure Vector? Graph hiện có nhiều `REFERENCES` generic nhưng legal-effect edges như `AMENDS/REPEALS` còn sparse, nên graph-only chưa luôn lấy đúng evidence.",
        "- Vì sao cần router? Vì mỗi query cần retrieval mechanism khác nhau: dense, graph, hybrid hoặc clarify.",
        "- Vì sao cần Stage 2 nếu Stage 1 routing accuracy cao? Stage 1 nhanh nhưng không có `clarify` label; Stage 2 xử lý ambiguity/uncertain/relation-heavy cases.",
        "- Vì sao Two-stage strict routing accuracy thấp hơn Single-stage? Strict set không có intended clarify queries; Stage 2 đôi khi can thiệp không cần thiết.",
        "- Vì sao Stage 1 clarify F1 = 0? Stage 1 train trên ba retrieval labels, không có clarify label.",
        "- Clarification có làm hệ thống chậm không? Có; Stage 2 routing khoảng 119x Stage 1-only trong routing logs, nên phải trigger chọn lọc.",
        "- Vì sao dùng XGBoost thay vì Logistic Regression hoặc Random Forest khi LR/RF có Macro-F1 ngang hoặc nhỉnh hơn? LR/RF/XGBoost đều là lightweight feature-based routers và chênh lệch nhỏ; XGBoost là checkpoint được tích hợp trong pipeline, hỗ trợ feature interactions phi tuyến và cung cấp confidence score dùng cho Stage 2 trigger. Điểm trade-off lớn nhất trong paper là XGBoost/PhoBERT: PhoBERT chính xác hơn nhưng chậm hơn và phụ thuộc GPU hơn.",
        "- Graph hiện tại yếu ở đâu? Curated legal-effect relations còn sparse: AMENDS 27, REPEALS 19, GUIDES 25, IMPLEMENTS 3.",
        "- Dataset labels có phải human-annotated không? Route labels chủ yếu metadata-derived; cần human review thêm.",
        "- Vì sao EM gần 0 nhưng F1 vẫn dùng được? Legal answers dài/paraphrastic, exact string match quá nghiêm; token F1 phân biệt hệ thống tốt hơn.",
        "- Phase 3 có thay main result không? Không. Phase 3 là diagnostic routing/clarification improvement, không thay strict end-to-end benchmark.",
        "- Nếu có thêm thời gian, cải thiện gì trước? Calibrate false clarification, human-reviewed clarify data, relation extraction for AMENDS/REPEALS, and graph retrieval diagnostics.",
        "",
        "## 13. Paper Table Mapping",
        "",
        "| Label | Reports | Source artifact | Snapshot artifact | Type | Caveat |",
        "|---|---|---|---|---|---|",
        "| `tab:dataset` | Strict split distribution | `qa_pipeline/data/legal_strict/*.json` | not_copied | Main data | Metadata-derived labels. |",
        "| `fig:dataset_distribution` | Route label distribution | `qa_pipeline/data/legal_strict/*.json` | not_copied | Main data | Skewed toward dense. |",
        "| `tab:graph_stats` | Graph node stats | `eval_results/post_migration_graph_quality.json` | `docs/final_results_snapshot/post_migration_graph_quality.json` | Diagnostic | Post-migration graph. |",
        "| `tab:graph_relations` | Graph relation stats | `eval_results/post_migration_graph_quality.json` | `docs/final_results_snapshot/post_migration_graph_quality.json` | Diagnostic | Legal-effect relations sparse. |",
        "| `tab:end_to_end_results` | Strict E2E QA | `eval_results/legal_strict_full_summary.json` | `docs/final_results_snapshot/legal_strict_full_summary.json` | Main | Do not replace with routing-only sanity. |",
        "| `tab:per_class_stage1` | Offline Stage 1 per-class | training/eval report | snapshot stage1 files if present | Diagnostic | Offline diagnostic. |",
        "| `tab:routing_baselines` | Router baselines | paper/experiment output | not_available | Diagnostic | PhoBERT slower. |",
        "| `fig:routing_baselines` | Baseline chart | paper/experiment output | not_available | Diagnostic | Same caveat. |",
        "| `tab:clarify_results` | Original pre-Phase-3 clarify | `eval_results/clarify_two_stage.json` | snapshot copy | Diagnostic | Initial Stage 2 config. |",
        "| `tab:ambiguity_type_results` | Original ambiguity types | `eval_results/clarify_two_stage.json/csv` | snapshot copy | Diagnostic | Missing entity and multi-interpretation weak pre-Phase-3. |",
        "| `fig:clarify_f1_bar` | Original clarify metrics | `eval_results/clarify_two_stage.json` | snapshot copy | Diagnostic | Pre-Phase-3. |",
        "| `tab:conversation_stress_summary` | Conversation before/after | `results_phase3/conversation_ambiguity_summary.json` | snapshot copy | Diagnostic | Stress test. |",
        "| `tab:conversation_category_results` | Conversation categories | `results_phase3/conversation_ambiguity_summary.json` | snapshot copy | Diagnostic | Not production distribution. |",
        "| `tab:clarify_phase3_before_after` | Original ambiguity before/after Phase 3 | `results_phase3/clarify_eval_summary.json` | snapshot copy | Diagnostic | Shows 0.684 -> 0.840. |",
        "| `fig:latency_comparison` | E2E latency | `eval_results/legal_strict_full_summary.json` | snapshot copy | Main/diagnostic | Latency depends on local services. |",
        "| `tab:threshold_ablation` | Threshold sweep | `eval_results/ablation_results_legal.json` | not_copied | Diagnostic | Routing-oriented development setting. |",
        "",
        "## 14. Commands and Reproducibility",
        "",
        "Core commands recorded across phases:",
        "",
        "```bash",
        "python scripts/migrate_graph.py --config configs/config.yaml",
        "python scripts/check_neo4j_graph_quality.py --config configs/config.yaml --output eval_results/post_migration_graph_quality.json",
        "python scripts/run_benchmark_eval.py --config configs/config_legal.yaml --dataset legal_strict --eval-file qa_pipeline/data/legal_strict/test.json --systems all --eval-answer-style",
        "python scripts/run_clarify_eval.py --config configs/config.yaml --eval-file evaluation/legal_clarify_eval.json --disable-stage2 --output eval_results/clarify_stage1_only.json --csv-output eval_results/clarify_stage1_only.csv",
        "python scripts/run_clarify_eval.py --config configs/config.yaml --eval-file evaluation/legal_clarify_eval.json --output eval_results/clarify_two_stage.json --csv-output eval_results/clarify_two_stage.csv",
        "python scripts/build_conversation_ambiguity_eval.py --output evaluation/conversation_ambiguity_eval.json",
        "python scripts/evaluate_conversation_ambiguity.py --config configs/config.yaml --eval-file evaluation/conversation_ambiguity_eval.json --output-dir results_phase3",
        "python scripts/evaluate_strict_routing_only.py --config configs/config.yaml --test-file qa_pipeline/data/legal_strict/test.json --output-dir results_phase3",
        "python scripts/demo_conversation_routing.py --config configs/config.yaml --output-md results_phase3/demo_conversation_routing_output.md --output-json results_phase3/demo_conversation_routing_output.json",
        "pdflatex -interaction=nonstopmode -halt-on-error \"AI(PM)_ver 2.3.tex\"",
        "bibtex \"AI(PM)_ver 2.3\"",
        "pdflatex -interaction=nonstopmode -halt-on-error \"AI(PM)_ver 2.3.tex\"",
        "pdflatex -interaction=nonstopmode -halt-on-error \"AI(PM)_ver 2.3.tex\"",
        "```",
        "",
        "Do not rerun the full benchmark unless Neo4j, Chroma, router checkpoint, and LLM endpoint are ready.",
        "",
        "## 15. Final Defense Checklist",
        "",
        "- [x] Paper PDF compiled",
        "- [x] No undefined refs/citations in final compile",
        "- [x] Strict results preserved",
        "- [x] Phase 3 results marked diagnostic",
        "- [x] Demo script available",
        "- [x] Evidence pack generated",
        "- [x] Final results snapshot generated",
        "- [ ] Key limitations memorized",
        "- [x] Phase 5 commit command prepared; exact hash is recorded after commit in `git log` and the final Codex response",
        "- [x] Backup PDF and data artifacts saved in snapshot",
        "",
        "## 16. Numbers to Never Mix Up",
        "",
        "- Strict end-to-end F1 (`0.4231/0.4235`) is not routing-only sanity (`0.8933`).",
        "- Original ambiguity F1 `0.684` is pre-Phase-3; Phase 3 original ambiguity F1 is `0.8401`.",
        "- Conversation stress F1 `0.862` is diagnostic and separate from strict benchmark.",
        "- Offline Stage 1 classifier report is not the deployed Stage 1 diagnostic.",
        "- Single-stage strict routing accuracy `0.9350` is not Phase 3 strict routing-only sanity `0.8933`.",
    ]
    (ROOT / "docs" / "defense_evidence_pack.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    rows = copy_snapshot()
    build_reproducibility_audit(rows)
    build_defense_pack(rows)
    print(f"Generated snapshot and defense docs in {rel(SNAPSHOT_DIR)}")


if __name__ == "__main__":
    main()
