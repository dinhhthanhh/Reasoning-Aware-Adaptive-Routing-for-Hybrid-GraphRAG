"""Small conversation-routing demo for Phase 2.

The default demo is routing-only and does not run retrieval or generation.
It prints and saves a readable trace for representative scenarios.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from router.two_stage_router import TwoStageRouter


DEMO_CASES: list[dict[str, str]] = [
    {
        "name": "Direct dense lookup",
        "history": "",
        "query": "Điều kiện kết hôn theo Luật Hôn nhân và gia đình gồm những gì?",
        "expected": "dense_retrieval",
    },
    {
        "name": "Relation-heavy graph/hybrid",
        "history": "",
        "query": "Quyết định 732/QĐ-UBND bãi bỏ hoặc sửa đổi quy định nào?",
        "expected": "graph_traversal",
    },
    {
        "name": "Pronoun with valid history",
        "history": "Người dùng đang hỏi về Nghị định 100/2019/NĐ-CP về xử phạt vi phạm hành chính trong lĩnh vực giao thông đường bộ.",
        "query": "Văn bản đó còn hiệu lực không?",
        "expected": "not_clarify",
    },
    {
        "name": "Pronoun without history",
        "history": "",
        "query": "Văn bản đó còn hiệu lực không?",
        "expected": "clarify",
    },
    {
        "name": "Pronoun with irrelevant history",
        "history": "Người dùng hỏi cách tra cứu văn bản pháp luật trên cổng thông tin điện tử, nhưng chưa nêu văn bản cụ thể.",
        "query": "Văn bản đó còn hiệu lực không?",
        "expected": "clarify",
    },
    {
        "name": "Missing entity",
        "history": "",
        "query": "Mức phạt trong trường hợp này là bao nhiêu?",
        "expected": "clarify",
    },
    {
        "name": "Multi-interpretation",
        "history": "",
        "query": "Doanh nghiệp có bị phạt không?",
        "expected": "clarify",
    },
]


def _backend_for_route(route: str | None) -> str:
    if route == "dense_retrieval":
        return "vector"
    if route == "graph_traversal":
        return "graph"
    if route == "hybrid_reasoning":
        return "hybrid"
    if route == "clarify":
        return "none"
    return "unknown"


def _field(output: Any, name: str, default: Any = None) -> Any:
    return getattr(output, name, default)


def _render_case(case: dict[str, str], output: Any) -> str:
    features = _field(output, "features")
    final_route = _field(output, "route")
    clarify_question = _field(output, "clarify_question") or (
        "Bạn vui lòng cung cấp rõ văn bản, điều khoản, cơ quan hoặc bối cảnh pháp lý cần hỏi."
        if final_route == "clarify"
        else ""
    )
    final_answer = clarify_question if final_route == "clarify" else "[generation skipped: routing-only demo]"
    resolved_referent = _field(output, "resolved_referent", "not_available")

    lines = [
        f"## {case['name']}",
        "",
        f"- Expected: `{case['expected']}`",
        f"- History: {case['history'] or '[empty]'}",
        f"- User query: {case['query']}",
        f"- Stage 1 route: `{_field(output, 'stage1_route')}`",
        f"- Stage 1 confidence: `{_field(output, 'stage1_confidence')}`",
        f"- Ambiguity score: `{getattr(features, 'ambiguity_score', None)}`",
        f"- Has pronoun: `{getattr(features, 'has_pronoun', None)}`",
        f"- History length: `{getattr(features, 'history_length', None)}`",
        f"- History resolves ambiguity feature: `{getattr(features, 'history_resolves_ambiguity', None)}`",
        f"- History resolution status: `{getattr(features, 'history_resolution_status', None)}`",
        f"- History resolution confidence: `{getattr(features, 'history_resolution_confidence', None)}`",
        f"- Query has contextual reference: `{getattr(features, 'query_has_contextual_reference', None)}`",
        f"- Stage 2 triggered: `{_field(output, 'stage2_invoked')}`",
        f"- Stage 2 override: `{_field(output, 'stage2_override')}`",
        f"- Final route: `{final_route}`",
        f"- Resolved referent: `{resolved_referent}`",
        f"- Retrieved backend: `{_backend_for_route(final_route)}`",
        f"- Final answer / clarification: {final_answer}",
        "",
        "<details>",
        "<summary>Router reasoning</summary>",
        "",
        str(_field(output, "reasoning", "")),
        "",
        "</details>",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small conversation routing demo")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output", default="results/demo_conversation_routing_output.md")
    parser.add_argument("--output-dir", default=None, help="Directory for demo_conversation_routing_output.md")
    parser.add_argument("--no-generation", action="store_true", help="Routing-only mode; this is the default behavior")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    router = TwoStageRouter(config)
    sections = [
        "# Demo Conversation Routing Output",
        "",
        "This demo runs the current router with fixed scenarios. Retrieval and final answer generation are skipped; non-clarify routes show the backend that would be used.",
        "",
    ]
    machine_rows: list[dict[str, Any]] = []
    for case in DEMO_CASES:
        output = router.route(
            query=case["query"],
            history=case["history"] or None,
            session_id=f"demo_{case['name'].lower().replace(' ', '_')}",
        )
        sections.append(_render_case(case, output))
        machine_rows.append({
            "name": case["name"],
            "expected": case["expected"],
            "query": case["query"],
            "history": case["history"],
            "stage1_route": _field(output, "stage1_route"),
            "stage1_confidence": _field(output, "stage1_confidence"),
            "stage2_invoked": _field(output, "stage2_invoked"),
            "stage2_override": _field(output, "stage2_override"),
            "final_route": _field(output, "route"),
            "backend": _backend_for_route(_field(output, "route")),
            "history_resolution_status": getattr(_field(output, "features"), "history_resolution_status", None),
            "resolved_referent": _field(output, "resolved_referent", None),
            "latency_ms": _field(output, "latency_ms"),
        })

    text = "\n".join(sections)
    output_path = (
        Path(args.output_dir) / "demo_conversation_routing_output.md"
        if args.output_dir
        else Path(args.output)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(machine_rows, f, ensure_ascii=False, indent=2)

    print(text)


if __name__ == "__main__":
    main()
