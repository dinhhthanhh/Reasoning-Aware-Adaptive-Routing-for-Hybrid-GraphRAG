"""Update RQ1/RQ5 tables in docs/chuong5.tex from comparison eval summary.json."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY = ROOT / "results_phapdien" / "comparison_full" / "summary.json"
CHUONG5 = ROOT / "docs" / "chuong5.tex"
FIG_DIR = ROOT / "docs" / "Hinh_ve"

CONFIG_LABELS = {
    "pure_vector": "Pure Vector",
    "pure_graph": "Pure Graph",
    "pure_hybrid": "Pure Hybrid",
    "single_stage": "Single-stage",
    "router": "Hai giai đoạn (đề xuất)",
    "oracle": "Oracle",
}

TABLE_ORDER = [
    "pure_vector",
    "pure_graph",
    "pure_hybrid",
    "single_stage",
    "router",
]

COST_LABELS = {
    "pure_vector": "Pure Vector",
    "pure_graph": "Pure Graph",
    "pure_hybrid": "Pure Hybrid",
    "single_stage": "Single-stage",
    "router": "Hai giai đoạn",
}


def _fmt_pct(v: float) -> str:
    return f"{v * 100:.1f}\\%"


def _fmt_ms(v: float) -> str:
    return f"{v:,.0f}".replace(",", "{,}")


def _metric_row(parts: list[str], bests: dict[str, float], s: dict, cmp_min: set[str]) -> str:
    for idx, key in [(1, "avg_f1"), (2, "avg_em"), (3, "hit@1"), (4, "hit@3"),
                     (5, "latency_mean_ms"), (6, "latency_p95_ms")]:
        val = s[key]
        best = bests[key]
        if abs(val - best) < 1e-6:
            parts[idx] = f"\\textbf{{{parts[idx].strip()}}}"
    return " & ".join(parts)


def build_end_to_end_table(by_config: dict[str, dict]) -> tuple[str, str]:
    bests = {
        "avg_f1": max(by_config[c]["avg_f1"] for c in TABLE_ORDER),
        "avg_em": max(by_config[c]["avg_em"] for c in TABLE_ORDER),
        "hit@1": max(by_config[c]["hit@1"] for c in TABLE_ORDER),
        "hit@3": max(by_config[c]["hit@3"] for c in TABLE_ORDER),
        "latency_mean_ms": min(by_config[c]["latency_mean_ms"] for c in TABLE_ORDER),
        "latency_p95_ms": min(by_config[c]["latency_p95_ms"] for c in TABLE_ORDER),
    }
    main_lines = []
    for cfg in TABLE_ORDER:
        s = by_config[cfg]
        s2 = _fmt_pct(s["stage2_rate"]) if cfg in ("single_stage", "router") else "--"
        fb = _fmt_pct(s["fallback_rate"]) if cfg in ("single_stage", "router") else "--"
        parts = [
            CONFIG_LABELS[cfg],
            f"{s['avg_f1']:.3f}",
            f"{s['avg_em']:.3f}",
            f"{s['hit@1']:.3f}",
            f"{s['hit@3']:.3f}",
            _fmt_ms(s["latency_mean_ms"]),
            _fmt_ms(s["latency_p95_ms"]),
            s2,
            fb + " \\\\",
        ]
        main_lines.append(_metric_row(parts, bests, s, set()))

    o = by_config["oracle"]
    oracle_line = (
        f"Oracle       & {o['avg_f1']:.3f} & {o['avg_em']:.3f} & "
        f"{o['hit@1']:.3f} & {o['hit@3']:.3f} & "
        f"{_fmt_ms(o['latency_mean_ms'])} & {_fmt_ms(o['latency_p95_ms'])} & -- & -- \\\\"
    )
    return "\n".join(main_lines), oracle_line


def build_cost_table(by_config: dict[str, dict]) -> tuple[str, str]:
    main_lines = []
    for cfg in TABLE_ORDER:
        s = by_config[cfg]
        s2 = _fmt_pct(s["stage2_rate"]) if cfg in ("single_stage", "router") else "0\\%"
        fb = _fmt_pct(s["fallback_rate"]) if cfg in ("single_stage", "router") else "0\\%"
        main_lines.append(
            f"{COST_LABELS[cfg]} & {s['avg_f1']:.3f} & {_fmt_ms(s['latency_mean_ms'])} & "
            f"{_fmt_ms(s['latency_p95_ms'])} & {s2} & {fb} \\\\"
        )
    o = by_config["oracle"]
    oracle_line = (
        f"Oracle & {o['avg_f1']:.3f} & {_fmt_ms(o['latency_mean_ms'])} & "
        f"{_fmt_ms(o['latency_p95_ms'])} & 0\\% & 0\\% \\\\"
    )
    return "\n".join(main_lines), oracle_line


def replace_table_body(tex: str, label: str, main_body: str, oracle_body: str) -> str:
    pattern = (
        rf"(\\label\{{{label}\}}.*?\\midrule\n)"
        rf"(.*?)"
        rf"(\\midrule\n).*?(\\\\\n)"
        rf"(\\bottomrule)"
    )
    return re.sub(
        pattern,
        rf"\1{main_body}\n\3{oracle_body}\n\4",
        tex,
        count=1,
        flags=re.DOTALL,
    )


def generate_f1_bar(by_config: dict[str, dict], out_path: Path) -> None:
    labels = ["Vector", "Graph", "Hybrid", "Single", "Two-stage", "Oracle"]
    keys = TABLE_ORDER + ["oracle"]
    values = [by_config[k]["avg_f1"] for k in keys]
    colors = ["#5ab4ac"] * 5 + ["#d8b365"]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=colors, edgecolor="#333333", linewidth=0.8)
    ax.set_ylabel("Answer F1")
    ax.set_ylim(0, max(max(values) * 1.15, 0.5))
    ax.set_title("End-to-end Answer F1 (phapdien_strict, N=600)")
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def update_observations(tex: str, by_config: dict[str, dict]) -> str:
    ts = by_config["router"]
    ss = by_config["single_stage"]
    pv = by_config["pure_vector"]
    pg = by_config["pure_graph"]
    ph = by_config["pure_hybrid"]
    oracle = by_config["oracle"]
    delta_oracle = ts["avg_f1"] - oracle["avg_f1"]
    delta_ss = ts["avg_f1"] - ss["avg_f1"]
    s2_pct = ts["stage2_rate"] * 100
    extra_lat = ts["latency_mean_ms"] - oracle["latency_mean_ms"]

    tex = re.sub(
        r"\\textit\{Lưu ý tái lập thí nghiệm:?\}.*?\n\n",
        "",
        tex,
        count=1,
        flags=re.DOTALL,
    )

    replacements = [
        (
            r"\\caption\{Điểm Answer~F1.*?\}",
            (
                f"\\caption{{Điểm Answer~F1 trên benchmark nghiêm ngặt $600$ câu hỏi "
                f"(\\texttt{{phapdien\\_strict}}). Pure Vector: ${pv['avg_f1']:.3f}$; "
                f"Hai giai đoạn: ${ts['avg_f1']:.3f}$; Oracle: ${oracle['avg_f1']:.3f}$; "
                f"Single-stage: ${ss['avg_f1']:.3f}$.}}"
            ),
        ),
        (
            r"\\textbf\{Quan sát 1 -- Giai đoạn~2.*?\.",
            (
                f"\\textbf{{Quan sát 1 -- Giai đoạn~2 đóng vai trò suy luận tường minh, không chỉ sửa lỗi định tuyến.}} "
                f"Hai giai đoạn (F1~$={ts['avg_f1']:.3f}$) so với Oracle ($={oracle['avg_f1']:.3f}$, "
                f"$\\Delta={delta_oracle:+.3f}$)."
            ),
        ),
        (
            r"\\textbf\{Quan sát 2 -- Giai đoạn~2.*?\.",
            (
                f"\\textbf{{Quan sát 2 -- Giai đoạn~2 giải thích phần lớn cải thiện từ định tuyến.}} "
                f"Single-stage ($={ss['avg_f1']:.3f}$) vs hai giai đoạn ($={ts['avg_f1']:.3f}$, "
                f"$\\Delta={delta_ss:+.3f}$)."
            ),
        ),
        (
            r"\\textbf\{Quan sát 3 -- Pure Vector.*?\.",
            (
                f"\\textbf{{Quan sát 3 -- Pure Vector là baseline cố định mạnh.}} "
                f"F1~$={pv['avg_f1']:.3f}$, Hit@1~$={pv['hit@1']:.3f}$."
            ),
        ),
        (
            r"\\textbf\{Quan sát 4 -- Pure Graph.*?\.",
            (
                f"\\textbf{{Quan sát 4 -- Pure Graph đánh đổi độ trôi chảy lấy EM.}} "
                f"EM~$={pg['avg_em']:.3f}$, F1~$={pg['avg_f1']:.3f}$."
            ),
        ),
        (
            r"\\textbf\{Quan sát 5 -- Pure Hybrid.*?\.",
            (
                f"\\textbf{{Quan sát 5 -- Pure Hybrid luôn bật tốn chi phí hợp nhất.}} "
                f"F1~$={ph['avg_f1']:.3f}$, Hit@1~$={ph['hit@1']:.3f}$, "
                f"độ trễ TB~${_fmt_ms(ph['latency_mean_ms'])}$\\,ms trên Pháp điển thống nhất."
            ),
        ),
        (
            r"Single-stage nhanh nhất \(\$[^$]+\$ms\).*?F1~[^.]+\.",
            (
                f"Single-stage nhanh nhất (${_fmt_ms(ss['latency_mean_ms'])}$ms) "
                f"nhưng F1 thấp (${ss['avg_f1']:.3f}$)."
            ),
        ),
        (
            r"ở tỉ lệ kích hoạt \$[^$]+\$ quan sát được, hệ thống hai giai đoạn chạy ở \$[^$]+\$ms trung bình\.",
            (
                f"ở tỉ lệ kích hoạt ${s2_pct:.1f}\\%$, hệ thống hai giai đoạn chạy "
                f"${_fmt_ms(ts['latency_mean_ms'])}$ms trung bình."
            ),
        ),
        (
            r"mức tăng \$[^$]+\$ điểm F1 so với Oracle\.",
            f"mức tăng ${delta_oracle:+.3f}$ điểm F1 so với Oracle.",
        ),
        (
            r"hệ thống hai giai đoạn cộng thêm khoảng \$[^$]+\$ms",
            f"hệ thống hai giai đoạn cộng thêm khoảng ${_fmt_ms(extra_lat)}$ms",
        ),
        (
            r"Oracle Router chạy ở \$[^$]+\$ms",
            f"Oracle Router chạy ở ${_fmt_ms(oracle['latency_mean_ms'])}$ms",
        ),
    ]
    for pattern, repl in replacements:
        tex = re.sub(pattern, lambda m, r=repl: r, tex, count=1, flags=re.DOTALL)
    return tex


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    ap.add_argument("--chuong5", type=Path, default=CHUONG5)
    ap.add_argument("--fig", type=Path, default=FIG_DIR / "f1_bar.png")
    args = ap.parse_args()

    summaries = json.loads(args.summary.read_text(encoding="utf-8"))
    by_config = {s["config"]: s for s in summaries}
    missing = set(CONFIG_LABELS) - set(by_config)
    if missing:
        raise SystemExit(f"Missing configs in summary: {missing}")

    tex = args.chuong5.read_text(encoding="utf-8")
    e2e_main, e2e_oracle = build_end_to_end_table(by_config)
    cost_main, cost_oracle = build_cost_table(by_config)
    tex = replace_table_body(tex, "tab:end_to_end", e2e_main, e2e_oracle)
    tex = replace_table_body(tex, "tab:cost", cost_main, cost_oracle)
    tex = update_observations(tex, by_config)
    args.chuong5.write_text(tex, encoding="utf-8")
    generate_f1_bar(by_config, args.fig)

    print(f"Updated {args.chuong5}")
    print(f"Wrote {args.fig}")
    for cfg in TABLE_ORDER + ["oracle"]:
        s = by_config[cfg]
        print(
            f"  {cfg}: F1={s['avg_f1']:.3f} EM={s['avg_em']:.3f} "
            f"Hit@1={s['hit@1']:.3f} lat={s['latency_mean_ms']:.0f}ms"
        )


if __name__ == "__main__":
    main()
