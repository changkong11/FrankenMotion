from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

BASELINE_DIR = ROOT / (
    "outputs_amass/frankenmotion_clip_part_smplrifke_frankenstein-dataset/"
    "generations/t2m/test_latest-epoch=5999_20260703_170212"
)
LGFM_DIR = ROOT / (
    "outputs_amass_LG-FM/frankenmotion_clip_part_smplrifke_frankenstein-dataset/"
    "generations/t2m/test_last_20260718_171838"
)
OUT_DIR = ROOT / "LG_FrankenMotion方案管理"
METHOD_LABEL = "LG-FM"

BASE = "evaluation_results_20_frankenstein-dataset_guo+threshold"

SECTIONS = [
    ("Caption", "caption"),
    ("Action", "action"),
    ("Parts-Avg", "parts_average"),
]

METRICS = [
    ("FID", "FID", "lower", 2),
    ("FID+", "FID+", "lower", 3),
    ("MM-Dist", "MM_Dist", "lower", 2),
    ("R@1", "R1", "higher", 3),
    ("R@3", "R3", "higher", 3),
    ("M2T", "M2T", "higher", 3),
    ("M2M", "M2M", "higher", 3),
    ("TemporalAccel-Gap", "TemporalAccelGap", "lower", 3),
    ("Jerk-Gap", "JerkGap", "lower", 3),
    ("FootSlide-Gap", "FootSlideGap", "lower", 3),
    ("FootContact-Gap", "FootContactGap", "lower", 3),
    ("TorsoLimb-Gap", "TorsoLimbGap", "lower", 3),
]


def read_metric_avg_std(directory: Path, suffix: str, column: str) -> tuple[float, float] | None:
    path = directory / f"{BASE}_{suffix}.csv"
    df = pd.read_csv(path)

    # Older result files store only the raw cosine. Derive the GT-relative gap
    # from every run so legacy and newly evaluated experiments remain comparable.
    if column == "TorsoLimbGap" and column not in df.columns:
        gt_value = float(df.loc[df["run"] == "GT", "TorsoLimb"].iloc[0])
        values = (df[df["run"].str.startswith("run_")]["TorsoLimb"] - gt_value).abs()
        return float(values.mean()), float(values.std(ddof=0))

    if column not in df.columns:
        return None

    avg = df[df["run"] == "AVERAGE"].iloc[0]
    std = df[df["run"] == "STD"].iloc[0]
    return float(avg[column]), float(std[column])


def fmt(mean: float, std: float, digits: int) -> str:
    pm = "\u00b1"
    return f"{mean:.{digits}f} {pm} {std:.{digits}f}"


def gain_percent(baseline: float, lgfm: float, direction: str) -> float:
    if abs(baseline) < 1e-12:
        return 0.0
    if direction == "lower":
        return (baseline - lgfm) / abs(baseline) * 100.0
    return (lgfm - baseline) / abs(baseline) * 100.0


def is_lgfm_better(baseline: float, lgfm: float, direction: str) -> bool:
    if direction == "lower":
        return lgfm < baseline
    return lgfm > baseline


def build_rows() -> list[dict]:
    rows = []
    for section_name, suffix in SECTIONS:
        for metric_name, column, direction, digits in METRICS:
            baseline_stats = read_metric_avg_std(BASELINE_DIR, suffix, column)
            method_stats = read_metric_avg_std(LGFM_DIR, suffix, column)
            if baseline_stats is None or method_stats is None:
                continue
            b_mean, b_std = baseline_stats
            l_mean, l_std = method_stats
            rows.append(
                {
                    "section": section_name,
                    "metric": metric_name,
                    "opt": "lower" if direction == "lower" else "higher",
                    "baseline_mean": b_mean,
                    "baseline_std": b_std,
                    "lgfm_mean": l_mean,
                    "lgfm_std": l_std,
                    "baseline": fmt(b_mean, b_std, digits),
                    "lgfm": fmt(l_mean, l_std, digits),
                    "gain": gain_percent(b_mean, l_mean, direction),
                    "lgfm_better": is_lgfm_better(b_mean, l_mean, direction),
                }
            )
    return rows


def draw_table(rows: list[dict], output_base: Path) -> None:
    display_rows = []
    current_section = None
    for row in rows:
        if row["section"] != current_section:
            display_rows.append({"kind": "section", "section": row["section"]})
            current_section = row["section"]
        display_rows.append({"kind": "metric", **row})

    n_rows = len(display_rows)
    fig_h = max(10.0, 0.36 * n_rows + 1.7)
    fig, ax = plt.subplots(figsize=(13.6, fig_h))
    ax.set_axis_off()

    left = 0.035
    right = 0.975
    top = 0.955
    bottom = 0.055
    usable_h = top - bottom
    dy = usable_h / (n_rows + 2)

    x_section = left
    x_metric = 0.185
    x_opt = 0.405
    x_baseline = 0.555
    x_lgfm = 0.765
    x_gain = 0.93

    title = f"Guo+Threshold Evaluation (20 runs): Baseline vs. {METHOD_LABEL}"
    ax.text(
        0.5,
        0.99,
        title,
        ha="center",
        va="top",
        fontsize=15,
        fontweight="bold",
        transform=ax.transAxes,
    )

    y = top
    ax.hlines(y + dy * 0.38, left, right, linewidth=1.6, color="black", transform=ax.transAxes)
    headers = [
        (x_section, "Subset", "left"),
        (x_metric, "Metric", "left"),
        (x_opt, "Opt.", "center"),
        (x_baseline, "Baseline", "center"),
        (x_lgfm, METHOD_LABEL, "center"),
        (x_gain, "Gain", "right"),
    ]
    for x, text, ha in headers:
        ax.text(x, y, text, ha=ha, va="center", fontsize=11.5, fontweight="bold", transform=ax.transAxes)
    ax.hlines(y - dy * 0.48, left, right, linewidth=1.1, color="black", transform=ax.transAxes)
    y -= dy

    shade = False
    for item in display_rows:
        if item["kind"] == "section":
            y -= dy * 0.13
            ax.hlines(y + dy * 0.47, left, right, linewidth=0.8, color="black", transform=ax.transAxes)
            ax.text(
                x_section,
                y,
                item["section"],
                ha="left",
                va="center",
                fontsize=11.2,
                fontweight="bold",
                transform=ax.transAxes,
            )
            y -= dy
            shade = False
            continue

        if shade:
            ax.axhspan(
                y - dy * 0.45,
                y + dy * 0.45,
                xmin=left,
                xmax=right,
                color="#f5f5f5",
                linewidth=0,
                transform=ax.transAxes,
            )
        shade = not shade

        b_weight = "bold" if not item["lgfm_better"] else "normal"
        l_weight = "bold" if item["lgfm_better"] else "normal"
        gain = item["gain"]
        gain_text = f"{gain:+.1f}%"

        ax.text(x_metric, y, item["metric"], ha="left", va="center", fontsize=10.5, transform=ax.transAxes)
        ax.text(x_opt, y, item["opt"], ha="center", va="center", fontsize=10.2, transform=ax.transAxes)
        ax.text(
            x_baseline,
            y,
            item["baseline"],
            ha="center",
            va="center",
            fontsize=10.2,
            fontweight=b_weight,
            transform=ax.transAxes,
        )
        ax.text(
            x_lgfm,
            y,
            item["lgfm"],
            ha="center",
            va="center",
            fontsize=10.2,
            fontweight=l_weight,
            transform=ax.transAxes,
        )
        ax.text(x_gain, y, gain_text, ha="right", va="center", fontsize=10.2, transform=ax.transAxes)
        y -= dy

    ax.hlines(y + dy * 0.35, left, right, linewidth=1.6, color="black", transform=ax.transAxes)
    footnote = f"Bold indicates the better method for each metric. Gain is directional improvement of {METHOD_LABEL} over Baseline."
    ax.text(left, bottom * 0.62, footnote, ha="left", va="center", fontsize=9.2, transform=ax.transAxes)

    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_latex(rows: list[dict], output_path: Path) -> None:
    lines = [
        "\\begin{tabular}{lllccc}",
        "\\toprule",
        f"Subset & Metric & Opt. & Baseline & {METHOD_LABEL} & Gain " + r"\\",
        "\\midrule",
    ]
    current_section = None
    for row in rows:
        if row["section"] != current_section:
            if current_section is not None:
                lines.append("\\midrule")
            current_section = row["section"]
            section = current_section
        else:
            section = ""

        base = row["baseline"]
        lgfm = row["lgfm"]
        if row["lgfm_better"]:
            lgfm = f"\\textbf{{{lgfm}}}"
        else:
            base = f"\\textbf{{{base}}}"
        lines.append(
            f"{section} & {row['metric']} & {row['opt']} & "
            f"{base} & {lgfm} & {row['gain']:+.1f}\\% \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "",
            "% Bold indicates the better method for each metric.",
            f"% Gain is directional improvement of {METHOD_LABEL} over Baseline.",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    summary = pd.DataFrame(rows).drop(columns=["baseline", "lgfm", "lgfm_better"])
    summary.to_csv(OUT_DIR / "guo_threshold_20_baseline_vs_lgfm_summary.csv", index=False)

    output_base = OUT_DIR / "guo_threshold_20_baseline_vs_lgfm_three_line_table"
    draw_table(rows, output_base)
    write_latex(rows, output_base.with_suffix(".tex"))
    print(output_base.with_suffix(".png"))
    print(output_base.with_suffix(".pdf"))
    print(output_base.with_suffix(".tex"))


if __name__ == "__main__":
    main()
