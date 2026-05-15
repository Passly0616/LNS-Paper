"""
plot_results_english_stronger.py

Read the CSV files exported by experiment_stronger_english.cpp and generate
manuscript-friendly English figures.

Usage:
    python plot_results_english_stronger.py

Place this script in the same folder as the generated CSV files. Figures and a
small statistical summary will be saved to ./figures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "figures"
OUT_DIR.mkdir(exist_ok=True)

# English-only plotting. No Chinese font fallback is used.
plt.rcParams["axes.unicode_minus"] = False

COLUMN_ALIASES: dict[str, str] = {}


LABELS = {
    "edd_objective": "EDD-Greedy",
    "nearest_greedy_objective": "Nearest-Greedy",
    "matching_edd_objective": "Matching+EDD",
    "local_search_objective": "Local Search",
    "sa_objective": "SA",
    "exact_optimum": "Exact optimum",
    "sa_total_lateness": "SA total lateness",
    "sa_distance": "SA distance",
}


def read_csv(name: str) -> pd.DataFrame | None:
    path = BASE_DIR / name
    if not path.exists():
        print(f"[skip] {name} not found")
        return None
    try:
        df = pd.read_csv(path)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.rename(columns={c: COLUMN_ALIASES.get(c, c) for c in df.columns})
    if "method" in df.columns:
        df["method"] = df["method"].replace({"StaticPair+Sort": "Matching+EDD"})
    return df


def save_current(name: str) -> None:
    path = OUT_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[saved] {path}")


def line_plot(
    df: pd.DataFrame,
    x_col: str,
    y_cols: Iterable[str],
    title: str,
    xlabel: str,
    ylabel: str,
    filename: str,
) -> None:
    plt.figure(figsize=(7.2, 4.6))
    for col in y_cols:
        if col in df.columns:
            plt.plot(df[x_col], df[col], marker="o", label=LABELS.get(col, col))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend()
    save_current(filename)


def bar_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    filename: str,
) -> None:
    plt.figure(figsize=(7.2, 4.6))
    plt.bar(df[x_col].astype(str), df[y_col])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=20, ha="right")
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    save_current(filename)


def box_plot(
    data: list[pd.Series],
    labels: list[str],
    title: str,
    ylabel: str,
    filename: str,
) -> None:
    plt.figure(figsize=(7.2, 4.6))
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    save_current(filename)


def write_statistical_summary(df: pd.DataFrame) -> None:
    required = {"sa_objective", "local_search_objective", "matching_edd_objective"}
    if not required.issubset(df.columns):
        return

    lines: list[str] = []
    lines.append("Paired statistical summary")
    lines.append("==========================")
    lines.append("")

    comparisons = [
        ("SA vs Local Search", "local_search_objective", "sa_objective"),
        ("SA vs Matching+EDD", "matching_edd_objective", "sa_objective"),
    ]

    try:
        from scipy.stats import wilcoxon  # type: ignore
    except Exception:
        wilcoxon = None

    for name, base_col, sa_col in comparisons:
        diff = df[base_col] - df[sa_col]
        rel = diff / df[base_col].replace(0, pd.NA) * 100
        lines.append(name)
        lines.append(f"  mean absolute improvement: {diff.mean():.4f}")
        lines.append(f"  median absolute improvement: {diff.median():.4f}")
        lines.append(f"  mean relative improvement (%): {rel.mean():.4f}")
        lines.append(f"  median relative improvement (%): {rel.median():.4f}")
        if wilcoxon is not None:
            try:
                stat, p_value = wilcoxon(df[base_col], df[sa_col], alternative="greater")
                lines.append(f"  one-sided Wilcoxon statistic: {stat:.4f}")
                lines.append(f"  one-sided Wilcoxon p-value: {p_value:.6g}")
            except Exception as exc:
                lines.append(f"  Wilcoxon test failed: {exc}")
        else:
            lines.append("  scipy is not installed; Wilcoxon p-value was not computed.")
        lines.append("")

    out = OUT_DIR / "paired_statistical_summary.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[saved] {out}")


def main() -> None:
    df1 = read_csv("table1_overall.csv")
    if df1 is not None and {"method", "objective"}.issubset(df1.columns):
        plot_df = df1[df1["method"] != "Distance-Reference"]
        plot_df = plot_df[plot_df["method"] != "Distance-LB"]
        bar_plot(
            plot_df,
            "method",
            "objective",
            "Overall objective comparison",
            "Method",
            "Objective value F",
            "fig1_overall_objective.png",
        )

    df2 = read_csv("table2_by_k.csv")
    if df2 is not None and "k" in df2.columns:
        line_plot(
            df2,
            "k",
            ["edd_objective", "matching_edd_objective", "local_search_objective", "sa_objective"],
            "Effect of target count",
            "Number of targets k",
            "Objective value F",
            "fig2_objective_by_k.png",
        )
        if "sa_total_lateness" in df2.columns:
            line_plot(
                df2,
                "k",
                ["sa_total_lateness"],
                "SA lateness under different target counts",
                "Number of targets k",
                "Total lateness",
                "fig3_sa_lateness_by_k.png",
            )

    df3 = read_csv("table3_by_beta.csv")
    if df3 is not None and "beta" in df3.columns:
        line_plot(
            df3,
            "beta",
            ["edd_objective", "matching_edd_objective", "local_search_objective", "sa_objective"],
            "Effect of deadline looseness",
            "Deadline looseness beta",
            "Objective value F",
            "fig4_objective_by_beta.png",
        )

    df4 = read_csv("table4_by_lambda.csv")
    if df4 is not None and "lambda" in df4.columns:
        line_plot(
            df4,
            "lambda",
            ["sa_objective", "matching_edd_objective"],
            "Effect of lateness penalty",
            "Lateness penalty lambda",
            "Objective value F",
            "fig5_objective_by_lambda.png",
        )
        line_plot(
            df4,
            "lambda",
            ["sa_distance", "sa_total_lateness"],
            "SA distance-lateness tradeoff",
            "Lateness penalty lambda",
            "Value",
            "fig6_sa_distance_lateness_tradeoff.png",
        )

    df5 = read_csv("table5_ablation.csv")
    if df5 is not None and {"method", "objective"}.issubset(df5.columns):
        bar_plot(
            df5,
            "method",
            "objective",
            "Neighborhood ablation study",
            "Variant",
            "Objective value F",
            "fig7_ablation_objective.png",
        )

    df6 = read_csv("table6_extended_k.csv")
    if df6 is not None and "k" in df6.columns:
        line_plot(
            df6,
            "k",
            ["edd_objective", "matching_edd_objective", "local_search_objective", "sa_objective"],
            "Extended experiment with larger target counts",
            "Number of targets k",
            "Objective value F",
            "fig8_extended_objective_by_k.png",
        )

    df7 = read_csv("table7_extended_beta.csv")
    if df7 is not None and "beta" in df7.columns:
        line_plot(
            df7,
            "beta",
            ["edd_objective", "matching_edd_objective", "local_search_objective", "sa_objective"],
            "Extended experiment with looser deadlines",
            "Deadline looseness beta",
            "Objective value F",
            "fig9_extended_objective_by_beta.png",
        )

    df8 = read_csv("table8_extended_lambda.csv")
    if df8 is not None and "lambda" in df8.columns:
        line_plot(
            df8,
            "lambda",
            ["sa_objective", "matching_edd_objective"],
            "Extended experiment with larger lateness penalties",
            "Lateness penalty lambda",
            "Objective value F",
            "fig10_extended_objective_by_lambda.png",
        )

    df9 = read_csv("table9_exact_gap.csv")
    if df9 is not None and "k" in df9.columns:
        line_plot(
            df9,
            "k",
            ["exact_optimum", "matching_edd_objective", "local_search_objective", "sa_objective"],
            "Small-scale comparison with exact optimum",
            "Number of targets k",
            "Objective value F",
            "fig11_exact_objective_comparison.png",
        )
        gap_cols = [c for c in ["matching_edd_gap_pct", "local_search_gap_pct", "sa_gap_pct"] if c in df9.columns]
        if gap_cols:
            line_plot(
                df9,
                "k",
                gap_cols,
                "Optimality gap on small instances",
                "Number of targets k",
                "Gap to exact optimum (%)",
                "fig12_exact_gap_pct.png",
            )

    df10 = read_csv("table10_paired_results.csv")
    if df10 is not None:
        if {"sa_vs_local_improvement_pct", "sa_vs_matching_edd_improvement_pct"}.issubset(df10.columns):
            box_plot(
                [df10["sa_vs_local_improvement_pct"], df10["sa_vs_matching_edd_improvement_pct"]],
                ["SA vs Local Search", "SA vs Matching+EDD"],
                "Paired relative improvement of SA",
                "Relative improvement (%)",
                "fig13_paired_improvement_boxplot.png",
            )
        if {"local_search_objective", "sa_objective"}.issubset(df10.columns):
            plt.figure(figsize=(5.8, 5.2))
            plt.scatter(df10["local_search_objective"], df10["sa_objective"])
            low = min(df10["local_search_objective"].min(), df10["sa_objective"].min())
            high = max(df10["local_search_objective"].max(), df10["sa_objective"].max())
            plt.plot([low, high], [low, high], linestyle="--", linewidth=1)
            plt.title("Paired objective comparison")
            plt.xlabel("Local Search objective")
            plt.ylabel("SA objective")
            plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
            save_current("fig14_paired_sa_vs_local_scatter.png")
        write_statistical_summary(df10)

    print("Done. Figures and summaries are in the figures folder.")


if __name__ == "__main__":
    main()
