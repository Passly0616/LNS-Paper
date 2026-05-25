"""
results.py

读取 results/csv 下的实验 CSV，重新生成论文所需的全部图片。
图片标题、坐标轴和图例均使用中文。
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager


BASE_DIR = Path(__file__).resolve().parent
CSV_DIR = BASE_DIR / "results" / "csv"
OUT_DIR = BASE_DIR / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PLOT_DPI = 300

SERIES_LABELS = {
    "EDD目标值": "EDD贪心",
    "匹配EDD目标值": "匹配EDD",
    "局部搜索目标值": "局部搜索",
    "模拟退火目标值": "模拟退火",
    "LNS目标值": "LNS",
    "LNS总距离": "LNS总距离",
    "LNS总迟到时间": "LNS总迟到时间",
    "精确最优值": "精确最优",
    "匹配EDD最优差距百分比": "匹配EDD",
    "模拟退火最优差距百分比": "模拟退火",
    "LNS最优差距百分比": "LNS",
}

SERIES_STYLES = {
    "EDD目标值": {"color": "#7A7A7A", "marker": "o", "linewidth": 2.0},
    "匹配EDD目标值": {"color": "#3A86FF", "marker": "s", "linewidth": 2.0},
    "局部搜索目标值": {"color": "#2A9D8F", "marker": "^", "linewidth": 2.0},
    "模拟退火目标值": {"color": "#F4A261", "marker": "D", "linewidth": 2.0},
    "LNS目标值": {"color": "#D1495B", "marker": "o", "linewidth": 2.4},
    "LNS总距离": {"color": "#4D908E", "marker": "o", "linewidth": 2.2},
    "LNS总迟到时间": {"color": "#BC4749", "marker": "s", "linewidth": 2.2},
    "精确最优值": {"color": "#222222", "marker": "o", "linewidth": 2.4},
    "匹配EDD最优差距百分比": {"color": "#3A86FF", "marker": "s", "linewidth": 2.0},
    "模拟退火最优差距百分比": {"color": "#F4A261", "marker": "D", "linewidth": 2.0},
    "LNS最优差距百分比": {"color": "#D1495B", "marker": "o", "linewidth": 2.4},
}

METHOD_COLORS = {
    "EDD贪心": "#7A7A7A",
    "最近邻贪心": "#6C757D",
    "匹配EDD": "#3A86FF",
    "局部搜索": "#2A9D8F",
    "模拟退火": "#F4A261",
    "LNS": "#D1495B",
}

BUDGET_ALGO_LABELS = {
    "LNS": "LNS",
    "SA": "模拟退火",
    "LocalSearch": "局部搜索",
}

BUDGET_ALGO_STYLES = {
    "LNS": {"color": "#D1495B", "marker": "o", "linewidth": 2.4},
    "SA": {"color": "#F4A261", "marker": "D", "linewidth": 2.1},
    "LocalSearch": {"color": "#2A9D8F", "marker": "^", "linewidth": 2.1},
}

ABLATION_LABELS = {
    "destroy_ratio_0.15": "破坏比例 0.15",
    "destroy_ratio_0.45": "破坏比例 0.45",
    "full": "完整 LNS",
    "LNS": "完整 LNS",
    "no_adaptive_weight": "无自适应权重",
    "no_local_search": "无局部强化",
    "no_probability_acceptance": "无概率接受",
    "no_regret": "无后悔插入",
}


def configure_matplotlib() -> None:
    preferred_fonts = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
    ]
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    chosen_font = next((font for font in preferred_fonts if font in available_fonts), "DejaVu Sans")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [chosen_font, *preferred_fonts, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": PLOT_DPI,
            "savefig.dpi": PLOT_DPI,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )
    print(f"[字体] 使用 {chosen_font}")


def read_csv(name: str) -> pd.DataFrame | None:
    path = CSV_DIR / name
    if not path.exists():
        path = BASE_DIR / name
    if not path.exists():
        print(f"[跳过] 未找到 {name}")
        return None
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def save_current(stem: str, formats: Sequence[str] = ("png",)) -> None:
    plt.tight_layout()
    for fmt in formats:
        path = OUT_DIR / f"{stem}.{fmt}"
        plt.savefig(path, bbox_inches="tight")
        print(f"[保存] {path}")
    plt.close()


def sort_by_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    return df.sort_values(column).reset_index(drop=True)


def line_plot(
    df: pd.DataFrame,
    x_col: str,
    y_cols: Iterable[str],
    title: str,
    xlabel: str,
    ylabel: str,
    stem: str,
    formats: Sequence[str] = ("png",),
) -> None:
    plot_df = sort_by_column(df, x_col)
    plt.figure(figsize=(7.6, 4.8))
    for col in y_cols:
        if col not in plot_df.columns:
            continue
        style = SERIES_STYLES.get(col, {"marker": "o", "linewidth": 2.0})
        label = SERIES_LABELS.get(col, col)
        plt.plot(plot_df[x_col], plot_df[col], label=label, markersize=6, **style)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend(frameon=False)
    save_current(stem, formats=formats)


def bar_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    stem: str,
    label_map: dict[str, str] | None = None,
    color_map: dict[str, str] | None = None,
    formats: Sequence[str] = ("png",),
) -> None:
    plot_df = df.copy()
    raw_labels = plot_df[x_col].astype(str).tolist()
    display_labels = [(label_map or {}).get(label, label) for label in raw_labels]
    colors = [(color_map or {}).get(label, "#4C78A8") for label in raw_labels]

    plt.figure(figsize=(7.8, 4.8))
    positions = range(len(plot_df))
    plt.bar(positions, plot_df[y_col], color=colors)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(list(positions), display_labels, rotation=18, ha="right")
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    save_current(stem, formats=formats)


def annotate_bar_values(positions: Sequence[float], values: Sequence[float], decimals: int = 2) -> None:
    if not values:
        return
    value_list = list(values)
    spread = max(value_list) - min(value_list)
    offset = max(spread * 0.03, max(abs(v) for v in value_list) * 0.015, 0.02)
    for x_pos, value in zip(positions, value_list):
        va = "bottom" if value >= 0 else "top"
        y_pos = value + offset if value >= 0 else value - offset
        plt.text(
            x_pos,
            y_pos,
            f"{value:.{decimals}f}",
            ha="center",
            va=va,
            fontsize=9,
        )


def plot_relative_ablation(df: pd.DataFrame, stem: str) -> None:
    if not {"方法", "目标值"}.issubset(df.columns):
        return

    method_map = {
        "LNS-小破坏": "小破坏",
        "LNS-标准": "标准版",
        "LNS-大破坏": "大破坏",
        "LNS-无概率接受": "无概率接受",
    }
    baseline_rows = df[df["方法"].astype(str).str.contains("标准", na=False)]
    if baseline_rows.empty:
        print("[跳过] 未找到 fig08 的标准版基线")
        return

    baseline = float(baseline_rows.iloc[0]["目标值"])
    plot_df = df.copy()
    plot_df["显示名称"] = plot_df["方法"].astype(str).map(method_map).fillna(plot_df["方法"].astype(str))
    plot_df["相对标准版变化百分比"] = (plot_df["目标值"] - baseline) / baseline * 100.0
    order = ["小破坏", "标准版", "大破坏", "无概率接受"]
    plot_df["_order"] = plot_df["显示名称"].apply(lambda x: order.index(x) if x in order else 999)
    plot_df = plot_df.sort_values("_order").reset_index(drop=True)

    values = plot_df["相对标准版变化百分比"].tolist()
    positions = list(range(len(plot_df)))
    colors = ["#2A9D8F" if value < 0 else "#D1495B" if value > 0 else "#3A86FF" for value in values]

    plt.figure(figsize=(7.2, 4.6))
    plt.bar(positions, values, color=colors, width=0.62)
    plt.axhline(0, color="#444444", linewidth=1.0)
    annotate_bar_values(positions, values, decimals=3)
    margin = max(max(abs(v) for v in values) * 0.30, 0.08)
    plt.ylim(min(values) - margin, max(values) + margin)
    plt.title("LNS 消融：相对标准版的目标值变化")
    plt.xlabel("LNS 变体")
    plt.ylabel("相对标准版变化（%）")
    plt.xticks(positions, plot_df["显示名称"].tolist())
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    save_current(stem)


def box_plot(
    series_list: list[pd.Series],
    labels: list[str],
    title: str,
    ylabel: str,
    stem: str,
) -> None:
    plt.figure(figsize=(7.6, 4.8))
    plt.boxplot(series_list, tick_labels=labels, showmeans=True)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    save_current(stem)


def scatter_plot(
    x: pd.Series,
    y: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    stem: str,
) -> None:
    plt.figure(figsize=(6.0, 5.2))
    plt.scatter(x, y, color="#D1495B", alpha=0.8)
    low = min(x.min(), y.min())
    high = max(x.max(), y.max())
    plt.plot([low, high], [low, high], linestyle="--", linewidth=1.0, color="#666666")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    save_current(stem)


def write_statistical_summary(df: pd.DataFrame) -> None:
    required = {"LNS目标值", "模拟退火目标值", "局部搜索目标值", "匹配EDD目标值"}
    if not required.issubset(df.columns):
        return

    lines = ["配对统计摘要", "============", ""]
    comparisons = [
        ("LNS 相对模拟退火", "模拟退火目标值", "LNS目标值"),
        ("LNS 相对局部搜索", "局部搜索目标值", "LNS目标值"),
        ("LNS 相对匹配EDD", "匹配EDD目标值", "LNS目标值"),
    ]

    try:
        from scipy.stats import wilcoxon  # type: ignore
    except Exception:
        wilcoxon = None

    for name, base_col, lns_col in comparisons:
        diff = df[base_col] - df[lns_col]
        rel = diff / df[base_col].replace(0, pd.NA) * 100
        lines.append(name)
        lines.append(f"  平均绝对改进: {diff.mean():.4f}")
        lines.append(f"  中位数绝对改进: {diff.median():.4f}")
        lines.append(f"  平均相对改进(%): {rel.mean():.4f}")
        lines.append(f"  中位数相对改进(%): {rel.median():.4f}")
        if wilcoxon is not None:
            try:
                stat, p_value = wilcoxon(df[base_col], df[lns_col], alternative="greater")
                lines.append(f"  单侧 Wilcoxon 统计量: {stat:.4f}")
                lines.append(f"  单侧 Wilcoxon p 值: {p_value:.6g}")
            except Exception as exc:
                lines.append(f"  Wilcoxon 检验失败: {exc}")
        else:
            lines.append("  未安装 scipy，未计算 Wilcoxon p 值。")
        lines.append("")

    out = OUT_DIR / "paired_statistical_summary.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[保存] {out}")


def plot_legacy_figures() -> None:
    df1 = read_csv("table1_overall.csv")
    if df1 is not None and {"方法", "目标值"}.issubset(df1.columns):
        bar_plot(
            df1,
            "方法",
            "目标值",
            "总体目标值对比",
            "算法",
            "目标函数值 F",
            "fig01_overall_objective",
            color_map=METHOD_COLORS,
        )
        if "平均时间毫秒" in df1.columns:
            bar_plot(
                df1,
                "方法",
                "平均时间毫秒",
                "总体平均运行时间对比",
                "算法",
                "平均时间（毫秒）",
                "fig02_overall_runtime",
                color_map=METHOD_COLORS,
            )

    df2 = read_csv("table2_by_k.csv")
    if df2 is not None and "目标数" in df2.columns:
        line_plot(
            df2,
            "目标数",
            ["EDD目标值", "匹配EDD目标值", "局部搜索目标值", "模拟退火目标值", "LNS目标值"],
            "不同目标数下的目标值变化",
            "目标数 k",
            "目标函数值 F",
            "fig03_objective_by_k",
        )
        line_plot(
            df2,
            "目标数",
            ["LNS总迟到时间"],
            "不同目标数下 LNS 的总迟到时间",
            "目标数 k",
            "总迟到时间",
            "fig04_lns_tardiness_by_k",
        )

    df3 = read_csv("table3_by_beta.csv")
    if df3 is not None and "宽松系数" in df3.columns:
        line_plot(
            df3,
            "宽松系数",
            ["EDD目标值", "匹配EDD目标值", "局部搜索目标值", "模拟退火目标值", "LNS目标值"],
            "不同期限宽松系数下的目标值变化",
            "期限宽松系数 β",
            "目标函数值 F",
            "fig05_objective_by_beta",
        )

    df4 = read_csv("table4_by_lambda.csv")
    if df4 is not None and "惩罚系数" in df4.columns:
        line_plot(
            df4,
            "惩罚系数",
            ["LNS目标值", "模拟退火目标值"],
            "不同迟到惩罚系数下的目标值变化",
            "迟到惩罚系数 λ",
            "目标函数值 F",
            "fig06_objective_by_lambda",
        )
        line_plot(
            df4,
            "惩罚系数",
            ["LNS总距离", "LNS总迟到时间"],
            "LNS 的距离与迟到权衡",
            "迟到惩罚系数 λ",
            "数值",
            "fig07_lns_distance_tardiness_tradeoff",
        )

    df5 = read_csv("table5_ablation.csv")
    if df5 is not None and {"方法", "目标值"}.issubset(df5.columns):
        plot_relative_ablation(df5, "fig08_lns_ablation")

    df6 = read_csv("table6_extended_k.csv")
    if df6 is not None and "目标数" in df6.columns:
        line_plot(
            df6,
            "目标数",
            ["EDD目标值", "匹配EDD目标值", "局部搜索目标值", "模拟退火目标值", "LNS目标值"],
            "扩展规模实验：更大目标数",
            "目标数 k",
            "目标函数值 F",
            "fig09_extended_k_objective",
        )

    df7 = read_csv("table7_exact_gap.csv")
    if df7 is not None and "目标数" in df7.columns:
        line_plot(
            df7,
            "目标数",
            ["精确最优值", "匹配EDD目标值", "模拟退火目标值", "LNS目标值"],
            "小规模实例与精确最优值对比",
            "目标数 k",
            "目标函数值 F",
            "fig10_exact_optimum_comparison",
        )
        line_plot(
            df7,
            "目标数",
            ["匹配EDD最优差距百分比", "模拟退火最优差距百分比", "LNS最优差距百分比"],
            "小规模实例的最优差距",
            "目标数 k",
            "相对精确最优差距（%）",
            "fig11_optimality_gap_pct",
        )

    df8 = read_csv("table8_paired_results.csv")
    if df8 is not None:
        if {
            "LNS相对模拟退火提升百分比",
            "LNS相对局部搜索提升百分比",
            "LNS相对匹配EDD提升百分比",
        }.issubset(df8.columns):
            box_plot(
                [
                    df8["LNS相对模拟退火提升百分比"],
                    df8["LNS相对局部搜索提升百分比"],
                    df8["LNS相对匹配EDD提升百分比"],
                ],
                ["相对模拟退火", "相对局部搜索", "相对匹配EDD"],
                "LNS 的配对相对改进分布",
                "相对改进（%）",
                "fig12_lns_paired_improvement_boxplot",
            )
        if {"模拟退火目标值", "LNS目标值"}.issubset(df8.columns):
            scatter_plot(
                df8["模拟退火目标值"],
                df8["LNS目标值"],
                "LNS 与模拟退火的配对目标值对比",
                "模拟退火目标值",
                "LNS 目标值",
                "fig13_lns_vs_sa_scatter",
            )
        write_statistical_summary(df8)


def plot_budget_and_ablation_from_summary() -> None:
    summary = read_csv("summary.csv")
    if summary is None:
        return

    budget = summary[summary["experiment"] == "budget"].copy()
    if not budget.empty:
        plt.figure(figsize=(7.8, 4.8))
        for algo in ["LNS", "SA", "LocalSearch"]:
            subset = sort_by_column(budget[budget["algorithm"] == algo], "k")
            if subset.empty:
                continue
            plt.plot(
                subset["k"],
                subset["mean_F"],
                label=BUDGET_ALGO_LABELS.get(algo, algo),
                markersize=6,
                **BUDGET_ALGO_STYLES.get(algo, {"marker": "o", "linewidth": 2.0}),
            )
        plt.title("统一时间预算下不同目标数的平均目标值")
        plt.xlabel("目标数 k")
        plt.ylabel("平均目标函数值 F")
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        plt.legend(frameon=False)
        save_current("budget_mean_objective_by_k", formats=("png", "svg"))

        plt.figure(figsize=(7.8, 4.8))
        for algo in ["LNS", "SA", "LocalSearch"]:
            subset = sort_by_column(budget[budget["algorithm"] == algo], "k")
            if subset.empty:
                continue
            plt.plot(
                subset["k"],
                subset["mean_gap_to_best_pct"],
                label=BUDGET_ALGO_LABELS.get(algo, algo),
                markersize=6,
                **BUDGET_ALGO_STYLES.get(algo, {"marker": "o", "linewidth": 2.0}),
            )
        plt.title("统一时间预算下相对最优结果的平均差距")
        plt.xlabel("目标数 k")
        plt.ylabel("平均差距（%）")
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        plt.legend(frameon=False)
        save_current("budget_gap_to_best_by_k")

    ablation = summary[summary["experiment"] == "ablation"].copy()
    if not ablation.empty:
        ablation = ablation.sort_values("mean_F").reset_index(drop=True)
        raw_labels = ablation["variant"].astype(str).tolist()
        labels = [ABLATION_LABELS.get(label, label) for label in raw_labels]
        colors = ["#D1495B" if label in {"full", "LNS"} else "#8D99AE" for label in raw_labels]

        plt.figure(figsize=(8.4, 4.8))
        positions = range(len(ablation))
        gap_values = ablation["mean_gap_to_best_pct"].tolist()
        plt.bar(positions, gap_values, color=colors)
        annotate_bar_values(list(positions), gap_values, decimals=2)
        plt.title("统一时间预算下 LNS 扩展消融平均差距")
        plt.xlabel("变体")
        plt.ylabel("相对同实例最优结果的平均差距（%）")
        plt.xticks(list(positions), labels, rotation=18, ha="right")
        plt.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
        save_current("ablation_k32", formats=("png", "svg"))


def main() -> None:
    configure_matplotlib()
    plot_legacy_figures()
    plot_budget_and_ablation_from_summary()
    print("完成。所有图片和统计摘要已重新生成到 results/figures。")


if __name__ == "__main__":
    main()
