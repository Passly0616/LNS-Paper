"""
results.py

读取 experiments.cpp 生成的 CSV，并生成全中文论文图片和统计摘要。
用法：
    python results.py
输出：
    figures/图1_总体目标值对比.png 等
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "figures"
OUT_DIR.mkdir(exist_ok=True)

# 常见中文字体。没有某个字体时 matplotlib 会自动尝试下一个。
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC",
    "Arial Unicode MS", "DejaVu Sans"
]
plt.rcParams["axes.unicode_minus"] = False


def read_csv(name: str) -> pd.DataFrame | None:
    path = BASE_DIR / name
    if not path.exists():
        print(f"[跳过] 未找到 {name}")
        return None
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def save_current(name: str) -> None:
    path = OUT_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[保存] {path}")


def line_plot(df: pd.DataFrame, x_col: str, y_cols: Iterable[str], title: str, xlabel: str, ylabel: str, filename: str) -> None:
    plt.figure(figsize=(7.4, 4.8))
    for col in y_cols:
        if col in df.columns:
            plt.plot(df[x_col], df[col], marker="o", label=col.replace("目标值", ""))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend()
    save_current(filename)


def bar_plot(df: pd.DataFrame, x_col: str, y_col: str, title: str, xlabel: str, ylabel: str, filename: str) -> None:
    plt.figure(figsize=(7.4, 4.8))
    plt.bar(df[x_col].astype(str), df[y_col])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=20, ha="right")
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    save_current(filename)


def box_plot(series_list: list[pd.Series], labels: list[str], title: str, ylabel: str, filename: str) -> None:
    plt.figure(figsize=(7.4, 4.8))
    plt.boxplot(series_list, labels=labels, showmeans=True)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    save_current(filename)


def write_statistical_summary(df: pd.DataFrame) -> None:
    required = {"LNS目标值", "模拟退火目标值", "局部搜索目标值", "匹配EDD目标值"}
    if not required.issubset(df.columns):
        return
    lines: list[str] = []
    lines.append("配对统计摘要")
    lines.append("============")
    lines.append("")

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
                stat, p = wilcoxon(df[base_col], df[lns_col], alternative="greater")
                lines.append(f"  单侧 Wilcoxon 统计量: {stat:.4f}")
                lines.append(f"  单侧 Wilcoxon p 值: {p:.6g}")
            except Exception as exc:
                lines.append(f"  Wilcoxon 检验失败: {exc}")
        else:
            lines.append("  未安装 scipy，未计算 Wilcoxon p 值。")
        lines.append("")

    out = OUT_DIR / "配对统计摘要.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[保存] {out}")


def main() -> None:
    df1 = read_csv("table1_overall.csv")
    if df1 is not None and {"方法", "目标值"}.issubset(df1.columns):
        bar_plot(df1, "方法", "目标值", "总体目标值对比", "算法", "目标函数值 F", "图1_总体目标值对比.png")
        if "平均时间毫秒" in df1.columns:
            bar_plot(df1, "方法", "平均时间毫秒", "总体平均运行时间对比", "算法", "平均时间（毫秒）", "图2_总体运行时间对比.png")

    df2 = read_csv("table2_by_k.csv")
    if df2 is not None and "目标数" in df2.columns:
        line_plot(df2, "目标数", ["EDD目标值", "匹配EDD目标值", "局部搜索目标值", "模拟退火目标值", "LNS目标值"],
                  "不同目标数下的目标值变化", "目标数 k", "目标函数值 F", "图3_不同目标数目标值.png")
        line_plot(df2, "目标数", ["LNS总迟到时间"], "不同目标数下 LNS 的总迟到时间", "目标数 k", "总迟到时间", "图4_LNS迟到时间随目标数变化.png")

    df3 = read_csv("table3_by_beta.csv")
    if df3 is not None and "宽松系数" in df3.columns:
        line_plot(df3, "宽松系数", ["EDD目标值", "匹配EDD目标值", "局部搜索目标值", "模拟退火目标值", "LNS目标值"],
                  "不同期限宽松系数下的目标值变化", "期限宽松系数 β", "目标函数值 F", "图5_不同宽松系数目标值.png")

    df4 = read_csv("table4_by_lambda.csv")
    if df4 is not None and "惩罚系数" in df4.columns:
        line_plot(df4, "惩罚系数", ["LNS目标值", "模拟退火目标值"], "不同迟到惩罚系数下的目标值变化", "迟到惩罚系数 λ", "目标函数值 F", "图6_不同惩罚系数目标值.png")
        line_plot(df4, "惩罚系数", ["LNS总距离", "LNS总迟到时间"], "LNS 的距离—迟到权衡", "迟到惩罚系数 λ", "数值", "图7_LNS距离迟到权衡.png")

    df5 = read_csv("table5_ablation.csv")
    if df5 is not None and {"方法", "目标值"}.issubset(df5.columns):
        bar_plot(df5, "方法", "目标值", "LNS 消融实验", "LNS 变体", "目标函数值 F", "图8_LNS消融实验.png")

    df6 = read_csv("table6_extended_k.csv")
    if df6 is not None and "目标数" in df6.columns:
        line_plot(df6, "目标数", ["EDD目标值", "匹配EDD目标值", "局部搜索目标值", "模拟退火目标值", "LNS目标值"],
                  "扩展规模实验：更大目标数", "目标数 k", "目标函数值 F", "图9_扩展规模目标值.png")

    df7 = read_csv("table7_exact_gap.csv")
    if df7 is not None and "目标数" in df7.columns:
        line_plot(df7, "目标数", ["精确最优值", "匹配EDD目标值", "模拟退火目标值", "LNS目标值"],
                  "小规模实例与精确最优值对比", "目标数 k", "目标函数值 F", "图10_精确最优值对比.png")
        line_plot(df7, "目标数", ["匹配EDD最优差距百分比", "模拟退火最优差距百分比", "LNS最优差距百分比"],
                  "小规模实例的最优差距", "目标数 k", "相对精确最优差距（%）", "图11_最优差距百分比.png")

    df8 = read_csv("table8_paired_results.csv")
    if df8 is not None:
        if {"LNS相对模拟退火提升百分比", "LNS相对局部搜索提升百分比", "LNS相对匹配EDD提升百分比"}.issubset(df8.columns):
            box_plot([df8["LNS相对模拟退火提升百分比"], df8["LNS相对局部搜索提升百分比"], df8["LNS相对匹配EDD提升百分比"]],
                     ["相对模拟退火", "相对局部搜索", "相对匹配EDD"], "LNS 的配对相对改进分布", "相对改进（%）", "图12_LNS配对改进箱线图.png")
        if {"模拟退火目标值", "LNS目标值"}.issubset(df8.columns):
            plt.figure(figsize=(5.8, 5.2))
            plt.scatter(df8["模拟退火目标值"], df8["LNS目标值"])
            low = min(df8["模拟退火目标值"].min(), df8["LNS目标值"].min())
            high = max(df8["模拟退火目标值"].max(), df8["LNS目标值"].max())
            plt.plot([low, high], [low, high], linestyle="--", linewidth=1)
            plt.title("LNS 与模拟退火的配对目标值对比")
            plt.xlabel("模拟退火目标值")
            plt.ylabel("LNS 目标值")
            plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
            save_current("图13_LNS与模拟退火散点对比.png")
        write_statistical_summary(df8)

    print("完成。图片和统计摘要已保存在 figures 文件夹。")


if __name__ == "__main__":
    main()
