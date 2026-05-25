# LNS：带时间惩罚双载荷网格配送问题

这是一个围绕 **Large Neighborhood Search (LNS)** 的课程/论文项目，内容包括：

- 固定迭代预算实验
- 统一时间预算实验
- LNS 扩展消融实验
- 论文源码与最终 PDF
- 结果表格、图像与绘图脚本

最终论文位于 [paper/Paper.pdf](paper/Paper.pdf)，论文源码位于 [paper/latex.tex](paper/latex.tex)。

## 目录结构

```text
LNS/
├─ experiments.cpp                 # 固定迭代预算实验
├─ improved_experiments.cpp        # 统一时间预算 / 扩展消融实验
├─ run_improved_experiments.ps1    # 统一时间预算实验脚本
├─ results.py                      # 读取 CSV 并生成论文图片
├─ paper/
│  ├─ latex.tex                    # 论文 LaTeX 源码
│  └─ Paper.pdf                    # 最终论文 PDF
└─ results/
   ├─ csv/                         # 实验数据
   ├─ figures/                     # 论文图片
   ├─ latex_tables/                # 自动生成的 LaTeX 表格
   └─ experiment_config.txt        # 最近一次 improved 实验配置
```

## 仓库内容说明

- `experiments.cpp`
  用于生成固定迭代预算下的 `table1_overall.csv` 到 `table8_paired_results.csv`。

- `improved_experiments.cpp`
  用于生成统一时间预算与扩展消融相关结果，例如：
  - `results/csv/summary.csv`
  - `results/csv/per_instance_results.csv`
  - `results/csv/ablation_results.csv`
  - `results/latex_tables/*.tex`

- `results.py`
  负责把 `results/csv/` 下的结果重新绘制为论文使用的中文图片。

## 快速使用

### 1. 运行统一时间预算 / 扩展消融实验

```powershell
powershell -ExecutionPolicy Bypass -File .\run_improved_experiments.ps1
```

默认参数：

- 每个 `k` 生成 20 个实例
- 每个实例重复 3 次
- `k = 16,24,32,40,48,64`
- 每次运行预算 300 ms
- 随机种子 `20260524`

### 2. 重新生成论文图片

```powershell
python .\results.py
```

脚本会读取 `results/csv/`，输出到 `results/figures/`。

### 3. 编译论文

论文主文件是：

```text
paper/latex.tex
```

项目使用中文 LaTeX，建议用 `xelatex` 或支持中文的 Tectonic/TeX 环境编译。

## 依赖

- C++17 编译器（如 `g++`）
- PowerShell
- Python 3
- Python 包：
  - `pandas`
  - `matplotlib`

可选：

- 支持中文的 LaTeX 编译环境，用于重新生成论文 PDF

## 已保留 / 已删除

为了便于上传 GitHub，本仓库保留了：

- 源码
- 论文源文件与最终 PDF
- 已生成的结果 CSV / 图像 / LaTeX 表格

并删除了不必要的本地编译产物，例如：

- `experiments.exe`
- `improved_experiments.exe`
- 临时说明和调试文件

## 说明

这个仓库当前更偏向“论文复现包”：

- 可以直接查看论文与结果
- 也可以重新运行实验并重绘图片
- 不依赖二进制文件即可重新构建主要内容
