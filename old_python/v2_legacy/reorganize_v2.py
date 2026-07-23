"""
reorganize_v2.py
================
1. 將 results/v2_20k 依照 safety_guardrails_evaluation 的分法重新分類
2. 產出兩版本結構對照報告 (PDF)

映射關係:
  safety_guardrails_evaluation/          v2_20k/ (整理後)
  ─────────────────────────────────────  ───────────────────────────────────────
  [root] 4 root png                      → v2_20k/                 (layer PDFs)
  cache/baseline|split                   → cache/ (保留)
  data_align/baseline|split/
    01_Metrics_Trends/test1|test2        → 01_Unified_Training/Model_Comparison/
                                           + 02_Safety_Evaluation/02_Metric_Trends/
    02_Reliability_Curves/layer_X        → 02_Safety_Evaluation/01_Reliability_Diagrams/
    02_Reliability_Curves_split_y/lay.   → (same, split by Y)
    03_Quadrant_Histograms/y1|y2|y3     → 02_Safety_Evaluation/03_Quadrant_Histograms/
    04_Brier_Components/                 → 02_Safety_Evaluation/04_Brier_Components/
  data_aug/split/ (same sub-structure)

v2_20k 新結構:
  v2_20k/
    00_Reports/                     ← layer PDFs
    01_Model_Training/
      01_Model_Comparison/          ← model_comparison_layer_X_yX.png
      02_ROC_Curves/                ← roc_curve_layer_X_yX.png
    02_Reliability_Evaluation/
      baseline/
        01_Metrics_Trends/
        02_Reliability_Diagrams/    ← per layer_X
        03_Quadrant_Histograms/y1|y2|y3/layer_X/
        04_Brier_Components/        ← per layer_X
        05_Bimodal_KDE_Histograms/  ← per layer_X
      split/  (same)
    cache/
"""

import os, shutil, re, sys
import pandas as pd

SRC  = r"results/v2_20k"
DST  = r"results/v2_20k_organized"

# ── 如果目的地存在先刪掉 ──────────────────────────────────
if os.path.exists(DST):
    shutil.rmtree(DST)

moved = 0
errors = []

def mv(src_path, dst_path):
    global moved
    if not os.path.exists(src_path):
        return
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copy2(src_path, dst_path)
    moved += 1

def mv_dir(src_dir, dst_dir):
    """複製整個目錄"""
    if not os.path.exists(src_dir):
        return
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)
    global moved
    cnt = sum(len(f) for _, _, f in os.walk(src_dir))
    moved += cnt
    print(f"  [DIR] {src_dir} -> {dst_dir}  ({cnt} files)")

print("=" * 60)
print("  reorganize_v2: 整理 v2_20k 資料夾結構")
print("=" * 60)

# ── 1. Reports (PDFs) ────────────────────────────────────────
print("\n[1] PDF 報告 -> 00_Reports/")
for f in os.listdir(SRC):
    if f.endswith(".pdf"):
        mv(f"{SRC}/{f}", f"{DST}/00_Reports/{f}")
        print(f"  {f}")

# ── 2. Model Training ────────────────────────────────────────
print("\n[2] 模型訓練圖 -> 01_Model_Training/")
for f in os.listdir(f"{SRC}/01_Unified_Training/Model_Comparison"):
    mv(f"{SRC}/01_Unified_Training/Model_Comparison/{f}",
       f"{DST}/01_Model_Training/01_Model_Comparison/{f}")
for f in os.listdir(f"{SRC}/01_Unified_Training/ROC_Curves"):
    mv(f"{SRC}/01_Unified_Training/ROC_Curves/{f}",
       f"{DST}/01_Model_Training/02_ROC_Curves/{f}")
mc = len(os.listdir(f"{SRC}/01_Unified_Training/Model_Comparison"))
rc = len(os.listdir(f"{SRC}/01_Unified_Training/ROC_Curves"))
print(f"  Model_Comparison: {mc} files  |  ROC_Curves: {rc} files")

# ── 3. Safety Evaluation ─────────────────────────────────────
print("\n[3] 安全評估圖 -> 02_Reliability_Evaluation/")

for mode in ["baseline", "split"]:
    mode_src = f"{SRC}/02_Safety_Evaluation"
    mode_dst = f"{DST}/02_Reliability_Evaluation/{mode}"

    # 3a. Metrics Trends
    src_mt = f"{mode_src}/02_Metric_Trends/{mode}"
    dst_mt = f"{mode_dst}/01_Metrics_Trends"
    if os.path.exists(src_mt):
        mv_dir(src_mt, dst_mt)

    # 3b. Reliability Diagrams
    src_rd = f"{mode_src}/01_Reliability_Diagrams/{mode}"
    if os.path.exists(src_rd):
        # flatten: files are directly inside (not per layer)
        # safety_guardrails has layer_X subfolder per test-set
        # v2 has directly the png files → keep flat but rename
        for f in os.listdir(src_rd):
            fp = f"{src_rd}/{f}"
            if os.path.isfile(fp):
                mv(fp, f"{mode_dst}/02_Reliability_Diagrams/{f}")
        print(f"  [{mode}] Reliability_Diagrams: {len(os.listdir(src_rd))} files")

    # 3c. Quadrant Histograms  (v2: mode/test/layer → dst: y/test/layer)
    src_qh = f"{mode_src}/03_Quadrant_Histograms/{mode}"
    if os.path.exists(src_qh):
        for test in os.listdir(src_qh):
            test_path = f"{src_qh}/{test}"
            if not os.path.isdir(test_path): continue
            for layer in os.listdir(test_path):
                layer_path = f"{test_path}/{layer}"
                if not os.path.isdir(layer_path): continue
                for f in os.listdir(layer_path):
                    # 檔名格式: layer_X_testX_yX_bars.png
                    # 依 y 分群
                    m = re.search(r'_(y[123])_', f)
                    y_tag = m.group(1) if m else "other"
                    dst_qh = f"{mode_dst}/03_Quadrant_Histograms/{y_tag}/{test}/{layer}/{f}"
                    mv(f"{layer_path}/{f}", dst_qh)
        # count
        cnt = sum(len(fls) for _, _, fls in os.walk(src_qh))
        print(f"  [{mode}] Quadrant_Histograms: {cnt} files")

    # 3d. Brier Components  (v2: mode/layer → dst: mode/test/layer)
    src_bc = f"{mode_src}/04_Brier_Components/{mode}"
    if os.path.exists(src_bc):
        for f in os.listdir(src_bc):
            fp = f"{src_bc}/{f}"
            if not os.path.isfile(fp): continue
            # 檔名: layer_X_testX_yX_brier_*.png
            m_test  = re.search(r'layer_\d+_(test\d)_', f)
            m_layer = re.search(r'(layer_\d+)_', f)
            test_tag  = m_test.group(1)  if m_test  else "test1"
            layer_tag = m_layer.group(1) if m_layer else "layer_1"
            dst_bc = f"{mode_dst}/04_Brier_Components/{test_tag}/{layer_tag}/{f}"
            mv(fp, dst_bc)
        cnt = len(os.listdir(src_bc))
        print(f"  [{mode}] Brier_Components: {cnt} files")

    # 3e. Bimodal KDE Histograms  (v2: mode/test/layer → same)
    src_kde = f"{mode_src}/05_Bimodal_KDE_Histograms/{mode}"
    if os.path.exists(src_kde):
        for test in os.listdir(src_kde):
            test_path = f"{src_kde}/{test}"
            if not os.path.isdir(test_path): continue
            for layer in os.listdir(test_path):
                layer_path = f"{test_path}/{layer}"
                if not os.path.isdir(layer_path): continue
                for f in os.listdir(layer_path):
                    mv(f"{layer_path}/{f}",
                       f"{mode_dst}/05_Bimodal_KDE_Histograms/{test}/{layer}/{f}")
        cnt = sum(len(fls) for _, _, fls in os.walk(src_kde))
        print(f"  [{mode}] Bimodal_KDE_Histograms: {cnt} files")

# ── 4. Cache ─────────────────────────────────────────────────
print("\n[4] Cache -> cache/")
mv_dir(f"{SRC}/02_Safety_Evaluation/cache", f"{DST}/cache")

print(f"\n{'='*60}")
print(f"  完成！共複製 {moved} 個檔案")
print(f"  輸出目錄: {DST}")
print(f"{'='*60}")

# ── 產出結構對照報告 ─────────────────────────────────────────
print("\n產出版本結構對照報告 ...")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def tree_lines(path, prefix="", max_depth=4, cur_depth=0):
    """回傳 (indent, name, is_dir, count) 的 list"""
    if cur_depth > max_depth or not os.path.exists(path):
        return []
    items = sorted(os.listdir(path))
    result = []
    for i, item in enumerate(items):
        fp = os.path.join(path, item)
        is_last = (i == len(items) - 1)
        connector = "└── " if is_last else "├── "
        child_prefix = prefix + ("    " if is_last else "│   ")
        is_dir = os.path.isdir(fp)
        if is_dir:
            n_files = sum(len(f) for _, _, f in os.walk(fp))
            result.append((prefix + connector, item + "/", True, n_files))
            if cur_depth < max_depth:
                result.extend(tree_lines(fp, child_prefix, max_depth, cur_depth+1))
        else:
            result.append((prefix + connector, item, False, None))
    return result

def draw_tree(ax, root, title, max_depth=3, color_dir="#2C3E50", color_file="#5D8AA8"):
    ax.axis("off")
    ax.set_title(title, fontsize=9, fontweight="bold", color="#2C3E50",
                 loc="left", pad=6)
    lines = tree_lines(root, max_depth=max_depth)
    # header
    root_name = os.path.basename(root) + "/"
    n_total = sum(len(f) for _, _, f in os.walk(root))
    all_lines = [(f"📁 {root_name}", True, n_total)] + \
                [(f"{p}{n}", d, c) for p, n, d, c in lines]

    y = 0.97
    line_h = min(0.030, 0.97 / max(len(all_lines), 1))
    for item in all_lines:
        text, is_dir, cnt = item
        clr = color_dir if is_dir else color_file
        suffix = f"  ({cnt} files)" if is_dir and cnt is not None else ""
        ax.text(0.01, y, text + suffix,
                transform=ax.transAxes, fontsize=6.2,
                color=clr, fontfamily="monospace", va="top")
        y -= line_h
        if y < 0.02:
            ax.text(0.01, 0.01, "  ... (truncated)", transform=ax.transAxes,
                    fontsize=6, color="#999999")
            break

OUT_PDF = f"{DST}/version_comparison_report.pdf"
with PdfPages(OUT_PDF) as pdf:
    # ── 封面 ──────────────────────────────────────────────
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("#1A252F")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_facecolor("#1A252F")
    ax.text(0.5, 0.75, "Version Comparison Report",
            ha="center", va="center", fontsize=28,
            color="white", fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.62,
            "safety_guardrails_evaluation  vs  v2_20k (reorganized)",
            ha="center", va="center", fontsize=13,
            color="#AEC6CF", transform=ax.transAxes)

    # Stats
    v1_root = "results/safety_guardrails_evaluation"
    v2_root = DST
    v1_files = sum(len(f) for _, _, f in os.walk(v1_root))
    v2_files = sum(len(f) for _, _, f in os.walk(v2_root))

    stats = [
        ("Version",      "safety_guardrails_evaluation",    "v2_20k (reorganized)"),
        ("Dataset",      "data_aug  (augmented 85k)",       "pure 20k sampled"),
        ("Data Split",   "aligned / augmented / eval sets", "baseline / split"),
        ("Models",       "5 probe classifiers",             "5 probe classifiers"),
        ("Targets",      "Y1 / Y2 / Y3",                   "Y1 / Y2 / Y3"),
        ("Layers",       "1 – 6",                           "1 – 6"),
        ("Chart Types",  "4 types",                         "5 types (+KDE)"),
        ("Extra",        "—",                               "Layer PDFs + Log"),
        ("Total Files",  str(v1_files),                     str(v2_files)),
    ]
    table_y = 0.52
    col_x = [0.15, 0.42, 0.72]
    headers = ["", "V1 (safety_guardrails)", "V2 (v2_20k)"]
    for j, h in enumerate(headers):
        ax.text(col_x[j], table_y + 0.025, h,
                ha="left", va="center", fontsize=9, fontweight="bold",
                color="#FFD700", transform=ax.transAxes)
    ax.axhline(y=table_y + 0.015, xmin=0.12, xmax=0.9,
               color="#FFD700", lw=0.8)
    for row_i, row in enumerate(stats):
        y_pos = table_y - row_i * 0.038
        for col_i, val in enumerate(row):
            clr = "#CCDDDD" if col_i == 0 else "white"
            ax.text(col_x[col_i], y_pos, val,
                    ha="left", va="center", fontsize=8, color=clr,
                    transform=ax.transAxes)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ── 頁2: 結構對照 tree ──────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27))
    fig.patch.set_facecolor("#FAFAFA")
    fig.suptitle("Folder Structure Comparison", fontsize=13,
                 fontweight="bold", color="#2C3E50")

    draw_tree(axes[0], "results/safety_guardrails_evaluation",
              "V1: safety_guardrails_evaluation/", max_depth=3)
    draw_tree(axes[1], DST,
              "V2: v2_20k_organized/ (new structure)", max_depth=3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ── 頁3: 差異說明表 ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("#FAFAFA")
    ax.axis("off")
    ax.set_title("Key Differences: V1 vs V2", fontsize=13,
                 fontweight="bold", color="#2C3E50", pad=20)

    diff_rows = [
        ["面向", "V1 (safety_guardrails)", "V2 (v2_20k)", "說明"],
        ["資料集規模", "85k (augmented)", "20k (純淨抽樣)", "V2 排除 augment 污染"],
        ["訓練切分", "一個 CLF 訓練 Y1/Y2/Y3 共同", "分別訓練 Y1/Y2/Y3 各自 CLF", "V2 更針對性"],
        ["測試集設計", "aligned/augmented/eval", "test1(訓練)/test2(未見)", "V2 更嚴格的 OOD 測試"],
        ["圖表類型", "4 種", "5 種 (+KDE 雙峰圖)", "V2 多了分布可視化"],
        ["Log 詳細度", "基本", "詳細數值 log（不看圖也能讀）", "V2 每格都有統計量"],
        ["報告格式", "PNG 各自存放", "PNG + 每模型一份 PDF 報告", "V2 方便閱讀"],
        ["資料夾分類", "data_align / data_aug", "baseline / split", "概念對應，命名不同"],
        ["子類分層", "y1/y2/y3 → test → layer", "mode → test → layer → y", "V2 整理後對齊"],
        ["Model_Comparison", "01_Metrics_Trends 內", "獨立 01_Model_Training/", "V2 更清晰"],
        ["KDE 雙峰圖", "無", "05_Bimodal_KDE_Histograms/", "V2 新增"],
        ["ECE/Brier 趨勢", "有（per eval set）", "有（baseline+split）", "V2 保留"],
    ]

    col_widths = [0.14, 0.22, 0.28, 0.30]
    col_starts = [0.01, 0.15, 0.38, 0.66]
    row_h = 0.072
    y_start = 0.93

    for ri, row in enumerate(diff_rows):
        y_pos = y_start - ri * row_h
        is_header = (ri == 0)
        bg = "#2C3E50" if is_header else ("#F0F4F8" if ri % 2 == 0 else "white")
        rect = plt.Rectangle((0, y_pos - row_h * 0.5), 1, row_h,
                              transform=ax.transAxes, facecolor=bg, zorder=0)
        ax.add_patch(rect)
        for ci, cell in enumerate(row):
            clr = "white" if is_header else "#2C3E50"
            fs = 8 if not is_header else 8.5
            fw = "bold" if is_header or ci == 0 else "normal"
            # V2 欄加色
            if ci == 2 and not is_header:
                clr = "#1A6A3A"
            ax.text(col_starts[ci] + 0.005, y_pos,
                    cell, transform=ax.transAxes,
                    fontsize=fs, fontweight=fw, color=clr,
                    va="center", wrap=True)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # metadata
    d = pdf.infodict()
    d["Title"]   = "v2_20k vs safety_guardrails_evaluation Comparison"
    d["Author"]  = "Safety-Training Analysis"

print(f"\n報告已產出: {OUT_PDF}")
print("Done!")
