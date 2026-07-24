"""
Step 4: 大圖拼接工具 (Combine Plots into Grid Overviews)
========================================================
將 step3 產出的單張 PNG 依各分析類型拼接成大圖總覽，存放於
results/plots/combined/ 底下。

用法:
    python evaluation_pipeline/step4_combine_plots.py [options]

選項:
    --chart   all | trends | trends_split_y | reliability_combined |
              quadrant_hist | score_hist | brier_components |
              step_mappings | joint_calibration
    --target  y1 | y2 | y3 | all         (預設 all)
    --split   test1 | test2 | eval | all  (預設 all；部分圖固定 test2)
    --model   SGD | MLP | LGB | LR | RF | all  (預設 all)

輸出目錄結構:
    results/plots/combined/
    ├── 01_Metrics_Trends/
    ├── 01_Metrics_Trends_split_y/
    ├── 02_Reliability_Curves_combined/
    ├── 03_Quadrant_Histograms/
    ├── 04_Score_Histograms/
    ├── 05_Brier_Components/
    ├── 06_Step_Mappings/
    └── 07_Joint_Calibration/
"""

import os
import sys
import argparse

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("錯誤: 需要 Pillow 套件。請執行: uv pip install pillow")
    sys.exit(1)

# ─── 路徑常數 ─────────────────────────────────────────────────────────────────

PLOTS_DIR    = "results/plots"
COMBINED_DIR = "results/plots/combined"

ALL_TARGETS = ['y1', 'y2', 'y3']
ALL_SPLITS  = ['test1', 'test2', 'eval']
ALL_LAYERS  = list(range(1, 7))
ALL_MODELS  = ['SGD', 'MLP', 'LGB', 'LR', 'RF']
ALL_GROUPS  = [0, 1]

# ─── 字型載入 ─────────────────────────────────────────────────────────────────

def _load_font(size: int = 20) -> ImageFont.FreeTypeFont:
    """嘗試載入 TrueType 字型；找不到時退回 PIL 預設字型。"""
    candidates = [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\Arial.ttf",
        r"C:\Windows\Fonts\calibri.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int]:
    """相容新舊版 Pillow 的文字尺寸查詢。"""
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        return draw.textsize(text, font=font)  # type: ignore[return-value]


# ─── 核心拼圖工具 ────────────────────────────────────────────────────────────

def _make_placeholder(w: int, h: int, text: str = "N/A") -> Image.Image:
    """產生灰色佔位圖（找不到來源圖時使用）。"""
    img  = Image.new('RGB', (w, h), color=(210, 210, 210))
    draw = ImageDraw.Draw(img)
    font = _load_font(20)
    tw, th = _text_size(draw, text, font)
    draw.text(((w - tw) // 2, (h - th) // 2), text, fill=(100, 100, 100), font=font)
    return img


def _draw_centered(draw: ImageDraw.ImageDraw, text: str, cx: int, cy: int,
                   font, color: tuple = (30, 30, 30)) -> None:
    """在 (cx, cy) 置中繪製文字。"""
    tw, th = _text_size(draw, text, font)
    draw.text((cx - tw // 2, cy - th // 2), text, fill=color, font=font)


def combine_grid(
    image_paths_2d: list[list[str | None]],
    output_path: str,
    title: str = "",
    tile_w: int = 700,
    tile_h: int = 500,
    row_labels: list[str] | None = None,
    col_labels: list[str] | None = None,
    label_font_size: int = 18,
    title_font_size: int = 24,
) -> None:
    """
    將 2D 圖路徑清單拼接成單張大圖 PNG。

    Parameters
    ----------
    image_paths_2d : list[list[str | None]]
        (rows × cols) 的圖路徑；None 或找不到的路徑會填入灰色佔位圖。
    output_path : str
        輸出 PNG 路徑。
    title : str
        大圖頂部標題。
    tile_w / tile_h : int
        每格子圖的像素大小（所有子圖均 resize 至此）。
    row_labels : list[str] | None
        左側列標籤（每列一個）。
    col_labels : list[str] | None
        頂部行標籤（每行一個）。
    """
    n_rows = len(image_paths_2d)
    n_cols = max((len(r) for r in image_paths_2d), default=0)
    if n_rows == 0 or n_cols == 0:
        print(f"  [WARN] 沒有可拼接的格子，跳過: {output_path}")
        return

    TITLE_H   = 54 if title       else 0
    COL_LBL_H = 38 if col_labels  else 0
    ROW_LBL_W = 90 if row_labels  else 0

    canvas_w = ROW_LBL_W + n_cols * tile_w
    canvas_h = TITLE_H + COL_LBL_H + n_rows * tile_h

    canvas = Image.new('RGB', (canvas_w, canvas_h), color=(255, 255, 255))
    draw   = ImageDraw.Draw(canvas)

    font_title = _load_font(title_font_size)
    font_label = _load_font(label_font_size)

    # 標題
    if title:
        _draw_centered(draw, title, canvas_w // 2, TITLE_H // 2, font_title)

    # 行標籤（頂部）
    if col_labels:
        for c, lbl in enumerate(col_labels[:n_cols]):
            cx = ROW_LBL_W + c * tile_w + tile_w // 2
            cy = TITLE_H + COL_LBL_H // 2
            _draw_centered(draw, lbl, cx, cy, font_label, color=(60, 60, 180))

    # 列標籤（左側）+ 子圖貼入
    for r, row in enumerate(image_paths_2d):
        row_y = TITLE_H + COL_LBL_H + r * tile_h

        if row_labels and r < len(row_labels):
            _draw_centered(draw, row_labels[r],
                           ROW_LBL_W // 2, row_y + tile_h // 2,
                           font_label, color=(180, 60, 60))

        for c in range(n_cols):
            img_path = row[c] if c < len(row) else None
            cell_x   = ROW_LBL_W + c * tile_w
            cell_y   = row_y

            if img_path and os.path.exists(img_path):
                try:
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize((tile_w, tile_h), Image.LANCZOS)
                    canvas.paste(img, (cell_x, cell_y))
                except Exception as exc:
                    print(f"  [WARN] 讀取失敗 {img_path}: {exc}")
                    canvas.paste(_make_placeholder(tile_w, tile_h, "Error"),
                                 (cell_x, cell_y))
            else:
                if img_path:
                    print(f"  [WARN] 找不到圖: {img_path}")
                canvas.paste(_make_placeholder(tile_w, tile_h, "N/A"),
                             (cell_x, cell_y))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    canvas.save(output_path)
    print(f"  └─ 已儲存: {output_path}")


# ─── 各圖拼接函數 ────────────────────────────────────────────────────────────

def combine_01_trends(targets: list[str], splits: list[str]) -> None:
    """
    01_Metrics_Trends — 3×3 大圖 (列=splits, 行=targets)，共 2 張
      · brier_score_trend
      · log_loss_trend
    """
    print("\n[拼圖] 01_Metrics_Trends ...")
    metric_defs = [
        ("brier_score_trend", "Brier Score Trends"),
        ("log_loss_trend",    "Log Loss Trends"),
    ]
    for suffix, metric_label in metric_defs:
        grid: list[list[str | None]] = []
        for split in splits:
            row: list[str | None] = []
            for target in targets:
                path = (f"{PLOTS_DIR}/01_Metrics_Trends/{target}/{split}"
                        f"/{target}_{split}_{suffix}.png")
                row.append(path)
            grid.append(row)

        out = f"{COMBINED_DIR}/01_Metrics_Trends/combined_{suffix}.png"
        combine_grid(
            grid, out,
            title=f"Metrics Trends — {metric_label}",
            tile_w=660, tile_h=530,   # 原圖 1060x852 → 比例 1.24
            row_labels=splits,
            col_labels=targets,
        )


def combine_02_trends_split_y(targets: list[str], splits: list[str]) -> None:
    """
    01_Metrics_Trends_split_y — 3×3 大圖 (列=splits, 行=targets)，共 4 張
      · group0 brier / group0 log_loss
      · group1 brier / group1 log_loss
    """
    print("\n[拼圖] 01_Metrics_Trends_split_y ...")
    metric_defs = [
        ("brier_score_trend", "Brier Score Trends"),
        ("log_loss_trend",    "Log Loss Trends"),
    ]
    for g in ALL_GROUPS:
        for suffix, metric_label in metric_defs:
            grid: list[list[str | None]] = []
            for split in splits:
                row: list[str | None] = []
                for target in targets:
                    path = (f"{PLOTS_DIR}/01_Metrics_Trends_split_y/{target}/{split}"
                            f"/{target}_{split}_group{g}_{suffix}.png")
                    row.append(path)
                grid.append(row)

            out = (f"{COMBINED_DIR}/01_Metrics_Trends_split_y"
                   f"/combined_group{g}_{suffix}.png")
            combine_grid(
                grid, out,
                title=f"Metrics Trends (y1=={g}) — {metric_label}",
                tile_w=660, tile_h=530,   # 原圖 1060x852 → 比例 1.24
                row_labels=splits,
                col_labels=targets,
            )


def combine_03_reliability_combined(targets: list[str], models: list[str]) -> None:
    """
    02_Reliability_Curves_combined — 3×6 大圖 (列=splits, 行=layers)，共 15 張
      · 每個 (target, model) 一張
    """
    print("\n[拼圖] 02_Reliability_Curves_combined ...")
    col_labels = [f"Layer {i}" for i in ALL_LAYERS]
    for target in targets:
        for model in models:
            grid: list[list[str | None]] = []
            for split in ALL_SPLITS:
                row: list[str | None] = []
                for layer in ALL_LAYERS:
                    path = (f"{PLOTS_DIR}/02_Reliability_Curves_combined"
                            f"/{target}/{split}/layer_{layer}"
                            f"/{target}_{split}_layer{layer}_{model}_combined_reliability.png")
                    row.append(path)
                grid.append(row)

            out = (f"{COMBINED_DIR}/02_Reliability_Curves_combined"
                   f"/{target}/{target}_{model}_combined_reliability_3x6.png")
            combine_grid(
                grid, out,
                title=f"Reliability Curves (Combined) — Target: {target} | Model: {model}",
                tile_w=550, tile_h=512,   # 原圖 920x856 → 比例 1.07
                row_labels=ALL_SPLITS,
                col_labels=col_labels,
            )


def combine_04_quadrant_hist(targets: list[str], models: list[str]) -> None:
    """
    03_Quadrant_Histograms — 2×3 大圖 (固定 split=test2)，共 15 張
      · 列 0: layer_1 ~ 3，列 1: layer_4 ~ 6
      · 每個 (target, model) 一張
    """
    print("\n[拼圖] 03_Quadrant_Histograms ...")
    layer_groups = [ALL_LAYERS[:3], ALL_LAYERS[3:]]   # [1,2,3], [4,5,6]
    row_labels   = ["Layer 1~3", "Layer 4~6"]

    for target in targets:
        for model in models:
            grid: list[list[str | None]] = []
            for grp_layers in layer_groups:
                row: list[str | None] = []
                for layer in grp_layers:
                    path = (f"{PLOTS_DIR}/03_Quadrant_Histograms"
                            f"/{target}/test2/layer_{layer}"
                            f"/{target}_test2_layer{layer}_{model}_quadrant_histogram.png")
                    row.append(path)
                grid.append(row)

            out = (f"{COMBINED_DIR}/03_Quadrant_Histograms"
                   f"/{target}/{target}_{model}_quadrant_2x3.png")
            combine_grid(
                grid, out,
                title=f"Quadrant Histograms (test2) — Target: {target} | Model: {model}",
                tile_w=840, tile_h=595,   # 原圖 2086x1478 → 比例 1.41
                row_labels=row_labels,
                col_labels=None,   # 個別子圖標題已含層號資訊
            )


def combine_05_score_hist(targets: list[str], models: list[str]) -> None:
    """
    04_Score_Histograms — 4×3 大圖 (固定 split=test2)，共 15 張
      · 上半 (列 0-1): y1==0，下半 (列 2-3): y1==1
      · 每半各為 2 列 × 3 行 (layer_1~3, layer_4~6)
      · 每個 (target, model) 一張
    """
    print("\n[拼圖] 04_Score_Histograms ...")
    layer_groups = [ALL_LAYERS[:3], ALL_LAYERS[3:]]
    row_labels   = ["y1=0  L1~3", "y1=0  L4~6", "y1=1  L1~3", "y1=1  L4~6"]

    for target in targets:
        for model in models:
            grid: list[list[str | None]] = []
            for g in ALL_GROUPS:
                for grp_layers in layer_groups:
                    row: list[str | None] = []
                    for layer in grp_layers:
                        path = (f"{PLOTS_DIR}/04_Score_Histograms"
                                f"/{target}/test2/layer_{layer}"
                                f"/{target}_test2_layer{layer}_{model}_iso_{g}_score_histogram.png")
                        row.append(path)
                    grid.append(row)

            out = (f"{COMBINED_DIR}/04_Score_Histograms"
                   f"/{target}/{target}_{model}_score_hist_4x3.png")
            combine_grid(
                grid, out,
                title=f"Score Histograms (test2) — Target: {target} | Model: {model}",
                tile_w=840, tile_h=299,   # 原圖 2086x743 → 比例 2.81
                row_labels=row_labels,
                col_labels=None,
            )


def _combine_2x6_chart(
    src_folder: str,
    dst_folder: str,
    file_suffix: str,
    chart_label: str,
    targets: list[str],
    models: list[str],
    tile_w: int,
    tile_h: int,
    split: str = "test2",
) -> None:
    """
    共用函數：05_Brier_Components / 06_Step_Mappings / 07_Joint_Calibration
    格局: 2 列 (y1==0, y1==1) × 6 行 (layer_1~6)
    共 15 張大圖 (3 targets × 5 models)
    """
    col_labels = [f"Layer {i}" for i in ALL_LAYERS]
    row_labels = ["y1=0", "y1=1"]

    for target in targets:
        for model in models:
            grid: list[list[str | None]] = []
            for g in ALL_GROUPS:
                row: list[str | None] = []
                for layer in ALL_LAYERS:
                    path = (f"{PLOTS_DIR}/{src_folder}"
                            f"/{target}/{split}/layer_{layer}"
                            f"/{target}_{split}_layer{layer}_{model}_iso_{g}_{file_suffix}.png")
                    row.append(path)
                grid.append(row)

            # 檔名加入 split 以區分不同評估集
            out = (f"{COMBINED_DIR}/{dst_folder}"
                   f"/{target}/{target}_{model}_{split}_{file_suffix}_2x6.png")
            combine_grid(
                grid, out,
                title=f"{chart_label} ({split}) — Target: {target} | Model: {model}",
                tile_w=tile_w, tile_h=tile_h,
                row_labels=row_labels,
                col_labels=col_labels,
            )


def combine_06_brier_components(targets: list[str], models: list[str]) -> None:
    """05_Brier_Components — 2×6 大圖，共 15 張。"""
    print("\n[拼圖] 05_Brier_Components ...")
    _combine_2x6_chart(
        src_folder="05_Brier_Components",
        dst_folder="05_Brier_Components",
        file_suffix="brier_components",
        chart_label="Brier Components",
        targets=targets, models=models,
        tile_w=530, tile_h=438,   # 原圖 1335x1105 → 比例 1.21
    )


def combine_07_step_mappings(targets: list[str], models: list[str]) -> None:
    """06_Step_Mappings — 2×6 大圖，共 15 張。"""
    print("\n[拼圖] 06_Step_Mappings ...")
    _combine_2x6_chart(
        src_folder="06_Step_Mappings",
        dst_folder="06_Step_Mappings",
        file_suffix="step_mapping",
        chart_label="Step Mappings",
        targets=targets, models=models,
        tile_w=540, tile_h=402,   # 原圖 1038x773 → 比例 1.34
    )


def combine_08_joint_calibration(
    targets: list[str],
    models: list[str],
    splits: list[str] | None = None,
) -> None:
    """07_Joint_Calibration — 2×6 大圖；預設畫 test1 + test2，共 30 張。"""
    if splits is None:
        splits = ["test1", "test2"]
    print("\n[拼圖] 07_Joint_Calibration ...")
    for sp in splits:
        _combine_2x6_chart(
            src_folder="07_Joint_Calibration",
            dst_folder="07_Joint_Calibration",
            file_suffix="joint_calibration",
            chart_label="Joint Calibration",
            targets=targets, models=models,
            tile_w=500, tile_h=500,   # 原圖 1035x1035 → 比例 1.00
            split=sp,
        )


# ─── CLI 主程式 ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM Safety Probe Pipeline - Step 4: Combine Plots into Grid Overviews",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
圖表類型說明:
  trends              01_Metrics_Trends        3×3 (splits × targets) × 2 指標
  trends_split_y      01_Metrics_Trends_split_y 3×3 × 4 張 (group0/1 × 指標)
  reliability_combined 02_Reliability_Curves_combined  3×6 × 15 張
  quadrant_hist       03_Quadrant_Histograms   2×3 (test2) × 15 張
  score_hist          04_Score_Histograms      4×3 (test2) × 15 張
  brier_components    05_Brier_Components      2×6 (test2) × 15 張
  step_mappings       06_Step_Mappings         2×6 (test2) × 15 張
  joint_calibration   07_Joint_Calibration     2×6 (test2) × 15 張

注意: quadrant_hist / score_hist / brier_components /
      step_mappings / joint_calibration 固定使用 test2，--split 對其無效。
        """,
    )
    parser.add_argument(
        "--chart",
        choices=[
            'all', 'trends', 'trends_split_y', 'reliability_combined',
            'quadrant_hist', 'score_hist', 'brier_components',
            'step_mappings', 'joint_calibration',
        ],
        default='all',
        help="要拼接的圖表類型 (預設: all)",
    )
    parser.add_argument(
        "--target",
        choices=['y1', 'y2', 'y3', 'all'],
        default='all',
        help="目標任務 (預設: all)",
    )
    parser.add_argument(
        "--split",
        choices=['test1', 'test2', 'eval', 'all'],
        default='all',
        help="資料集切分 (預設: all；部分圖固定 test2，此參數對其無效)",
    )
    parser.add_argument(
        "--model",
        choices=['SGD', 'MLP', 'LGB', 'LR', 'RF', 'all'],
        default='all',
        help="分類器模型 (預設: all)",
    )
    args = parser.parse_args()

    # 解析過濾條件
    targets = ALL_TARGETS if args.target == 'all' else [args.target]
    splits  = ALL_SPLITS  if args.split  == 'all' else [args.split]
    models  = ALL_MODELS  if args.model  == 'all' else [args.model]
    charts  = (
        ['trends', 'trends_split_y', 'reliability_combined',
         'quadrant_hist', 'score_hist', 'brier_components',
         'step_mappings', 'joint_calibration']
        if args.chart == 'all' else [args.chart]
    )

    # 確認來源目錄存在
    if not os.path.isdir(PLOTS_DIR):
        print(f"錯誤: 找不到 {PLOTS_DIR}/ 目錄。請確保已執行 step3_plot.py")
        sys.exit(1)

    print("開始大圖拼接...")
    print(f"  篩選條件 -> 任務: {targets} | 評估集: {splits} | 模型: {models} | 圖表: {charts}")

    if 'trends' in charts:
        combine_01_trends(targets, splits)

    if 'trends_split_y' in charts:
        combine_02_trends_split_y(targets, splits)

    if 'reliability_combined' in charts:
        combine_03_reliability_combined(targets, models)

    if 'quadrant_hist' in charts:
        combine_04_quadrant_hist(targets, models)

    if 'score_hist' in charts:
        combine_05_score_hist(targets, models)

    if 'brier_components' in charts:
        combine_06_brier_components(targets, models)

    if 'step_mappings' in charts:
        combine_07_step_mappings(targets, models)

    if 'joint_calibration' in charts:
        combine_08_joint_calibration(targets, models)

    print("\n[OK] 所有大圖拼接完成！(已儲存至 results/plots/combined/)")


if __name__ == '__main__':
    main()
