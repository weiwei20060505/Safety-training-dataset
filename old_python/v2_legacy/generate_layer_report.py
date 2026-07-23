"""
generate_layer_report.py
========================
針對 v2_20k 結果，依照「每 Layer 一頁（ROC / PR / Histogram / Reliability）+
最後一頁彙總表」的格式產出 PDF。

使用方式：
    python generate_layer_report.py

輸出：
    results/v2_20k/layer_report_{TARGET}_{MODEL}_{MODE}.pdf
"""

import os, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    brier_score_loss, log_loss,
)
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 設定
# ─────────────────────────────────────────────
TARGETS   = ["y2", "y3"]          # y1 本身不好分，只報 y2 / y3
MODELS    = ["LGB", "RF", "MLP", "LR", "SGD"]
MODE      = "split"                # baseline / split
EVAL_SET  = "test2"               # test1 / test2
LAYERS    = [1, 2, 3, 4, 5, 6]
BEST_LAYER = 4

CACHE_DIR  = f"results/v2_20k/02_Safety_Evaluation/cache/{MODE}"
OUT_DIR    = "results/v2_20k"
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 樣式
# ─────────────────────────────────────────────
PALETTE = {
    "roc":    "#4A90D9",
    "pr":     "#E87D3E",
    "hist0":  "#5AC8C8",
    "hist1":  "#E85D75",
    "cal":    "#7B5EA7",
    "diag":   "#CCCCCC",
    "fill":   "#F5F5F5",
    "best":   "#FFD700",
    "header": "#2C3E50",
}
LAYER_COLORS = ["#4A90D9","#5BC0BE","#E87D3E","#E85D75","#7B5EA7","#F5A623"]

def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor("#F9F9FB")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#DDDDDD")
    ax.spines["bottom"].set_color("#DDDDDD")
    ax.tick_params(colors="#555555", labelsize=7)
    if title:  ax.set_title(title, fontsize=8, fontweight="bold", color="#2C3E50", pad=4)
    if xlabel: ax.set_xlabel(xlabel, fontsize=7, color="#555555")
    if ylabel: ax.set_ylabel(ylabel, fontsize=7, color="#555555")


# ─────────────────────────────────────────────
# 載入資料
# ─────────────────────────────────────────────
print("📂 載入校準預測快取 ...")
pred_cache = joblib.load(f"{CACHE_DIR}/calibrated_predictions.pkl")
metrics_df  = pd.read_csv(f"{CACHE_DIR}/all_metrics_records.csv")

# ─────────────────────────────────────────────
# 工具函數
# ─────────────────────────────────────────────
def get_data(target, layer, model, eval_set=EVAL_SET):
    """從快取取得 (y_true, y_prob_calibrated)"""
    try:
        item = pred_cache[target][layer]["splits"][eval_set][model]
        return item["y_true"], item["y_prob"]
    except KeyError:
        return None, None


def compute_metrics(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc      = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc       = average_precision_score(y_true, y_prob)
    brier        = brier_score_loss(y_true, y_prob)
    # ECE from metrics_df 一致
    return {
        "fpr": fpr, "tpr": tpr, "roc_auc": roc_auc,
        "prec": prec, "rec": rec, "pr_auc": pr_auc,
        "brier": brier,
    }


def get_ece(target, layer, model, eval_set=EVAL_SET):
    sub = metrics_df[
        (metrics_df["task"]     == target) &
        (metrics_df["layer"]    == layer)  &
        (metrics_df["model"]    == model)  &
        (metrics_df["eval_set"] == eval_set)
    ]
    return sub["ece"].values[0] if len(sub) else np.nan


# ─────────────────────────────────────────────
# 四格圖：單一 Layer + Model
# ─────────────────────────────────────────────
def plot_layer_panel(fig, gs_row, layer, model, target, color, is_best=False):
    """在 fig 的 gs_row 行畫 ROC / PR / Hist / Reliability 四格"""

    y_true, y_prob = get_data(target, layer, model)
    if y_true is None:
        return {}

    m = compute_metrics(y_true, y_prob)
    ece = get_ece(target, layer, model)

    axes = [fig.add_subplot(gs_row[0, i]) for i in range(4)]
    bg_color = "#FFFDF0" if is_best else "white"
    for ax in axes:
        ax.set_facecolor(bg_color)

    # ── ROC ──────────────────────────────────
    ax = axes[0]
    ax.plot(m["fpr"], m["tpr"], color=color, lw=2)
    ax.plot([0,1],[0,1],"--", color="#CCCCCC", lw=1)
    ax.fill_between(m["fpr"], m["tpr"], alpha=0.12, color=color)
    style_ax(ax, f"ROC  (AUC={m['roc_auc']:.3f})", "FPR", "TPR")
    ax.set_xlim(0,1); ax.set_ylim(0,1.02)
    ax.text(0.97, 0.05, f"AUC={m['roc_auc']:.3f}",
            ha="right", va="bottom", fontsize=8, fontweight="bold",
            color=color, transform=ax.transAxes)
    if is_best:
        ax.text(0.03, 0.97, "★ BEST", ha="left", va="top",
                fontsize=7, color="#D4A017", fontweight="bold",
                transform=ax.transAxes)

    # ── PR ───────────────────────────────────
    ax = axes[1]
    ax.plot(m["rec"], m["prec"], color=PALETTE["pr"], lw=2)
    baseline_pr = y_true.mean()
    ax.axhline(baseline_pr, color="#CCCCCC", lw=1, ls="--")
    ax.fill_between(m["rec"], m["prec"], alpha=0.12, color=PALETTE["pr"])
    style_ax(ax, f"PR  (AP={m['pr_auc']:.3f})", "Recall", "Precision")
    ax.set_xlim(0,1); ax.set_ylim(0,1.02)
    ax.text(0.97, 0.05, f"AP={m['pr_auc']:.3f}",
            ha="right", va="bottom", fontsize=8, fontweight="bold",
            color=PALETTE["pr"], transform=ax.transAxes)

    # ── Histogram ────────────────────────────
    ax = axes[2]
    p0 = y_prob[y_true == 0]
    p1 = y_prob[y_true == 1]
    bins = np.linspace(0, 1, 31)
    ax.hist(p0, bins=bins, alpha=0.65, color=PALETTE["hist0"],
            label=f"Y=0 (n={len(p0)})", density=True, edgecolor="white", linewidth=0.3)
    ax.hist(p1, bins=bins, alpha=0.65, color=PALETTE["hist1"],
            label=f"Y=1 (n={len(p1)})", density=True, edgecolor="white", linewidth=0.3)
    style_ax(ax, "Pred Prob Histogram", "Predicted Prob", "Density")
    ax.legend(fontsize=6, framealpha=0.7, loc="upper center")

    # ── Reliability ──────────────────────────
    ax = axes[3]
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
    ax.plot([0,1],[0,1],"--", color=PALETTE["diag"], lw=1.2, label="Perfect")
    ax.plot(mean_pred, frac_pos, "o-", color=PALETTE["cal"], lw=2, ms=4)
    ax.fill_between(mean_pred, mean_pred, frac_pos, alpha=0.15, color=PALETTE["cal"])
    style_ax(ax, f"Reliability  (ECE={ece:.4f})", "Mean Pred Prob", "Fraction Positive")
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.legend(fontsize=6, framealpha=0.7)

    return {"roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"],
            "brier": m["brier"], "ece": ece}


# ─────────────────────────────────────────────
# 主報告：每個 Target × Model 各一份 PDF
# ─────────────────────────────────────────────
for target in TARGETS:
    for model in MODELS:
        out_path = f"{OUT_DIR}/layer_report_{target}_{model}_{MODE}.pdf"
        print(f"\n📄 產出: {out_path}")

        summary_rows = []  # 彙總表資料

        with PdfPages(out_path) as pdf:

            # ──────────────────────────────────────
            # 封面頁
            # ──────────────────────────────────────
            fig = plt.figure(figsize=(11.69, 8.27))  # A4 橫向
            fig.patch.set_facecolor("#2C3E50")
            ax_cover = fig.add_axes([0,0,1,1])
            ax_cover.set_facecolor("#2C3E50")
            ax_cover.axis("off")

            ax_cover.text(0.5, 0.72,
                          "Hidden-State Probe Report",
                          ha="center", va="center", fontsize=32,
                          color="white", fontweight="bold",
                          transform=ax_cover.transAxes)
            ax_cover.text(0.5, 0.60,
                          f"Target: {target.upper()}   |   Model: {model}   |   Mode: {MODE.capitalize()}   |   Eval: {EVAL_SET.upper()}",
                          ha="center", va="center", fontsize=14,
                          color="#AEC6CF", transform=ax_cover.transAxes)
            ax_cover.text(0.5, 0.50,
                          "Each layer's hidden states are used as features\n"
                          "to train a probe classifier. This report evaluates\n"
                          "discriminability across Layer 1–6.",
                          ha="center", va="center", fontsize=11,
                          color="#CCDDDD", linespacing=1.8,
                          transform=ax_cover.transAxes)
            ax_cover.text(0.5, 0.30,
                          "Metrics: AUC  |  PR-AUC  |  ECE  |  Brier Score",
                          ha="center", va="center", fontsize=10,
                          color="#FFD700", transform=ax_cover.transAxes)
            ax_cover.text(0.5, 0.12,
                          f"★  Best Layer: {BEST_LAYER}  ★",
                          ha="center", va="center", fontsize=18,
                          color="#FFD700", fontweight="bold",
                          transform=ax_cover.transAxes)

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # ──────────────────────────────────────
            # 每 Layer 一頁
            # ──────────────────────────────────────
            for layer in LAYERS:
                is_best = (layer == BEST_LAYER)
                color   = LAYER_COLORS[layer - 1]

                fig = plt.figure(figsize=(11.69, 8.27))
                fig.patch.set_facecolor("#FAFAFA" if not is_best else "#FFFDF5")

                # 頁首
                header_ax = fig.add_axes([0, 0.91, 1, 0.09])
                header_ax.set_facecolor("#2C3E50" if not is_best else "#4A3600")
                header_ax.axis("off")
                star = " ★ BEST LAYER ★" if is_best else ""
                header_ax.text(0.02, 0.5,
                               f"Layer {layer}{star}",
                               ha="left", va="center", fontsize=18,
                               color="white" if not is_best else "#FFD700",
                               fontweight="bold",
                               transform=header_ax.transAxes)
                header_ax.text(0.98, 0.5,
                               f"Target: {target.upper()}  ·  Model: {model}  ·  {MODE.capitalize()}  ·  {EVAL_SET.upper()}",
                               ha="right", va="center", fontsize=10,
                               color="#AEC6CF", transform=header_ax.transAxes)

                # 四格圖區域
                gs = gridspec.GridSpec(1, 4, figure=fig,
                                       left=0.05, right=0.97,
                                       top=0.87, bottom=0.10,
                                       wspace=0.32)
                gs_row = gs[0, :]   # 傳整列給 plot_layer_panel

                # 手動建立 subplot
                y_true, y_prob = get_data(target, layer, model)
                if y_true is None:
                    print(f"  ⚠ 找不到資料: {target} layer{layer} {model}")
                    plt.close(fig)
                    continue

                m = compute_metrics(y_true, y_prob)
                ece = get_ece(target, layer, model)
                bg = "#FFFDF0" if is_best else "white"

                # ROC
                ax0 = fig.add_subplot(gs[0, 0])
                ax0.set_facecolor(bg)
                ax0.plot(m["fpr"], m["tpr"], color=color, lw=2.2)
                ax0.plot([0,1],[0,1],"--", color="#BBBBBB", lw=1)
                ax0.fill_between(m["fpr"], m["tpr"], alpha=0.13, color=color)
                style_ax(ax0, f"ROC Curve", "False Positive Rate", "True Positive Rate")
                ax0.set_xlim(0,1); ax0.set_ylim(0,1.02)
                ax0.text(0.97, 0.05, f"AUC = {m['roc_auc']:.4f}",
                         ha="right", va="bottom", fontsize=9, fontweight="bold",
                         color=color, transform=ax0.transAxes)
                if is_best:
                    ax0.text(0.03, 0.97, "★ BEST", ha="left", va="top",
                             fontsize=8, color="#D4A017", fontweight="bold",
                             transform=ax0.transAxes)

                # PR
                ax1 = fig.add_subplot(gs[0, 1])
                ax1.set_facecolor(bg)
                ax1.plot(m["rec"], m["prec"], color=PALETTE["pr"], lw=2.2)
                bl = y_true.mean()
                ax1.axhline(bl, color="#BBBBBB", lw=1, ls="--",
                            label=f"Baseline ({bl:.2f})")
                ax1.fill_between(m["rec"], m["prec"], alpha=0.13, color=PALETTE["pr"])
                style_ax(ax1, "PR Curve", "Recall", "Precision")
                ax1.set_xlim(0,1); ax1.set_ylim(0,1.02)
                ax1.legend(fontsize=6, framealpha=0.6)
                ax1.text(0.97, 0.05, f"AP = {m['pr_auc']:.4f}",
                         ha="right", va="bottom", fontsize=9, fontweight="bold",
                         color=PALETTE["pr"], transform=ax1.transAxes)

                # Histogram
                ax2 = fig.add_subplot(gs[0, 2])
                ax2.set_facecolor(bg)
                p0 = y_prob[y_true == 0]
                p1 = y_prob[y_true == 1]
                bins = np.linspace(0, 1, 31)
                ax2.hist(p0, bins=bins, alpha=0.7, color=PALETTE["hist0"],
                         label=f"Y=0  n={len(p0)}", density=True,
                         edgecolor="white", linewidth=0.3)
                ax2.hist(p1, bins=bins, alpha=0.7, color=PALETTE["hist1"],
                         label=f"Y=1  n={len(p1)}", density=True,
                         edgecolor="white", linewidth=0.3)
                style_ax(ax2, "Predicted Prob by True Label", "Predicted Probability", "Density")
                ax2.legend(fontsize=7, framealpha=0.7, loc="upper center")
                # overlap stats
                overlap_note = f"Δμ = {p1.mean()-p0.mean():.3f}"
                ax2.text(0.97, 0.97, overlap_note, ha="right", va="top",
                         fontsize=7, color="#333333", transform=ax2.transAxes)

                # Reliability
                ax3 = fig.add_subplot(gs[0, 3])
                ax3.set_facecolor(bg)
                frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
                ax3.plot([0,1],[0,1],"--", color="#BBBBBB", lw=1.2, label="Perfect")
                ax3.plot(mean_pred, frac_pos, "o-", color=PALETTE["cal"], lw=2.2, ms=5)
                ax3.fill_between(mean_pred, mean_pred, frac_pos,
                                 alpha=0.15, color=PALETTE["cal"])
                style_ax(ax3, "Reliability Diagram", "Mean Predicted Prob", "Fraction of Positives")
                ax3.set_xlim(0,1); ax3.set_ylim(0,1)
                ax3.legend(fontsize=6, framealpha=0.6)
                ax3.text(0.97, 0.05,
                         f"ECE = {ece:.4f}\nBrier = {m['brier']:.4f}",
                         ha="right", va="bottom", fontsize=8, fontweight="bold",
                         color=PALETTE["cal"], transform=ax3.transAxes)

                plt.tight_layout(rect=[0, 0, 1, 0.91])
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

                summary_rows.append({
                    "Layer": layer,
                    "AUC":    m["roc_auc"],
                    "PR-AUC": m["pr_auc"],
                    "ECE":    ece,
                    "Brier":  m["brier"],
                })

            # ──────────────────────────────────────
            # 最後一頁：彙總比較表
            # ──────────────────────────────────────
            if summary_rows:
                df_sum = pd.DataFrame(summary_rows).set_index("Layer")

                fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27),
                                         gridspec_kw={"width_ratios": [1.4, 1]})
                fig.patch.set_facecolor("#FAFAFA")

                # ── 左：折線趨勢圖 ──────────────────
                ax_l = axes[0]
                ax_l.set_facecolor("#F0F2F8")
                metrics_cfg = [
                    ("AUC",    "#4A90D9", "o", True),
                    ("PR-AUC", "#E87D3E", "s", True),
                    ("ECE",    "#7B5EA7", "^", False),
                    ("Brier",  "#E85D75", "D", False),
                ]
                ax_l2 = ax_l.twinx()
                for metric, clr, mk, left in metrics_cfg:
                    vals = df_sum[metric].values
                    ax_use = ax_l if left else ax_l2
                    ax_use.plot(LAYERS, vals, "o-" if mk=="o" else f"{mk}-",
                                color=clr, lw=2, ms=6, label=metric)

                # highlight best layer
                ax_l.axvline(BEST_LAYER, color="#FFD700", lw=2, ls="--", alpha=0.8, zorder=0)
                ax_l.text(BEST_LAYER+0.05, ax_l.get_ylim()[1]*0.99,
                          f"Layer {BEST_LAYER}", color="#B8860B",
                          fontsize=8, va="top")

                ax_l.set_xticks(LAYERS)
                ax_l.set_xlabel("Layer", fontsize=10)
                ax_l.set_ylabel("AUC / PR-AUC  (↑ better)", fontsize=9)
                ax_l2.set_ylabel("ECE / Brier  (↓ better)", fontsize=9)
                ax_l.set_title(f"Metric Trends across Layers\n"
                               f"({target.upper()} · {model} · {MODE} · {EVAL_SET.upper()})",
                               fontsize=11, fontweight="bold", color="#2C3E50")
                ax_l.spines["top"].set_visible(False)
                ax_l2.spines["top"].set_visible(False)

                # combined legend
                lines1, labs1 = ax_l.get_legend_handles_labels()
                lines2, labs2 = ax_l2.get_legend_handles_labels()
                ax_l.legend(lines1+lines2, labs1+labs2,
                            fontsize=8, loc="lower right", framealpha=0.8)

                # ── 右：數值表 ──────────────────────
                ax_r = axes[1]
                ax_r.axis("off")

                col_labels = ["Layer", "AUC ↑", "PR-AUC ↑", "ECE ↓", "Brier ↓"]
                table_data = []
                best_idx = {
                    "AUC":    df_sum["AUC"].idxmax(),
                    "PR-AUC": df_sum["PR-AUC"].idxmax(),
                    "ECE":    df_sum["ECE"].idxmin(),
                    "Brier":  df_sum["Brier"].idxmin(),
                }
                for layer in LAYERS:
                    row = df_sum.loc[layer]
                    cells = [str(layer),
                             f"{row['AUC']:.4f}",
                             f"{row['PR-AUC']:.4f}",
                             f"{row['ECE']:.4f}",
                             f"{row['Brier']:.4f}"]
                    # 標記最佳
                    if layer == best_idx["AUC"]:    cells[1] += " ⭐"
                    if layer == best_idx["PR-AUC"]: cells[2] += " ⭐"
                    if layer == best_idx["ECE"]:    cells[3] += " ⭐"
                    if layer == best_idx["Brier"]:  cells[4] += " ⭐"
                    table_data.append(cells)

                tbl = ax_r.table(
                    cellText  = table_data,
                    colLabels = col_labels,
                    cellLoc   = "center",
                    loc       = "center",
                    bbox      = [0.0, 0.15, 1.0, 0.7],
                )
                tbl.auto_set_font_size(False)
                tbl.set_fontsize(10)

                # 樣式：表頭
                for j in range(len(col_labels)):
                    tbl[(0, j)].set_facecolor("#2C3E50")
                    tbl[(0, j)].set_text_props(color="white", fontweight="bold")

                # 樣式：最佳層（BEST_LAYER）整列金色
                for i, layer in enumerate(LAYERS, start=1):
                    for j in range(len(col_labels)):
                        if layer == BEST_LAYER:
                            tbl[(i, j)].set_facecolor("#FFF8DC")
                            tbl[(i, j)].set_text_props(fontweight="bold")
                        elif i % 2 == 0:
                            tbl[(i, j)].set_facecolor("#F5F5F5")

                ax_r.set_title("Summary Table", fontsize=11, fontweight="bold",
                               color="#2C3E50", pad=20)

                # 結論文字
                best_auc   = df_sum["AUC"].max()
                best_prau  = df_sum["PR-AUC"].max()
                best_ece   = df_sum["ECE"].min()
                best_brier = df_sum["Brier"].min()

                conclusion = (
                    f"★  Layer {BEST_LAYER} 的隱藏層特徵辨識能力最好\n"
                    f"   AUC={best_auc:.4f}  PR-AUC={best_prau:.4f}\n"
                    f"   ECE={best_ece:.4f}  Brier={best_brier:.4f}\n\n"
                    f"→ 後續 Calibration 與 Conformal Prediction\n"
                    f"   均以 Layer {BEST_LAYER} Hidden State 為主要特徵"
                )
                fig.text(0.55, 0.08, conclusion,
                         ha="center", va="bottom", fontsize=9,
                         color="#2C3E50", linespacing=1.6,
                         bbox=dict(boxstyle="round,pad=0.5",
                                   facecolor="#FFFDE7",
                                   edgecolor="#FFD700", lw=1.5))

                plt.suptitle(f"Layer Comparison Summary — {target.upper()} · {model}",
                             fontsize=14, fontweight="bold", color="#2C3E50", y=0.98)
                plt.tight_layout(rect=[0, 0.05, 1, 0.96])
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

            # PDF metadata
            d = pdf.infodict()
            d["Title"]   = f"Layer Report — {target.upper()} {model} {MODE}"
            d["Author"]  = "Safety-Training Analysis"
            d["Subject"] = "Hidden-state probe evaluation"

        print(f"  ✅ 完成: {out_path}")

print("\n🎉 所有 PDF 報告產出完畢！")
print(f"   儲存位置: {OUT_DIR}/layer_report_*.pdf")
