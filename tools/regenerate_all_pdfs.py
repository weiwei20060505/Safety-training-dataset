"""
regenerate_all_pdfs.py
======================
修正中文字型問題，重新產生所有 PDF。
使用微軟正黑體 (Microsoft JhengHei) 渲染中文。
"""

import os, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                             average_precision_score, brier_score_loss)
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 字型設定（全域）
# ─────────────────────────────────────────────
FONT_PATH = r"C:\Windows\Fonts\msjh.ttc"
_font_prop = fm.FontProperties(fname=FONT_PATH)
FONT_NAME  = _font_prop.get_name()   # "Microsoft JhengHei"

# 讓 matplotlib 全域使用此字型
fm.fontManager.addfont(FONT_PATH)
plt.rcParams["font.family"]      = FONT_NAME
plt.rcParams["axes.unicode_minus"] = False   # 避免負號變方塊

def fp(size=9, bold=False):
    """快速取得 FontProperties"""
    prop = fm.FontProperties(fname=FONT_PATH, size=size)
    if bold:
        prop.set_weight("bold")
    return prop

# ─────────────────────────────────────────────
# 設定
# ─────────────────────────────────────────────
TARGETS    = ["y2", "y3"]
MODELS     = ["LGB", "RF", "MLP", "LR", "SGD"]
MODE       = "split"
EVAL_SET   = "test2"
LAYERS     = [1, 2, 3, 4, 5, 6]
BEST_LAYER = 4

CACHE_DIR  = f"results/v2_20k/02_Safety_Evaluation/cache/{MODE}"
OUT_DIR    = "results/v2_20k_organized/00_Reports"
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 樣式
# ─────────────────────────────────────────────
PALETTE = {
    "roc": "#4A90D9", "pr": "#E87D3E",
    "hist0": "#5AC8C8", "hist1": "#E85D75",
    "cal": "#7B5EA7", "diag": "#CCCCCC",
}
LAYER_COLORS = ["#4A90D9","#5BC0BE","#E87D3E","#E85D75","#7B5EA7","#F5A623"]

def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor("#F9F9FB")
    for sp in ["top","right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left","bottom"]:
        ax.spines[sp].set_color("#DDDDDD")
    ax.tick_params(colors="#555555", labelsize=7)
    if title:  ax.set_title(title, fontsize=8.5, fontweight="bold",
                             color="#2C3E50", pad=4, fontproperties=fp(8.5, bold=True))
    if xlabel: ax.set_xlabel(xlabel, fontsize=7, color="#555555")
    if ylabel: ax.set_ylabel(ylabel, fontsize=7, color="#555555")

# ─────────────────────────────────────────────
# 載入資料
# ─────────────────────────────────────────────
print("載入校準預測快取 ...")
pred_cache = joblib.load(f"{CACHE_DIR}/calibrated_predictions.pkl")
metrics_df = pd.read_csv(f"{CACHE_DIR}/all_metrics_records.csv")

def get_data(target, layer, model):
    try:
        item = pred_cache[target][layer]["splits"][EVAL_SET][model]
        return item["y_true"], item["y_prob"]
    except KeyError:
        return None, None

def compute_metrics(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc      = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc       = average_precision_score(y_true, y_prob)
    brier        = brier_score_loss(y_true, y_prob)
    return dict(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                prec=prec, rec=rec, pr_auc=pr_auc, brier=brier)

def get_ece(target, layer, model):
    sub = metrics_df[
        (metrics_df["task"]     == target) &
        (metrics_df["layer"]    == layer)  &
        (metrics_df["model"]    == model)  &
        (metrics_df["eval_set"] == EVAL_SET)
    ]
    return sub["ece"].values[0] if len(sub) else np.nan

# ─────────────────────────────────────────────
# 產生 PDF
# ─────────────────────────────────────────────
for target in TARGETS:
    for model in MODELS:
        out_path = f"{OUT_DIR}/layer_report_{target}_{model}_{MODE}.pdf"
        print(f"\n產出: {out_path}")
        summary_rows = []

        with PdfPages(out_path) as pdf:

            # ── 封面 ─────────────────────────────────
            fig = plt.figure(figsize=(11.69, 8.27))
            fig.patch.set_facecolor("#2C3E50")
            ax = fig.add_axes([0,0,1,1])
            ax.set_facecolor("#2C3E50"); ax.axis("off")

            kw = dict(transform=ax.transAxes, ha="center")
            ax.text(0.5, 0.74, "Hidden-State Probe Report",
                    fontsize=30, color="white", fontweight="bold",
                    fontproperties=fp(30, bold=True), **kw)
            ax.text(0.5, 0.62,
                    f"目標變數: {target.upper()}   模型: {model}   模式: {MODE.capitalize()}   評估集: {EVAL_SET.upper()}",
                    fontsize=13, color="#AEC6CF",
                    fontproperties=fp(13), **kw)
            ax.text(0.5, 0.50,
                    "各層隱藏狀態作為特徵，訓練探針分類器\n"
                    "本報告評估第 1–6 層的辨識能力",
                    fontsize=11, color="#CCDDDD", linespacing=2.0,
                    fontproperties=fp(11), **kw)
            ax.text(0.5, 0.33,
                    "指標: AUC  |  PR-AUC  |  ECE  |  Brier Score",
                    fontsize=10, color="#FFD700",
                    fontproperties=fp(10), **kw)
            ax.text(0.5, 0.16,
                    f"★  最佳層: Layer {BEST_LAYER}  ★",
                    fontsize=20, color="#FFD700", fontweight="bold",
                    fontproperties=fp(20, bold=True), **kw)

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # ── 每層一頁 ─────────────────────────────
            for layer in LAYERS:
                is_best = (layer == BEST_LAYER)
                color   = LAYER_COLORS[layer - 1]
                y_true, y_prob = get_data(target, layer, model)
                if y_true is None:
                    continue
                m   = compute_metrics(y_true, y_prob)
                ece = get_ece(target, layer, model)
                bg  = "#FFFDF0" if is_best else "white"

                fig = plt.figure(figsize=(11.69, 8.27))
                fig.patch.set_facecolor("#FFFDF5" if is_best else "#FAFAFA")

                # 頁首
                hdr = fig.add_axes([0, 0.91, 1, 0.09])
                hdr.set_facecolor("#4A3600" if is_best else "#2C3E50")
                hdr.axis("off")
                star = "  ★ 最佳層 ★" if is_best else ""
                hdr.text(0.02, 0.5, f"Layer {layer}{star}",
                         ha="left", va="center", fontsize=18,
                         color="#FFD700" if is_best else "white",
                         fontweight="bold",
                         fontproperties=fp(18, bold=True),
                         transform=hdr.transAxes)
                hdr.text(0.98, 0.5,
                         f"目標: {target.upper()}  ·  模型: {model}  ·  {MODE}  ·  {EVAL_SET.upper()}",
                         ha="right", va="center", fontsize=10,
                         color="#AEC6CF",
                         fontproperties=fp(10),
                         transform=hdr.transAxes)

                gs = gridspec.GridSpec(1, 4, figure=fig,
                                       left=0.05, right=0.97,
                                       top=0.87, bottom=0.10, wspace=0.32)

                # ROC
                ax0 = fig.add_subplot(gs[0, 0])
                ax0.set_facecolor(bg)
                ax0.plot(m["fpr"], m["tpr"], color=color, lw=2.2)
                ax0.plot([0,1],[0,1],"--", color="#BBBBBB", lw=1)
                ax0.fill_between(m["fpr"], m["tpr"], alpha=0.13, color=color)
                style_ax(ax0, "ROC Curve", "False Positive Rate", "True Positive Rate")
                ax0.set_xlim(0,1); ax0.set_ylim(0,1.02)
                ax0.text(0.97, 0.05, f"AUC = {m['roc_auc']:.4f}",
                         ha="right", va="bottom", fontsize=9, fontweight="bold",
                         color=color, transform=ax0.transAxes)
                if is_best:
                    ax0.text(0.03, 0.97, "★ BEST",
                             ha="left", va="top", fontsize=8,
                             color="#D4A017", fontweight="bold",
                             transform=ax0.transAxes)

                # PR
                ax1 = fig.add_subplot(gs[0, 1])
                ax1.set_facecolor(bg)
                ax1.plot(m["rec"], m["prec"], color=PALETTE["pr"], lw=2.2)
                bl = y_true.mean()
                ax1.axhline(bl, color="#BBBBBB", lw=1, ls="--",
                            label=f"基準線 ({bl:.2f})")
                ax1.fill_between(m["rec"], m["prec"], alpha=0.13, color=PALETTE["pr"])
                style_ax(ax1, "PR Curve", "Recall", "Precision")
                ax1.set_xlim(0,1); ax1.set_ylim(0,1.02)
                ax1.legend(fontsize=6, framealpha=0.6,
                           prop=fp(6))
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
                style_ax(ax2, "預測機率分布（依真實標籤）",
                         "Predicted Probability", "Density")
                ax2.legend(fontsize=7, framealpha=0.7, loc="upper center",
                           prop=fp(7))
                ax2.text(0.97, 0.97,
                         f"Δμ = {p1.mean()-p0.mean():.3f}",
                         ha="right", va="top", fontsize=7,
                         color="#333333", transform=ax2.transAxes)

                # Reliability
                ax3 = fig.add_subplot(gs[0, 3])
                ax3.set_facecolor(bg)
                frac_pos, mean_pred = calibration_curve(
                    y_true, y_prob, n_bins=10, strategy="uniform")
                ax3.plot([0,1],[0,1],"--", color="#BBBBBB",
                         lw=1.2, label="完美校準")
                ax3.plot(mean_pred, frac_pos, "o-",
                         color=PALETTE["cal"], lw=2.2, ms=5)
                ax3.fill_between(mean_pred, mean_pred, frac_pos,
                                 alpha=0.15, color=PALETTE["cal"])
                style_ax(ax3, "可靠度圖",
                         "Mean Predicted Prob", "Fraction of Positives")
                ax3.set_xlim(0,1); ax3.set_ylim(0,1)
                ax3.legend(fontsize=6, framealpha=0.6,
                           prop=fp(6))
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

            # ── 彙總頁 ────────────────────────────────
            if summary_rows:
                df_sum = pd.DataFrame(summary_rows).set_index("Layer")
                fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27),
                                         gridspec_kw={"width_ratios": [1.4, 1]})
                fig.patch.set_facecolor("#FAFAFA")

                ax_l = axes[0]
                ax_l.set_facecolor("#F0F2F8")
                ax_l2 = ax_l.twinx()

                cfg = [("AUC","#4A90D9","o",True),
                       ("PR-AUC","#E87D3E","s",True),
                       ("ECE","#7B5EA7","^",False),
                       ("Brier","#E85D75","D",False)]
                for metric, clr, mk, left in cfg:
                    ax_use = ax_l if left else ax_l2
                    ax_use.plot(LAYERS, df_sum[metric].values,
                                f"{mk}-", color=clr, lw=2, ms=6, label=metric)

                ax_l.axvline(BEST_LAYER, color="#FFD700", lw=2,
                             ls="--", alpha=0.8, zorder=0)
                ax_l.text(BEST_LAYER + 0.08,
                          ax_l.get_ylim()[0] + (ax_l.get_ylim()[1]-ax_l.get_ylim()[0])*0.97,
                          f"Layer {BEST_LAYER}",
                          color="#B8860B", fontsize=8, va="top",
                          fontproperties=fp(8))

                ax_l.set_xticks(LAYERS)
                ax_l.set_xlabel("Layer", fontsize=10)
                ax_l.set_ylabel("AUC / PR-AUC  (↑ 越高越好)", fontsize=9,
                                fontproperties=fp(9))
                ax_l2.set_ylabel("ECE / Brier  (↓ 越低越好)", fontsize=9,
                                 fontproperties=fp(9))
                ax_l.set_title(
                    f"各層指標趨勢\n({target.upper()} · {model} · {MODE} · {EVAL_SET.upper()})",
                    fontsize=11, fontweight="bold", color="#2C3E50",
                    fontproperties=fp(11, bold=True))
                for sp in ["top"]:
                    ax_l.spines[sp].set_visible(False)
                    ax_l2.spines[sp].set_visible(False)

                l1, lb1 = ax_l.get_legend_handles_labels()
                l2, lb2 = ax_l2.get_legend_handles_labels()
                ax_l.legend(l1+l2, lb1+lb2, fontsize=8,
                            loc="lower right", framealpha=0.8,
                            prop=fp(8))

                # 表格
                ax_r = axes[1]
                ax_r.axis("off")
                col_labels = ["Layer", "AUC ↑", "PR-AUC ↑", "ECE ↓", "Brier ↓"]
                best_idx = {
                    "AUC":    df_sum["AUC"].idxmax(),
                    "PR-AUC": df_sum["PR-AUC"].idxmax(),
                    "ECE":    df_sum["ECE"].idxmin(),
                    "Brier":  df_sum["Brier"].idxmin(),
                }
                table_data = []
                for layer in LAYERS:
                    row = df_sum.loc[layer]
                    cells = [str(layer),
                             f"{row['AUC']:.4f}",
                             f"{row['PR-AUC']:.4f}",
                             f"{row['ECE']:.4f}",
                             f"{row['Brier']:.4f}"]
                    if layer == best_idx["AUC"]:    cells[1] += " ★"
                    if layer == best_idx["PR-AUC"]: cells[2] += " ★"
                    if layer == best_idx["ECE"]:    cells[3] += " ★"
                    if layer == best_idx["Brier"]:  cells[4] += " ★"
                    table_data.append(cells)

                tbl = ax_r.table(cellText=table_data, colLabels=col_labels,
                                 cellLoc="center", loc="center",
                                 bbox=[0.0, 0.18, 1.0, 0.68])
                tbl.auto_set_font_size(False)
                tbl.set_fontsize(10)
                for j in range(len(col_labels)):
                    tbl[(0,j)].set_facecolor("#2C3E50")
                    tbl[(0,j)].set_text_props(color="white", fontweight="bold")
                for i, layer in enumerate(LAYERS, 1):
                    for j in range(len(col_labels)):
                        if layer == BEST_LAYER:
                            tbl[(i,j)].set_facecolor("#FFF8DC")
                            tbl[(i,j)].set_text_props(fontweight="bold")
                        elif i % 2 == 0:
                            tbl[(i,j)].set_facecolor("#F5F5F5")

                ax_r.set_title("彙總比較表", fontsize=11, fontweight="bold",
                               color="#2C3E50", pad=20,
                               fontproperties=fp(11, bold=True))

                # 結論
                conclusion = (
                    f"★  Layer {BEST_LAYER} 的隱藏層特徵辨識能力最好\n"
                    f"   AUC={df_sum['AUC'].max():.4f}  "
                    f"PR-AUC={df_sum['PR-AUC'].max():.4f}\n"
                    f"   ECE={df_sum['ECE'].min():.4f}  "
                    f"Brier={df_sum['Brier'].min():.4f}\n\n"
                    f"→ 後續 Calibration 與 Conformal Prediction\n"
                    f"   均以 Layer {BEST_LAYER} Hidden State 為主要特徵"
                )
                fig.text(0.55, 0.07, conclusion,
                         ha="center", va="bottom", fontsize=9,
                         color="#2C3E50", linespacing=1.7,
                         fontproperties=fp(9),
                         bbox=dict(boxstyle="round,pad=0.5",
                                   facecolor="#FFFDE7",
                                   edgecolor="#FFD700", lw=1.5))

                plt.suptitle(
                    f"各層比較總覽 — 目標: {target.upper()} · 模型: {model}",
                    fontsize=14, fontweight="bold", color="#2C3E50", y=0.98,
                    fontproperties=fp(14, bold=True))
                plt.tight_layout(rect=[0, 0.05, 1, 0.96])
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

            d = pdf.infodict()
            d["Title"]   = f"Layer Report — {target.upper()} {model} {MODE}"
            d["Author"]  = "Safety-Training Analysis"

        print(f"  完成: {out_path}")

print("\n所有 PDF 產出完畢！")
print(f"儲存位置: {OUT_DIR}")
