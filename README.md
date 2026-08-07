# LLM 安全防護特徵分析、特徵探針與 post-hoc 機率校正系統

本項目是一個完整的機器學習特徵探針（Representation Probing）與後驗（post-hoc）機率校正框架。項目旨在利用大型語言模型（LLM）內部的激活特徵（Hidden States），在模型尚未輸出文字前判定輸入提示詞（Prompt）是否有害、模型回覆是否安全，以及安全判定是否一致，並導入雙軌保序迴歸（Isotonic Regression）機率校正，消除下採樣導致的機率扭曲，還原真實的預測信賴度。

---

## 📌 專案目錄結構

專案採用高度模組化的目錄設計，將舊版基準管線與 8 月 6 日新版框架管線完全解耦：

```text
Safety-training-dataset/
├── data/                          # 基準數據與測試集 (pkl, csv)
├── results/                       # 訓練成果、快取與各版本圖表目錄
│   ├── v1_baseline/               # 🔵 舊版 Baseline 管線輸出結果
│   │   ├── unified_training/      # 探針訓練模型與驗證圖
│   │   ├── safety_guardrails_evaluation/ # 雙軌校正快取與指標
│   │   └── plots/                 # 舊版視覺化圖表與拼接大圖
│   ├── v2_framework/              # 🟢 新版 Framework 管線輸出結果
│   │   ├── framework_training/    # 框架 16 個模型權重與 summary
│   │   ├── framework_calibration/ # 框架校正與快取結果
│   │   ├── conditional_training/  # 條件分流模型權重
│   │   ├── plots_framework/       # 新版 ROC 曲線與比較圖
│   │   └── plots_framework_stage2/# 新版校正診斷圖與聯合校正圖
│   └── logs/                      # 📄 數據檢查與調試日誌
│
├── pipeline_v1_baseline/          # 🔵 舊版 Baseline 管線程式碼
│   ├── unified_train.py           # 舊版基礎探針模型訓練
│   ├── step1_prepare_test_data.py # 舊版測試集擴增
│   ├── step2_calibrate.py         # 舊版雙軌機率校正
│   ├── step3_plot.py              # 舊版繪圖
│   ├── step4_combine_plots.py     # 舊版大圖拼接
│   ├── EXECUTION_SUMMARY.md       # 舊版執行摘要
│   └── utils_calibration.py       # 舊版指標計算工具
│
├── pipeline_v2_framework/         # 🟢 8月6日新版 Framework 管線程式碼
│   ├── train_framework_models.py  # 新版 16 個模型訓練入口
│   ├── calibrate_framework_models.py# 新版子群條件校正 (PAVA)
│   ├── conditional_models.py      # 新版模型架構 (RootSplit, Y-Head)
│   ├── wrapper_models.py          # 新版模型包裝
│   ├── plot_framework_results.py  # 新版 ROC 與條形圖繪製
│   ├── plot_framework_stage2.py   # 新版階段二診斷圖 (V1)
│   ├── plot_framework_stage2_v2.py# 新版階段二聯合校正大圖 (V2)
│   ├── plot_conditional_roc.py    # 新版條件 ROC 繪圖
│   └── utils_calibration.py       # 新版指標計算工具
│
├── tools/                         # 🛠️ 實用輔助工具
│   └── regenerate_all_pdfs.py     # Marp 簡報轉 PDF 工具
├── old_python/                    # 📦 歷史封存歸檔區
├── Async_run_experiment_train.py  # API 推論與特徵萃取 (Train)
├── Async_run_experiment_eval.py   # API 推論與特徵萃取 (Eval)
├── README.md                      # 本說明文件 (整合報告)
└── 🗺️專案階段狀態總結.md          # 專案當前狀態與決策總結
```

---

## 🎯 專案背景與三大分類任務

隨着 LLM 被廣泛部署，如何防範惡意對抗性提示詞（如 Jailbreak 越獄攻擊）成為核心議題。傳統的安全對齊主要依賴於生成後的文字匹配或外部安全篩選 API，這會帶來顯著的推論延遲。

本項目採用特徵探針技術，直接讀取 LLM 隱藏層的神經元激活狀態（Hidden States），在模型尚未輸出文字前進行預測。本框架定義了三個核心分類任務：

1.  **$y_1$ 任務 (模型回覆安全性預測，Model Reply Safety)**：
    *   預測模型回覆是否包含 `unsafe` 標籤（Unsafe = 1, Safe = 0）。
2.  **$y_2$ 任務 (提示詞有害性預測，Prompt Harmfulness)**：
    *   預測輸入提示詞是否有害（Harmful = 1, Benign = 0）。
3.  **$y_3$ 任務 (安全判定一致性預測，Safety Consistency)**：
    *   預測 LLM 的安全判定是否與輸入 Prompt 的真實有害性一致（Consistent = 1, Inconsistent = 0）。
    *   數學定義為指示函數：
        $$y_3 = \mathbb{I}(y_1 == y_2)$$

---

## ⚙️ 資料來源與特徵工程

1.  **原始資料集 (WildJailbreak)**：包含對抗性樣本（Adversarial）與常規樣本（Vanilla）。
2.  **特徵工程**：提取 LLM 在處理 Prompt 的最後一個 Token 時，在 6 個特定隱藏層的活化狀態（`last_input_hidden_state`）。
    *   特徵維度：$X \in \mathbb{R}^{M \times 6 \times 1024}$。
3.  **資料預處理流水線 (Pipeline)**：
    *   **資料分割**：按 $60\%(\text{Train}) : 20\%(\text{Val}) : 20\%(\text{Test})$ 分割。
    *   **標準化 (StandardScaler)**：將特徵縮放為均值為 $0$、方差為 $1$。
    *   **不平衡處理 (RandomUnderSampler)**：下採樣多數類樣本至與少數類相同（1:1）。
    *   **降維 (PCA)**：將 1024 維特徵投影降維至 $k=128$ 維。
4.  **數據集劃分與擴增**：
    *   **基準訓練集 (`data/experiment_results_train_10000.pkl`)**：10,000 筆。
    *   **擴充測試集 (`data/test1.pkl` & `data/test2.pkl`)**：各自從 75,000 筆剩餘資源池無重複抽樣並擴充至 **各 10,000 筆**，維持原始先驗分佈比例。
    *   **外部評估集 (`experiment_results_eval.pkl`)**：2,210 筆獨立對抗樣本。

---

## 🔵 舊版基準管線工作流 (Pipeline V1)

舊版管線的核心在於訓練獨立探針模型，並對輸出分數進行 post-hoc 條件機率校正。

### 核心腳本說明
1.  **`unified_train.py` (探針模型訓練)**：針對 6 層特徵，訓練 5 大經典機器學習模型（SGD, LR, MLP, RF, LGB）以預測 $y_1, y_2, y_3$。
2.  **`step1_prepare_test_data.py` (測試集劃分與擴增)**：切分出獨立 `test1` (1,000) 與 `test2` (1,000)，並從資源池擴充至各 10,000 筆以進行穩健評估。
3.  **`step2_calibrate.py` (雙軌條件機率校正)**：在 `test1` 上依據 $y_1 == 0$ (安全放行) 與 $y_1 == 1$ (不安全攔截) 獨立訓練 `iso_0` 與 `iso_1` 保序迴歸模型。
4.  **`step3_plot.py` (視覺化繪圖)**：產出包含可靠度曲線、階梯映射圖、Brier 組分圖等 6 大類別圖表。
5.  **`step4_combine_plots.py` (大圖拼接)**：將單張圖表依分析類型拼接成多欄網格大圖（Combined Grid）。

---

## 🟢 新版進階框架管線工作流 (Pipeline V2)

8 月 6 日新版框架引入了更進階的模型架構，嘗試利用「決策樹強迫分支」或「多頭神經網路」主動捕捉隱藏狀態與模型安全性 $y_1$ 的交互關係。

### 1. 四大模型策略設計 (`conditional_models.py`)
*   **RootSplit_LGBM (樹狀強迫分支)**：強迫 LightGBM 的根節點優先對 $y_1$ 進行切割，使資料根據安全判定走入完全不同的決策分支。
*   **Feature129_LGBM (129維全域探索)**：將 $y_1$ 作為第 129 維度特徵，與 128 維 Hidden State 一起訓練，讓樹模型自由尋找最佳交互作用點。
*   **YHead_MLP (多層感知機雙頭架構)**：設計一個前段隱藏層共用、後段分叉成兩個獨立 Heads 的神經網路，分別輸出 $y_1=0$ 與 $y_1=1$ 的預測分數。
*   **SingleHead129_MLP (129維單頭標準架構)**：直接將 $y_1$ 作為第 129 維輸入的標準神經網路，由權重矩陣自行捕捉全域交互關係。

### 2. 核心腳本說明
1.  **`train_framework_models.py`**：在 Layer 3, 4, 5, 6 上直接全量訓練上述 4 種策略（共 16 個模型），儲存模型並輸出指標。
2.  **`calibrate_framework_models.py`**：在 Test 1, Test 2, Eval 上對 16 個模型執行 PAVA 保序迴歸與子群 ($y_1$) 分流校正。
3.  **`plot_framework_results.py`**：繪製 1x4 ROC 曲線組合圖與 6 指標模型對比條形圖。
4.  **`plot_framework_stage2.py` / `plot_framework_stage2_v2.py`**：繪製 PAVA 階梯映射、可靠度曲線、雙 Y 軸 Brier 組分圖以及 **Confidence 雙 Y 軸聯合校正圖 (Joint Calibration)**。
5.  **`plot_conditional_roc.py`**：單獨繪製條件模型（RootSplit-LGBM 與 YHead-MLP）的 ROC 分支曲線。

---

## 🧮 機率校正與 Brier 分解數學原理

### A. 保序迴歸 (Isotonic Regression) 與 PAVA 演算法
在不平衡下採樣訓練後，預測機率通常呈 Sigmoid 形扭曲。保序迴歸旨在尋找一個非遞減的保序映射函數 $f(S)$，最小化均方誤差（MSE）：
$$\min_{f} \sum_{i=1}^{M} (y_i - f(S_i))^2 \quad \text{subject to } f(S_a) \le f(S_b) \text{ whenever } S_a \le S_b$$
本系統採用 **PAVA (Pool Adjacent Violators Algorithm)** 求解：
1.  **初始化**：將觀測值視為獨立區塊 $B_i = \{y_i\}$，初始權重 $w_i = 1$，區塊均值 $m_i = y_i$。
2.  **掃描與合併**：由左至右檢查，若 $m_i > m_{i+1}$（違背單調性），則將兩區塊合併為 $B_{\text{new}}$，更新均值為加權平均：
    $$m_{\text{new}} = \frac{|B_i| \cdot m_i + |B_{i+1}| \cdot m_{i+1}}{|B_i| + |B_{i+1}|}$$
3.  **向後檢查 (Backtracking)**：若 $m_{\text{new}} < m_{i-1}$，則繼續向左合併，直至區塊均值完全單調遞增。

### B. Brier Score 分解
Brier Score 用於評估預測概率的準確性：
$$\text{BS} = \frac{1}{M} \sum_{i=1}^{M} (p_i - y_i)^2$$
當預測值被離散化為 $B$ 個 Bins 時，可分解為三個具有明確物理意義的組分：
$$\text{BS} = \text{Reliability} - \text{Resolution} + \text{Uncertainty}$$
*   **Reliability (可靠度/校正誤差)**：越小越好。代表預測機率與實際觀測比例的偏離度。
    $$\text{Rel} = \sum_{b=1}^{B} \frac{N_b}{M} (p_b - \bar{y}_b)^2$$
*   **Resolution (分辨力)**：越大越好。代表模型區分正負樣本的能力。
    $$\text{Res} = \sum_{b=1}^{B} \frac{N_b}{M} (\bar{y}_b - \bar{y})^2$$
*   **Uncertainty (不確定性)**：由資料本身的先驗分佈決定，與模型無關。
    $$\text{Unc} = \bar{y}(1 - \bar{y})$$

---

## 📂 視覺化結果與診斷圖表結構

兩套管線產出的視覺化圖表分別存放於以下路徑：

### 🔵 Pipeline V1 視覺化輸出 (`results/v1_baseline/plots/`)
*   `01_Metrics_Trends/`：Brier Score 與 Log Loss 隨層數變化的獨立趨勢圖。
*   `01_Metrics_Trends_split_y/`：依 $y_1$ 分流 (iso_0, iso_1) 的 Brier / Log Loss 隨層數變化趨勢圖。
*   `02_Reliability_Curves_split_y/`：依 $y_1$ 分流的可靠度對比曲線。
*   `02_Reliability_Curves_combined/`：合併 $y_1 == 0$ 與 $y_1 == 1$ 校正前後之可靠度對比曲線。
*   `03_Quadrant_Histograms/`：2x2 全量四象限預測置信度直方圖。
*   `04_Score_Histograms/`：依 $y_1$ 分流的正負樣本預測分數直方圖。
*   `05_Brier_Components/`：依 $y_1$ 分流的 Brier 組分雙 Y 軸圖（左軸 Rel/Res 柱狀圖，右軸 Weight 樣本比例折線圖）。
*   `06_Step_Mappings/`：依 $y_1$ 分流的保序校正分數映射階梯圖。
*   `combined/`：上述圖表的多子圖拼接大圖總覽。

### 🟢 Pipeline V2 視覺化輸出 (`results/v2_framework/`)
*   `plots_framework/`：
    *   `model_comparison_combined.png`：跨模型在 Acc, Bal Acc, Prec, Rec, F1, AUC 的條形對比圖。
    *   `roc_curves_1x4_val.png`：Layer 3~6 四大模型策略的 ROC 曲線比較圖。
*   `plots_framework_stage2/`：
    *   `07_Joint_Calibration/`：Confidence 雙 Y 軸聯合校正圖（直方圖 + 散佈圖）。
    *   `02_Reliability_Curves_combined/`：四大策略模型校正前後的可靠度對比圖。
    *   `01_Metrics_Trends_split_y/`：Brier Score 與 Log Loss 隨層數變化趨勢圖（大圖 Y 軸強制統一刻度範圍）。

---

## 📈 核心實驗發現與技術結論

1.  **隱藏層特徵深度效應**：隨著隱藏層數的增加 (Layer 1 ➔ 6)，所有模型的分類性能（AUC）均呈現穩步上升趨勢，表明 LLM 深層的隱藏狀態包含了更清晰的安全語義特徵。
2.  **條件雙軌校正成效**：分流 `iso_0` 與 `iso_1` 進行機率校正，能顯著降低模型在外部對抗評估集（`eval`）上的校正誤差。例如，LightGBM (Layer 6) 的 Brier Score 在校正後從 `0.26` 降至 `0.15`。
3.  **條件模型策略對比**：在四大策略中，YHead-MLP 與 RootSplit-LGBM 作為強迫條件限制的模型，在面對對抗性攻擊時展現出更穩健的分類邊界，而 129維全域探索模型則在常規資料集上擁有較佳的擬合上限。
