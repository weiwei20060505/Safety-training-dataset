# LLM 安全防護特徵分析、特徵探針與 post-hoc 機率校正架構報告

本報告全面、詳盡地解析了本專案的技術設計、數學原理、程式碼架構與實驗發現。本專案旨在利用大型語言模型（LLM）內部的激活特徵（Hidden States）訓練安全特徵探針（Probes），並結合保序迴歸（Isotonic Regression）進行後驗（post-hoc）機率校正，以增強 LLM 安全分類器在分佈偏移（Covariate Shift）下的預測準確度與機率可靠性。

---

## 📌 目錄
1. [專案背景與核心任務](#1-專案背景與核心任務)
2. [資料來源、格式與特徵工程](#2-資料來源格式與特徵工程)
3. [統一機器學習特徵探針框架 (`unified_train.py`)](#3-統一機器學習特徵探針框架-unified_trainpy)
4. [評估與校正管線架構 (`evaluation_pipeline/`)](#4-評估與校正管線架構-evaluation_pipeline)
5. [預測正確性四象限診斷與圖表結構 (`results/plots/`)](#5-預測正確性四象限診斷與圖表結構-resultsplots)
6. [核心實驗發現與技術結論](#6-核心實驗發現與技術結論)

---

## 1. 專案背景與核心任務

### 1.1 背景與研究價值
隨著 LLM 被廣泛部署，如何防範惡意對抗性提示詞（如 Jailbreak 越獄攻擊）成為核心議題。傳統的安全對齊主要依賴於 RLHF 或安全提示詞過濾，但這些方法位於模型外部或最終輸出端。
本專案採用**特徵探針（Representation Probing）**技術，直接讀取 LLM 隱藏層的神經元激活狀態（Hidden States），在模型尚未輸出文字前判定輸入 Prompt 是否有害、模型的回覆是否安全，以及安全判定是否一致。這對於構建在線安全網關（Safety Guardrails）具有極高的物理推理速度與準確度優勢。

### 1.2 三大預測任務之數學定義
在機器學習流程中，定義了三個分類任務：
1. **$y_1$ 任務 (模型回覆安全性預測，Model Reply Safety)**：
   - **標籤**：$y_1 = 1$ 代表模型回覆為「不安全 (UNSAFE)」；$y_1 = 0$ 代表模型回覆為「安全 (SAFE)」。
2. **$y_2$ 任務 (提示詞有害性預測，Prompt Harmfulness)**：
   - **標籤**：$y_2 = 1$ 代表提示詞為「有害 (Harmful)」；$y_2 = 0$ 代表提示詞為「無害 (Benign)」。
3. **$y_3$ 任務 (安全判定一致性預測，Safety Consistency)**：
   - **標籤**：$y_3 = 1$ 代表「一致 (Consistent)」；$y_3 = 0$ 代表「不一致 (Inconsistent)」。
   - **數學定義**：
     $$y_3 = \mathbb{I}(y_1 == y_2)$$

---

## 2. 資料來源、格式與特徵工程

### 2.1 原始資料集 (WildJailbreak)
專案數據來自 AllenAI 開源的 [WildJailbreak](https://huggingface.co/datasets/allenai/wildjailbreak) 資料集，包含 Vanilla 原始樣本與 Adversarial 對抗性樣本。

### 2.2 特徵工程與維度資訊
LLM 在處理 Prompt 的最後一個 Token 時，提取其在 6 個特定隱藏層的活化狀態（`last_input_hidden_state`）。
- **特徵維度**：$X \in \mathbb{R}^{M \times 6 \times 1024}$。
- **數據集劃分**：
  - **基準訓練集 (`data/experiment_results_train_10000.pkl`)**：包含 10,000 筆資料。
  - **擴充測試集 (`data/test1.pkl` & `data/test2.pkl`)**：各自從 75,000 筆剩餘資源池無重複抽樣並擴充至 **各 10,000 筆**，維持原始先驗分佈比例。
  - **外部評估集 (`experiment_results_eval.pkl`)**：包含 2,210 筆獨立對抗樣本。

---

## 3. 統一機器學習特徵探針框架 (`unified_train.py`)

### 3.1 資料預處理流水線 (Pipeline)
為確保特徵在各模型間的兼容性與泛化力，對每個層的 $2\text{D}$ 特徵矩陣 $X_{\text{layer}} \in \mathbb{R}^{M \times 1024}$ 構建了以下流水線：
1. **資料分割 (Data Splitting)**：按 $60\%(\text{Train}) : 20\%(\text{Val}) : 20\%(\text{Test})$ 分割。
2. **標準化 (StandardScaler)**：將訓練集的特徵縮放為均值為 $0$、方差為 $1$。
3. **不平衡處理 (RandomUnderSampler)**：隨機下採樣多數類樣本使訓練集比率達到 $1:1$。
4. **降維 (PCA)**：將 $1024$ 維特徵降維至 $k=128$ 維。

### 3.2 五大分類器
- **SGD** (隨機梯度下降 Log Loss)
- **MLP** (多層感知機 128 隱藏單元)
- **LGB** (LightGBM 梯度提升樹)
- **LR** (邏輯斯迴歸 L2 正則化)
- **RF** (隨機森林 100 棵樹)

---

## 4. 評估與校正管線架構 (`evaluation_pipeline/`)

管線由三個專門腳本組成：
1. **`step1_prepare_test_data.py`**：劃分出獨立 `test1` (1,000) 與 `test2` (1,000)，並以無重疊資料擴充至 **各 10,000 筆**。
2. **`step2_calibrate.py`**：依據 $y_1 == 0$ 與 $y_1 == 1$ 分流訓練 `iso_0` 與 `iso_1` 保序迴歸模型，產出與快取評估指標。
3. **`step3_plot.py`**：提供 CLI 參數控制，產出 6 大類獨立繪圖產務。

---

## 5. 預測正確性四象限診斷與圖表結構 (`results/plots/`)

圖表結構已經優化，最頂層為 6 大類別：

`results/plots/<01_Metrics_Trends ~ 06_Step_Mappings>/<y1|y2|y3>/<test1|test2|eval>/layer_1~6/`

1. **`01_Metrics_Trends`**：Brier Score 與 Log Loss 獨立隨層數變化趨勢圖。
2. **`02_Reliability_Curves_split_y`**：依 $y_1$ 分流 (iso_0, iso_1) 之可靠度對比曲線。
3. **`03_Quadrant_Histograms`**：2x2 全量四象限預測置信度直方圖。
4. **`04_Score_Histograms`**：依 $y_1$ 分流 (iso_0, iso_1) 之正負樣本預測分數直方圖。
5. **`05_Brier_Components`**：依 $y_1$ 分流 (iso_0, iso_1) 之 Brier 組分雙 Y 軸圖（左軸 Rel/Res 柱狀圖，右軸 Weight 折線圖標明「樣本比例」）。
6. **`06_Step_Mappings`**：依 $y_1$ 分流 (iso_0, iso_1) 之保序校正分數映射階梯圖。

---

## 6. 核心實驗發現與技術結論

1. **隱藏層特徵深度**：隨著層數增加 (Layer 1 -> 6)，模型的分類性能與 AUC 顯著上升，說明深層隱藏狀態包含了更明確的安全語意。
2. **條件雙軌校正效果**：分流 `iso_0` 與 `iso_1` 校正顯著降低了對抗評估集 `eval` 上的 Brier Score 與 Log Loss（如 LGB Layer 6 的 Brier Score 從 `0.26` 降至 `0.15`）。
3. **高動態視覺化診斷**：雙 Y 軸 Brier 組分圖與分流階梯圖清晰地展示了校正後機率分佈的階梯特徵與主導 Bins 的樣本權重。
