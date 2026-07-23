# 安全訓練數據集特徵分析與機率校正框架 (Safety Training Dataset Probe & Calibration Framework)

一個完整的機器學習特徵探針與 post-hoc 機率校正框架，用於訓練和評估多個分類模型，利用大語言模型（LLM）的內部隱藏狀態（Hidden States）檢測提示詞的有害性、模型回應的安全一致性，並進行雙軌條件機率校正。

---

## 📋 目錄

- [項目簡介](#項目簡介)
- [核心特性](#核心特性)
- [快速開始](#快速開始)
- [專案結構](#專案結構)
- [工作流腳本說明](#工作流腳本說明)
- [輸出結果結構](#輸出結果結構)

---

## 🎯 項目簡介

本項目旨在評估和增強 LLM 在對抗性攻擊與常規場景下的安全表現。我們基於 **WildJailbreak 資料集**（包含 Vanilla 原始樣本與 Adversarial 對抗性樣本）進行雙軌特徵預測實驗。我們從 LLM 的 6 個特徵層中提取了輸入序列最後一個 Token 的隱藏狀態（`last_input_hidden_state`）作為模型特徵 $X$。

本專案構建了一個統一的機器學習訓練、評估與校正管線，針對隱藏狀態特徵進行以下三個分類任務的特徵探針（Probing）訓練：

1. **模型回覆安全性預測（Y1）**：預測 LLM 的回覆是否包含 `unsafe` 標籤（Unsafe = 1, Safe = 0）。
2. **提示詞有害性預測（Y2）**：預測輸入 Prompt 是否有害（Harmful = 1, Benign = 0）。
3. **安全判定一致性預測（Y3）**：預測 LLM 的安全判定是否與輸入 Prompt 的真實有害性標籤一致（Consistent = 1, Inconsistent = 0）。

---

## 🌟 核心特性

* **5 種主流機器學習模型**：SGD, LR, MLP, RF, LGB。
* **6 層特徵維度全覆蓋**：針對每一層隱藏狀態（1024 維）獨立進行資料預處理（StandardScaler 標準化、RandomUnderSampler 下採樣 1:1、PCA 降維至 128 維）、模型訓練與評估。
* **測試集獨立擴增 (10k/10k)**：自 75,000 資源池擴充 `test1` 與 `test2` 至各 10,000 筆，維持原始分佈比例。
* **雙軌 Isotonic 機率校正**：依據 $y_1 == 0$ 與 $y_1 == 1$ 分分流訓練 `iso_0` 與 `iso_1` 保序迴歸模型。
* **靈活 CLI 繪圖系統**：可參數化單獨選擇任務、資料集、層數、模型與圖表類型進行繪製。

---

## 🚀 快速開始

### 執行管線

```bash
# 1. 訓練基礎探針模型 (模型輸出至 results/unified_training/)
.venv\Scripts\python.exe unified_train.py

# 2. 劃分與擴增測試集 (輸出至 data/test1.pkl, data/test2.pkl)
.venv\Scripts\python.exe evaluation_pipeline/step1_prepare_test_data.py

# 3. 雙軌條件機率校正 (產出快取與指標至 results/safety_guardrails_evaluation/cache/)
.venv\Scripts\python.exe evaluation_pipeline/step2_calibrate.py

# 4. 全量或條件可視化繪圖 (輸出至 results/plots/)
.venv\Scripts\python.exe evaluation_pipeline/step3_plot.py

# 範例：僅單獨繪製 SGD 模型在 y1 任務 test1 資料集的可靠度曲線
.venv\Scripts\python.exe evaluation_pipeline/step3_plot.py --target y1 --split test1 --model SGD --chart reliability
```

---

## 📂 輸出結果結構 (`results/plots/`)

```text
results/plots/
├── 01_Metrics_Trends/                  (Brier Score 與 Log Loss 獨立趨勢圖)
├── 02_Reliability_Curves_split_y/      (分流 iso_0, iso_1 可靠度曲線)
├── 03_Quadrant_Histograms/            (四象限預測置信度直方圖: 全量 2x2)
├── 04_Score_Histograms/               (分流 iso_0, iso_1 正負樣本分數直方圖)
├── 05_Brier_Components/               (分流 iso_0, iso_1 Brier 組分雙 Y 軸圖)
└── 06_Step_Mappings/                  (分流 iso_0, iso_1 保序校正映射階梯圖)
```
