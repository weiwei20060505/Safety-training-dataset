# 安全訓練數據集特徵分析與機率校正框架 (Safety Training Dataset Probe & Calibration Framework)

一個完整的機器學習特徵探針與 post-hoc 機率校正框架，用於訓練和評估多個分類模型，以利用大語言模型（LLM）的內部隱藏狀態（Hidden States）檢測提示詞的有害性、模型回應的安全一致性，並在不同的資料分佈下進行精密機率校正。

---

## 📋 目錄

- [項目簡介](#項目簡介)
- [核心特性](#核心特性)
- [快速開始](#快速開始)
- [專案結構](#專案結構)
- [數據格式](#數據格式)
- [使用指南](#使用指南)
- [模型介紹](#模型介紹)
- [預測任務與數學定義](#預測任務與數學定義)
- [機率校正與評估指標](#機率校正與評估指標)
- [輸出結果說明](#輸出結果說明)
- [進階功能](#進階功能)
- [故障排查](#故障排查)

---

## 🎯 項目簡介

本項目旨在評估和增強 LLM 在對抗性攻擊與常規場景下的安全表現。我們基於 **WildJailbreak 資料集**（包含 Vanilla 原始樣本與 Adversarial 對抗性樣本）進行雙軌特徵預測實驗。我們從 LLM 的 6 個特徵層中提取了輸入序列最後一個 Token 的隱藏狀態（`last_input_hidden_state`）作為模型特徵 $X$。

本專案構建了一個統一的機器學習訓練、評估與校正管線，針對隱藏狀態特徵進行以下三個分類任務的特徵探針（Probing）訓練：

1. **模型回覆安全性預測（Y1）**：預測 LLM 的回覆是否包含 `unsafe` 標籤（Unsafe = 1, Safe = 0）。
2. **提示詞有害性預測（Y2）**：預測輸入 Prompt 是否有害（Harmful = 1, Benign = 0）。
3. **安全判定一致性預測（Y3）**：預測 LLM 的安全判定是否與輸入 Prompt 的真實有害性標籤一致（Consistent = 1, Inconsistent = 0）。

---

## 🌟 核心特性

*   **5 種主流機器學習模型**：
    *   SGD（隨機梯度下降分類器）
    *   LR（邏輯斯迴歸）
    *   MLP（多層感知機神經網路）
    *   RF（隨機森林）
    *   LGB（LightGBM 梯度提升機）
*   **6 層特徵維度全覆蓋**：針對每一層隱藏狀態（1024 維）獨立進行資料預處理（StandardScaler 標準化、RandomUnderSampler 下採樣平衡、PCA 降維至 128 維）、模型訓練與評估。
*   **雙軌學習曲線監控**：
    *   *動態組 (SGD, MLP, LGB)*：支援逐輪（Epoch/Tree）迭代訓練與驗證，附帶 Loss/Accuracy 監控與早停選優。
    *   *靜態組 (LR, RF)*：使用分層 5 折交叉驗證評估 5 種資料量等級（20% 至 100%）下的表現，繪製資料需求量學習曲線。
*   **Post-hoc 機率校正管線**：
    *   利用**保序迴歸（Isotonic Regression）**與 PAV 演算法，針對預測正確性（Correctness）或目標機率進行擬合。
    *   支援 3 種 Bin 劃分策略：**內建區間（Native Bins）**、**動態區間（Adaptive Bins）**與**均勻區間（Uniform Bins）**。
    *   包含**資料擴增 (Augmentation)**與**分佈對齊 (Alignment)**實驗，解決外部測試集分佈偏差（Covariate Shift）下的 ECE 與 Brier 分數優化。
*   **全方位可視化與診斷**：
    *   一鍵繪製 ROC 曲線與 6 大指標（Accuracy, Balanced Accuracy, Precision, Recall, F1, AUC）柱狀圖。
    *   3x3 多維指標對比圖（橫跨 6 層，包含 Brier Score, ECE, Log Loss）。
    *   預測正確/錯誤分組的機率分佈圖與分組校正線圖。

---

## 🚀 快速開始

### 前置要求

*   Python >= 3.10
*   [uv](https://github.com/astral-sh/uv)（強烈推薦）或 pip

### 安裝依賴

```bash
# 使用 uv（推薦）
uv sync

# 或使用 pip
pip install -r pyproject.toml
```

### 配置環境變數

在項目根目錄下創建 `.env` 檔案（參考 `.env.example`）：

```env
HF_TOKEN=your_huggingface_token
BASE_URL=https://api.your-model-server.com/v1
API_KEY=your_api_key
MODEL_NAME=your_model_name
```

### 執行管線

專案的執行流程已模組化為以下步驟：

#### 步驟 1：特徵提取與推論（若已有 pkl 資料可跳過）

```bash
# 對訓練集（預設抽取 2000 筆，可加 --all 使用全量）進行非同步推論與特徵提取
python Async_run_experiment_train.py

# 對外部評估集進行非同步推論與特徵提取
python Async_run_experiment_eval.py
```

#### 步驟 2：執行統一訓練與評估框架

```bash
python unified_train.py
```

*   **輸入**：`experiment_results_train_10000.pkl`（10,000 筆基準數據）
*   **輸出**：`results/unified_training/` 目錄
    *   `training_log.txt`：完整的訓練與分層評估指標日誌。
    *   `layer_*/`：保存各層 5 大模型的最優權重（`*_best.pkl`）與最終權重（`*_last.pkl`），以及 ROC、Accuracy 和 Learning Curves 曲線。

#### 步驟 3：資料調整與對齊準備

```bash
python run_06_three_versions_adjusted_data.py
```

*   **功能**：準備 `test1`、`test2` 及外部評估集。透過資料擴增（Augment）和比例對齊（Align），生成 `augmented_test1.pkl`、`aligned_test1.pkl` 等多個資料版本，作為後續校正評估基礎。

#### 步驟 4：核心機率校正、Brier 分解與可靠度分析

```bash
python run_07_brier_evaluation.py
```

*   **功能**：在不同資料分佈上進行 Isotonic Regression 擬合，計算 Brier Score 及 Log Loss，繪製指標隨層數變化折線圖。同時，除了產出原有的可靠度曲線外，新增**依 $y_i$ 真實標籤（$y_i == 1$ 與 $y_i == 0$）分割的 1x5 可靠度對比折線圖 (方案 A)**，保存至全新的 `02_Reliability_Curves_split_y` 目錄下。
*   **輸出**：`results/safety_guardrails_evaluation/{dataset_key}/02_Reliability_Curves_split_y/`

#### 步驟 5：預測正確性分組診斷與直方圖分析

```bash
python run_08_histogram_analytics.py
```

*   **功能**：針對預測正確與預測錯誤之四個象限，統計置信度分數的直方圖分佈並進行 Brier 指標分解。將生成的直方圖儲存至**依特徵層分類（`layer_{layer_num}`）的目錄結構**中，便於各模型在同層的特徵表現進行直觀對比。
*   **輸出**：`results/safety_guardrails_evaluation/{dataset_key}/03_Quadrant_Histograms/`


---

## 📁 專案結構

```
.
├── Async_run_experiment_train.py    # 非同步推論與訓練集隱藏狀態提取
├── Async_run_experiment_eval.py     # 非同步推論與評估集隱藏狀態提取
├── unified_train.py                 # 統一訓練框架（5大模型 x 6層 x 3任務）
├── run_06_three_versions_adjusted_data.py # 步驟3：產生經調整後的校正測試集
├── run_07_brier_evaluation.py       # 步驟4：核心校正評估、Brier 隨層折線圖與 1x5 分割可靠度圖
├── run_08_histogram_analytics.py    # 步驟5：四象限直方圖（按特徵層分類）與 Brier 分解分析
├── utils_calibration.py             # 校正輔助工具（含中文字型、Binning、Brier 分解與 1x5 繪圖）
├── wrapper_models.py                # 正確性分類器包裝器 (Correctness Wrapper)
├── check_data.py                    # 數據格式檢查與分佈統計工具
├── test_dataset.py                  # 測試數據集載入與類別佔比計算
├── test_LLM_model.py                # 測試 API 伺服器連線狀態
├── pyproject.toml                   # 項目依賴與環境設定
├── .env.example                     # 環境變數範本
└── results/                         # 結果輸出目錄
    ├── unified_training/            # 訓練權重與原始指標圖表
    └── safety_guardrails_evaluation/ # 安全探針後驗評估與分析結果
        ├── data_aug/                # 資料增強組 (Augmented) 評估圖表
        │   ├── 01_Metrics_Trends/   # Brier Score 與 Log Loss 隨層變化折線圖
        │   ├── 02_Reliability_Curves/ # 原始可靠度曲線圖表
        │   ├── 02_Reliability_Curves_split_y/ # 依 y_i 拆分之 1x5 可靠度對比折線圖 (方案 A)
        │   └── 03_Quadrant_Histograms/ # 四象限直方圖 (按層分類)
        └── data_align/              # 資料對齊組 (Aligned) 評估圖表 (同上結構)
```

---

## 📊 數據格式

### 特徵矩陣與標籤結構 (Pickle 格式)

```python
{
    'hidden_state': list,       # Shape: (num_samples, 6, 1024) - 6層，每層1024維
    'data_type': str,           # 'adversarial_harmful', 'adversarial_benign', 'vanilla_harmful', 'vanilla_benign'
    'model_reply': str,         # 'SAFE' 或 'UNSAFE'
    'prompt': str,              # 原始輸入提示詞
    'prompt_source': str,       # 'adversarial' 或 'vanilla'
}
```

### 數據集統計資訊

*   **基準訓練集 (`experiment_results_train_10000.pkl`)**：10,000 筆樣本。
    *   `data_type` 分布：對抗有害 3,197 筆，對抗無害 3,055 筆，常規無害 1,889 筆，常規有害 1,859 筆。
    *   `model_reply` 分布：UNSAFE 6,392 筆，SAFE 3,608 筆。
*   **外部評估集 (`experiment_results_eval.pkl`)**：2,210 筆樣本。

---

## 🎯 預測任務與數學定義

### Y1 - 模型回覆安全性預測 (Model Reply Safety)

*   **目標**：預測 LLM 的回覆是否安全（Unsafe = 1, Safe = 0）。
*   **標籤來源**：`model_reply` 欄位包含 `unsafe` (不分大小寫) 則為 1，否則為 0。
*   **公式**：
    $$Y_1 = \mathbb{I}(\text{"unsafe"} \in \text{model\_reply.lower()})$$

### Y2 - 提示詞有害性預測 (Prompt Harmfulness)

*   **目標**：預測原始輸入提示詞是否有害（Harmful = 1, Benign = 0）。
*   **標籤來源**：`data_type` 欄位包含 `harmful` 則為 1，否則為 0。
*   **公式**：
    $$Y_2 = \mathbb{I}(\text{"harmful"} \in \text{data\_type})$$

### Y3 - 安全判定一致性預測 (Safety Consistency)

*   **目標**：預測 LLM 的安全判定是否與提示詞有害性真實標籤一致（Consistent = 1, Inconsistent = 0）。
*   **公式**：
    $$Y_3 = \mathbb{I}(Y_1 == Y_2)$$
*   **物理意義**：
    *   $Y_3 = 1$（一致）：
        *   Harmful 提示詞 $\rightarrow$ UNSAFE 模型回應（成功攔截/標註）
        *   Benign 提示詞 $\rightarrow$ SAFE 模型回應（正常放行）
    *   $Y_3 = 0$（不一致）：
        *   Harmful 提示詞 $\rightarrow$ SAFE 模型回應（漏報/對抗攻擊成功）
        *   Benign 提示詞 $\rightarrow$ UNSAFE 模型回應（誤報/過度防禦）

---

## ⚖️ 機率校正與評估指標

### Correctness 預測正確性校正

對於 $Y_1$ 與 $Y_2$ 任務，為了使校正後的機率代表「模型預測正確的機率」，專案引入了 `CorrectnessClassifierWrapper`。
1. 若基礎模型預測類別 1 的機率為 $p$，且設定分類閥值為 $0.5$：
   *   當 $p \ge 0.5$ 時，預測類別為 1，預測正確的機率即為 $p$。
   *   當 $p < 0.5$ 時，預測類別為 0，預測正確的機率為 $1 - p$。
2. 校正的目標二元標籤轉換為：
   $$y_{\text{correctness}} = \mathbb{I}(\text{pred} == y_{\text{true}})$$
3. 使用保序迴歸（Isotonic Regression）擬合非遞減映射函數 $f(p_{\text{correctness}})$，以在驗證集上最小化均方誤差。對於 $Y_3$ 任務，則直接對 $Y_3$ 的二元機率進行校正。

### 評估指標數學公式

#### 1. Brier Score (布里爾分數，越低越好)

$$\text{Brier} = \frac{1}{N} \sum_{i=1}^{N} (p_i - y_i)^2$$

#### 2. Expected Calibration Error (ECE，預期校正誤差，越低越好)

將預測機率劃分為 $K$ 個 Bins，在每個 Bin $B_m$ 中計算平均預測置信度與實際正例比例的差值：

$$\text{ECE} = \sum_{m=1}^{K} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

#### 3. Log Loss (對數損失，越低越好)

$$\text{Log Loss} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]$$

---

## 📈 輸出結果說明

所有繪圖和模型權重均存儲於 `results/` 下。
*   `unified_training/layer_X/`：
    *   `*_best.pkl`：在驗證集上損失最低的最佳 Pipeline。
    *   `*_last.pkl`：最後一輪訓練的 Pipeline。
    *   `learning_curves_step_*.png`：動態組逐輪收斂曲線。
    *   `learning_curves_data_*.png`：靜態組資料需求量學習曲線。
*   `safety_guardrails_evaluation/`：
    *   `data_aug/` & `data_align/`：
        *   `01_Metrics_Trends/metrics_comparison_*.png`：跨特徵層的 Brier Score 與 Log Loss 折線圖。
        *   `02_Reliability_Curves/*/layer_*_reliability.png`：標準可靠度曲線。
        *   `02_Reliability_Curves_split_y/*/layer_*_reliability.png`：依 $y_i$ 拆分的 1x5 分組對比校正折線圖（方案 A）。
        *   `03_Quadrant_Histograms/*/layer_*/`：按特徵層分類儲存的四象限置信度直方圖。


---

## 🔧 進階功能

1.  **調整 PCA 降維維度**：
    在 `unified_train.py` 中的 `train_sgd` 等方法中，修改 `PCA(n_components=128)`。
2.  **更改資料下採樣策略**：
    預設使用 `RandomUnderSampler` 進行 $1:1$ 平衡，如需調整比例或關閉，可在 `unified_train.py` 的 `UnifiedModelTrainer` 中修改 `RandomUnderSampler` 的參數。
3.  **自定義機率校正 Bin 數**：
    在 `utils_calibration.py` 中，修改 `get_adaptive_bins` 的 `n_bins` 參數以適應不同的數據規模。

---

## 🔍 故障排查

### 問題 1：找不到數據文件

```
FileNotFoundError: [Errno 2] No such file or directory: 'experiment_results_train_10000.pkl'
```

*   **解決**：
    1.  確認數據文件是否位於專案根目錄。
    2.  若尚未提取特徵，請先配置 `.env`，並執行 `python Async_run_experiment_train.py` 獲取數據。

### 問題 2：Matplotlib 繪圖中文出現方塊 (Unicode 缺失)

*   **解決**：
    專案已在 `utils_calibration.py` 中引入 `setup_chinese_font`，會自動搜尋系統中的微軟正黑體 (`Microsoft JhengHei`)、黑體 (`SimHei`) 等。如果仍有問題，請確保您的系統安裝了中文字型，或者在 Linux 環境下配置相應的中文字型檔。

### 問題 3：LightGBM 警告

*   **解決**：
    LightGBM 訓練時如果輸出過多日誌，已在 `LGBMClassifier` 中設定 `verbose=-1` 關閉不必要的警告。

---

**版本**：1.1.0
**最後更新**：2026-07-14
