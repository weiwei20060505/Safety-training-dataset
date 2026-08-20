# LLM 安全防護特徵分析、特徵探針與 post-hoc 機率校正系統

本項目是一個完整的機器學習特徵探針（Representation Probing）與後驗（post-hoc）機率校正框架。項目旨在利用大型語言模型（LLM）內部的激活特徵（Hidden States），在模型尚未輸出文字前判定輸入提示詞（Prompt）是否有害、模型回覆是否安全，以及安全判定是否一致，並導入保序迴歸（Isotonic Regression）機率校正，還原真實的預測信賴度。

---

## 🎯 專案背景與三大分類任務

隨着 LLM 被廣泛部署，如何防範惡意對抗性提示詞（如 Jailbreak 越獄攻擊）成為核心議題。傳統的安全對齊主要依賴於生成後的文字匹配或外部安全篩選 API，這會帶來顯著的推論延遲。

本項目採用特徵探針技術，直接讀取 LLM 隱藏層的神經元激活狀態（Hidden States），在模型尚未輸出文字前進行預測。本框架定義了三個核心分類任務：

1. **$y_1$ 任務 (模型回覆安全性，Model Reply Safety)**：  
   預測模型回覆是否包含 `unsafe` 標籤（Unsafe = 1, Safe = 0）。
2. **$y_2$ 任務 (提示詞有害性，Prompt Harmfulness)**：  
   預測輸入提示詞是否有害（Harmful = 1, Benign = 0）。
3. **$y_3$ 任務 (安全判定一致性，Safety Consistency)**：  
   預測 LLM 的安全判定是否與輸入 Prompt 的真實有害性一致（Consistent = 1, Inconsistent = 0）。數學定義為指示函數：
   $$y_3 = \mathbb{I}(y_1 == y_2)$$
   當輸入提示詞有害且模型成功攔截為 unsafe，或提示詞無害且模型放行為 safe 時，$y_3$ 為 1。

---

## ⚙️ 資料來源與特徵工程

* **原始資料集 (WildJailbreak)**：包含對抗樣本（Adversarial Prompts）與常規樣本（Vanilla Prompts）。
* **類別分佈與占比統計**（從 260,000 筆訓練資料中，以隨機抽樣 `random_state=42` 提取 84,999 筆做為訓練集，繼承了原始資料庫完美的 50:50 平衡）：
  1. **`adversarial_harmful`** (對抗有害提示詞)：`32.97%`（28,028 筆）
  2. **`adversarial_benign`** (對抗無害提示詞)：`31.20%`（26,521 筆）
  3. **`vanilla_benign`** (一般無害提示詞)：`19.59%`（16,648 筆）
  4. **`vanilla_harmful`** (一般有害提示詞)：`16.24%`（13,802 筆）
* **特徵空間**：提取 LLM 處理 Prompt 時在 6 個特定隱藏層的 `last_input_hidden_state`。
  * 維度：$X \in \mathbb{R}^{M \times 6 \times 1024}$（全量 1024 維，完全移除 PCA 降維，保留完整神經元資訊）。

---

## 🧱 V1 全域管線架構與痛點分析

### 1. $y_2 \to y_3$ 的機率轉換邏輯
實驗發現直接預測 $y_3$ 難以收斂，而預測 $y_2$ 效果極佳。因此 V1 架構採用了**「兩階段映射策略」**：
* 探針先預測輸入 Prompt 為有害的機率 $P(y_2=1) = p$。
* 根據已知的 $y_1$ 狀態，將 $p$ 暴力轉換為 $y_3$ 的先驗分數（Ranking Score）：
  ```python
  pre_cal_score = np.where(y1 == 1, p, 1 - p)
  ```
* 最後將 `pre_cal_score` 送入單一全域的 PAVA (Isotonic Regression) 進行機率校正。

### 2. 模型選擇考量 (排除 RF)
在 V1 的訓練中，我們測試了 MLP、LGB、LR 以及 RF (Random Forest)。但綜合考量到**推理效能與實務部署**，目前的腳本與圖表已全面排除 RF，將重心放在擬合力最強的 **MLP**、運算極具效率的 **LightGBM (LGB)** 以及做為基準的 **Logistic Regression (LR)**。

### 3. V1 的致命傷：條件校正漂移 (Subgroup Calibration Drift)
當我們把 `np.where` 翻轉後的 $1-p$ ($y_1=0$) 與 $p$ ($y_1=1$) 強行丟進**同一個分數池**裡擬合 PAVA 曲線時，產生了嚴重的失真：
- **混合分佈危機**：$y_1=0$ 子群（受對抗樣本干擾）與 $y_1=1$ 子群（越獄成功樣本為主）原本具有完全不同的「過度自信 (Overconfidence)」特徵與先驗分佈。
- **互相拉扯**：單一 Isotonic 曲線無法同時適配這兩股異質分佈的誤差，導致模型在子群內部「越校正越失真」，造成嚴重的**先驗機率偏差 (Prior Shift)**。

---

## 🌟 V2 條件分流框架 (Conditional Framework)

為了解決 V1 全域校正的理論漏洞，V2 導入了從網路底層到校正端一脈相承的「**雙軌條件分流架構**」：

1. **神經網路底層分流 (Conditional Routing)**：
   我們設計了多種雙軌模型結構，直接將 $P(y_2 \mid h_L, y_1=0)$ 與 $P(y_2 \mid h_L, y_1=1)$ 徹底拆開學習：
   * **YHead_MLP**：共用底層特徵抽取，在頂層針對 $y_1$ 建立兩個不同的預測 Head。
   * **HardDual**：完全獨立的兩個模型，分別只看 $y_1=0$ 與 $y_1=1$ 的資料。
   * **Interaction/RootSplit** 等多種變體探針。

2. **雙軌條件校正 (Subgroup PAVA Calibration)**：
   校正階段不再混合資料，而是**嚴格根據 $y_1=0$ 與 $y_1=1$ 分拆為兩大子群，分別擬合兩條專屬的 PAVA 校正曲線**。
   這完美消除了 V1 子群間互相拉扯的漂移現象，讓 Log Loss 首度跌破 0.3 大關，可靠度圖表 (Reliability Diagram) 完美貼合 45 度理想對角線。

---

## 🎯 未來工作與展望 (Future Work)

1. **細粒度樣本分群 (Fine-Grained Subgrouping)**：
   除了目前的 $y_1$ 分流外，計畫加入資料庫內建的 **對抗樣本 (Adversarial) vs 常規樣本 (Vanilla)** 提示詞進行更細緻的分群。
2. **追求資料可交換性 (Exchangeability)**：
   探討與設計抽樣機制，嘗試讓 `Train`, `Test1`, `Test2` 與外置 `Eval` 之間達成統計上的可交換性，解決跨資料集 (Domain Shift) 下的校正失效。
3. **元評估 (Meta-Evaluation)**：
   在獨立測試集 (`Test 2`) 產出 ECE 與 Risk-Coverage 曲線，選拔終極金牌 Safety Guardrail。

---

## 📁 專案目錄結構

```text
Safety-training-dataset/
├── data/                          # 基準數據與測試集 (git忽略，需手動產生)
├── cache/                         # 預測快取與指標 (git忽略)
├── models/                        # 模型權重 (git忽略)
├── results/                       # 視覺化圖表與實驗紀錄 (.png, .log, .md 允許同步)
│   ├── v1_baseline/               # 🔵 V1 Baseline 產出
│   │   └── plots_custom/          # V1 自訂診斷圖 (包含 2x3 聯合 ROC 大圖)
│   └── v2_framework/              # 🟢 V2 Framework 產出
│
├── pipeline_v1_baseline/          # 🔵 V1 Baseline 管線程式碼
├── pipeline_v2_framework/         # 🟢 V2 Framework 管線程式碼 (YHead 等結構)
├── LLM 隱藏狀態機率校正與元評估框架 8月20日.md # Marp 專案核心簡報檔
├── AI_INSTRUCTIONS.md             # AI 協作需知 (含核心邏輯與守則)
├── 執行結果.md                    # 實驗執行日誌與計畫追蹤
├── run_full_v1.bat                # ⚡ V1 批次執行檔
└── run_full_v2.bat                # ⚡ V2 批次執行檔
```

---

## 🚀 快速執行 (Quick Start)

### 🔹 步驟 1：黃金比例數據集切割
執行資料切割腳本，利用固定隨機種子 (`random_state=42`) 從 85,000 筆主資料池中切出四份互不重疊的數據集：
```cmd
.venv\Scripts\python.exe scratch\prepare_v1_data.py
```
*(非 Windows 或全域環境請使用 `python scratch/prepare_v1_data.py`)*

#### 切割成果 (`data/`)：
* `v1_train_full.pkl`：**77,999 筆** 
* `v1_val.pkl`：**2,000 筆** (獨立驗證集，早停 Early Stopping 監控)
* `v1_test1.pkl`：**2,000 筆** (獨立校正集，擬合 PAVA 用)
* `v1_test2.pkl`：**3,000 筆** (獨立測試集，元評估與雙軌驗證用)

### 🔹 步驟 2：全自動管線批次執行

**執行 V1 管線** (探針訓練 ➔ 全域校正 ➔ 自訂圖表繪製)：
```cmd
.\run_full_v1.bat
```

**執行 V2 條件分流管線** (條件模型訓練 ➔ 子群校正 ➔ 聯合繪圖)：
```cmd
.\run_full_v2.bat
```

### 🔹 數學原理備註
*   **PAVA (Pool Adjacent Violators Algorithm)**：為非參數單調映射，能動態合併破壞單調性的區塊，將原始分數轉為分段常數的經驗機率，直接優化 Brier Score。
*   **轉折點遮罩 (Turning Point Mask)**：解決 Isotonic 密集水平點在 Matplotlib 渲染出灰色鋸齒瑕疵的繪圖技術。

---

## 🤖 AI 專屬專案架構與程式碼導航 (AI Developer Reference)

此區塊為 AI 協作助手設計，目的為避免 AI 在未讀取程式碼前產生幻覺，請 AI 在執行任務前務必參考以下規格：

### 1. 核心資料結構與變數
*   **Input Features ($X$)**：來自 `last_input_hidden_state`。程式碼讀取自 `df['hidden_state']`。維度為 `(N, 6, 1024)`，代表 6 個層 (Layer 1 ~ Layer 6)，每層 1024 維。模型訓練時是**逐層 (per-layer)** 拆開獨立訓練的。
*   **Labels ($y$)**：
    *   `y_1`：`df['model_reply'].str.contains('unsafe')` (0=Safe, 1=Unsafe)
    *   `y_2`：`df['data_type'].str.contains('harmful')` (0=Benign, 1=Harmful)
    *   `y_3`：`y_1 == y_2`

### 2. 機器學習模型超參數 (V1 Unified Train)
為了對齊實驗結果，本專案固定使用以下 sklearn / lightgbm 參數：
*   **MLP (多層感知機)**：`MLPClassifier(hidden_layer_sizes=(128,), random_state=42, alpha=0.01)`。實作上利用 `partial_fit` 搭配 Batch Size = 64 訓練 100 Epochs。
*   **LGB (LightGBM)**：`LGBMClassifier(n_estimators=1000, learning_rate=0.03, max_depth=4, num_leaves=31, reg_alpha=0.5, reg_lambda=0.5, min_child_samples=20)`。
*   **LR (邏輯迴歸)**：`LogisticRegression(max_iter=1000, random_state=42)`。
*   **⚠️ 模型限制**：Random Forest (RF) 因效能問題已自訓練列表 `models_to_train` 中拔除，請勿捏造 RF 數據。

### 3. 校正實作細節 (PAVA)
*   **核心工具**：使用 `sklearn.isotonic.IsotonicRegression(out_of_bounds='clip')`。
*   **V1 的單軌寫法**：直接對全量 `Test 1` 的預測機率進行 `.fit()`，再對 `Test 2` 與 `Eval` 進行 `.predict()`。
*   **V2 的雙軌寫法**：根據 $y_1$ 分流，先對 $y_1=0$ 的 `Test 1` 樣本 `.fit()` 得到 `iso_0`，再對 $y_1=1$ 的樣本 `.fit()` 得到 `iso_1`。預測時依據該筆資料的 $y_1$ 狀態，切換使用 `iso_0` 或 `iso_1` 進行機率校正。

### 4. 評估指標 (Metrics)
*   **Brier Score**：即均方誤差 (MSE)，$\frac{1}{N} \sum (\hat{p}_i - y_i)^2$。在 `results/v1_baseline/calibration` 的 csv 檔案中，分為 `raw_brier` (校正前) 與 `cal_brier` (PAVA 校正後)。
*   **Log Loss**：即二元交叉熵 (Binary Cross-Entropy)。
*   **AUC (ROC-AUC)**：不受機率校正影響（因為 Isotonic 是單調保序變換，不會改變預測樣本的排名），所以校正前後的 AUC 會完全一致。
