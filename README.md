# LLM 安全特徵探針與機率校正框架 (Safety Feature Probing & Calibration Framework)

本專案是一個用於提取、訓練、評估與校正大型語言模型（LLM）內部安全表徵特徵（Hidden States）的完整機器學習框架。本專案基於 **WildJailbreak 資料集**（包含 vanilla 原始提示詞與 adversarial 對抗性提示詞），從 LLM 的 6 個特徵層中提取最後一個 Token 的 `last_input_hidden_state`，進而訓練 5 種機器學習分類器（SGD, MLP, LightGBM, Logistic Regression, Random Forest）進行安全分類，並使用 **保序迴歸 (Isotonic Regression)** 對預測機率進行校正。

---

## 1. 專案目錄結構說明

本專案的核心檔案及其主要功能如下（已排除舊版代碼）：

```bash
├── .env.example                     # 環境變數範例配置檔案
├── pyproject.toml                   # 專案依賴與 Python 版本設定（基於 uv 規範）
├── uv.lock                          # 依賴精確鎖定檔案
├── Async_run_experiment_train.py    # [非同步] 對 WildJailbreak 訓練集進行 LLM 推論與特徵提取
├── Async_run_experiment_eval.py     # [非同步] 對 WildJailbreak 評估集進行 LLM 推論與特徵提取
├── check_data.py                    # 檢查提取後的特徵維度與 LLM 安全分類對應關係
├── unified_train.py                 # [核心] 統一機器學習模型訓練、評估與繪圖主程式（雙軌監控）
├── uncorrected_graph.py             # 繪製各模型在未校正狀態下的 Calibration Curves 與 Bins 數據
├── Isotonic_Regression_corrected.py # 對各模型進行保序迴歸校正，並繪製校正後的 Calibration Curves
└── results/                         # 輸出資料夾，存放訓練日誌、模型權重與分析圖表
    └── unified_training/
        ├── training_log.txt         # 完整模型訓練日誌
        └── layer_1~6/               # 各特徵層的專屬輸出目錄（包含圖片與 .pkl 模型）
```

---

## 2. 環境建置與安裝

本專案推薦使用 Python 包管理工具 `uv` 或標準的 `pip` 進行安裝。

### A. 使用 `uv` (推薦)
1. 安裝 `uv`（若尚未安裝）：
   ```bash
   pip install uv
   ```
2. 使用 `uv` 建立虛擬環境並安裝依賴：
   ```bash
   uv venv
   .venv\Scripts\activate   # Windows 系統
   uv pip install -r pyproject.toml
   ```

### B. 使用 `pip` 直接安裝
```bash
pip install datasets lightgbm matplotlib openai python-dotenv scikit-learn tqdm imbalanced-learn joblib
```

---

## 3. 設定環境變數

在專案根目錄下，複製 `.env.example` 並重新命名為 `.env`：
```bash
cp .env.example .env
```
編輯 `.env` 並填入對應的 API 金鑰與 Token：
- `HF_TOKEN`：用於下載 WildJailbreak 資料集的 Hugging Face API Token。
- `BASE_URL`：自定義大模型 API 的終端點（如 local vLLM 或 API 代理）。
- `API_KEY`：自定義大模型 API 的驗證金鑰。
- `MODEL_NAME`：大模型名稱（例如 `xecguard_experimental`）。

---

## 4. 完整執行流程

請依照以下順序執行各個腳本來重現實驗：

### 步驟 1：非同步提取特徵數據
我們採用非同步並行調用與指數退避機制，快速調用 LLM 並提取隱藏狀態：
```bash
# 提取訓練集特徵（可選隨機抽取 N 筆或使用全部）
python Async_run_experiment_train.py --n_samples 2000

# 提取評估集特徵
python Async_run_experiment_eval.py
```
*這將產生 `experiment_results_train.pkl`、`experiment_results.pkl` 等特徵檔案。*

### 步驟 2：資料完整性檢查與 LLM 安全率分析
分析 LLM 本身對於對抗樣本與原始樣本的混淆矩陣與安全性分類準確率：
```bash
python check_data.py
```

### 步驟 3：統一模型訓練與評估
訓練 5 種模型（SGD, MLP, LGB, LR, RF）對 6 層特徵分別擬合 $Y_1$ (有害性分類) 與 $Y_3$ (安全一致性分類)：
```bash
python unified_train.py
```
*程式會自動進行下採樣與 PCA 降維，並在 `results/unified_training/layer_{1~6}/` 中生成模型對比條形圖與雙軌學習曲線圖。*

### 步驟 4：評估未校正模型的機率偏差
分析模型原生輸出機率的精確度，並輸出 Bins 的詳細分佈：
```bash
python uncorrected_graph.py
```
*這將在各層目錄下輸出 `calibration_uncorrected_y1.png` 與 `calibration_uncorrected_y3.png`。*

### 步驟 5：執行 Isotonic Regression 校正
在驗證集上擬合 Isotonic Regression 並校正測試集機率：
```bash
python Isotonic_Regression_corrected.py
```
*這將儲存校正後的模型 `*_calibrated.pkl` 並輸出完美的可靠性對比圖 `calibration_corrected_y1/y3.png`。*

---

## 5. 訓練模型用的方法與超參數調整

為了公平比較各模型在 LLM 表徵空間上的分類邊界，所有模型都封裝在同一個管道（Pipeline）中，進行相同的特徵工程：

1. **StandardScaler**：特徵標準化，消除各特徵維度的尺度影響。
2. **RandomUnderSampler**：隨機下採樣以達到 $1:1$ 的平衡類別比例，消除偏置。
3. **PCA**：提取前 **128** 個主成分，減少共線性並加速模型收斂。

### 各分類器參數調優細節

* **Stochastic Gradient Descent (SGDClassifier)**:
  * 參數設定：`loss='log_loss'` (使其輸出機率分數), `penalty='l2'` (正則化), `alpha=0.01` (正則化強度), `learning_rate='adaptive'`, `eta0=0.01` (初始學習率)。
  * 特點：使用小批量迭代 `partial_fit`，實時監控 Epoch 學習曲線。
* **Logistic Regression (LR)**:
  * 參數設定：`C=0.01` (設定較大的正則化強度防範過擬合), `penalty='l2'`, `max_iter=1000`。
  * 特點：靜態組模型，透過 Cross-Validation 評估 5 份不同訓練資料量下的極限性能。
* **Multi-Layer Perceptron (MLPClassifier)**:
  * 參數設定：`hidden_layer_sizes=(128,)` (單隱藏層), `alpha=0.01` (L2 正則化), `batch_size=64`, 迭代 100 輪。
  * 特點：利用 `partial_fit` 進行非線性感知訓練，防範過擬合。
* **Random Forest (RF)**:
  * 參數設定：`n_estimators=100`, `max_depth=10` (放寬決策樹最大深度至 10 以增強擬合複雜表徵的能力)。
  * 特點：Bagging 集成模型，同樣透過切分 5 份資料量分析數據邊際效益。
* **LightGBM (LGBMClassifier)**:
  * 參數設定：`n_estimators=100`, `learning_rate=0.05`, `max_depth=10`, `num_leaves=31`, `reg_alpha=0.05` (L1 正則), `reg_lambda=0.05` (L2 正則), `class_weight='balanced'`。
  * 特點：放寬樹的深度與葉子節點數以捕捉更具彈性的非線性決策邊界，並加入雙重正則化防止過擬合。

---

## 6. 核心數學原理

### 1. 特徵標準化 (Standardization)
對於維度為 $d$ 的特徵向量 $x = [x_1, \dots, x_d]^T$，每個維度單獨進行變換：
$$\hat{x}_j = \frac{x_j - \mu_j}{\sigma_j}$$
其中 $\mu_j$ 與 $\sigma_j$ 分別為訓練集中該特徵分量的平均值與標準差。

### 2. 主成分分析 (PCA)
給定一個已標準化且中心化的特徵矩陣 $X \in \mathbb{R}^{N \times d}$，其樣本協方差矩陣 $\Sigma \in \mathbb{R}^{d \times d}$ 定義為：
$$\Sigma = \frac{1}{N-1} X^T X$$
通過特徵值分解求得正交的特徵向量 $v_1, v_2, \dots, v_d$ 與相對應的特徵值 $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_d$：
$$\Sigma v_j = \lambda_j v_j$$
選擇前 $k=128$ 個最大特徵值對應的向量構成投影矩陣 $V_k = [v_1, v_2, \dots, v_k] \in \mathbb{R}^{d \times k}$，則降維後的樣本表示為：
$$Z = X V_k$$

### 3. 保序迴歸 (Isotonic Regression)
給定模型在驗證集上預測的機率分數 $S_i$ 以及真實類別標籤 $y_i \in \{0, 1\}$，保序迴歸尋找一個階梯狀單調遞增函數 $f(S)$ 使得平滑殘差平方和最小化：
$$\min_{f} \sum_{i=1}^{M} (y_i - f(S_i))^2 \quad \text{subject to } f(S_a) \le f(S_b) \text{ whenever } S_a \le S_b$$
在 `scikit-learn` 中，該問題採用 **Pool Adjacent Violators (PAV) 演算法**進行非參數化求解。該方法非常適合修正模型預測中因下採樣或模型偏好導致的非線性扭曲，使預測機率重新回歸物理現實。
