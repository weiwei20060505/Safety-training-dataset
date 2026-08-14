# 🚀 V1 Baseline 完整重現與跨機器執行指南 (Cross-Machine Execution Guide)

本指南用於在已擁有原始 `data/experiment_results_train.pkl` (4.8GB) 數據檔的任意新電腦/環境上，完整重現 **V1 Baseline 終極訓練、機率校正與評估全流程**。

---

## 📋 事前準備與環境檢查

1. **取得最新程式碼**：
   在終端機 (Terminal) 執行：
   ```bash
   git pull origin main
   ```
2. **確認原始資料檔位置**：
   確保 `data/experiment_results_train.pkl` 與 `data/experiment_results_eval.pkl` 已放置於專案根目錄的 `data/` 資料夾中。
3. **確認 Python 環境與依賴套件**：
   建議使用 `.venv` 虛擬環境，需包含 `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `imbalanced-learn`, `joblib`, `pyarrow`, `matplotlib`。

---

## ⚙️ 執行流程 (Step-by-Step Workflow)

### 🔹 步驟 1：黃金比例數據集切割 (Data Partitioning)
執行資料切割腳本，利用固定隨機種子 (`random_state=42`) 從 85,000 筆主資料池中切出四份百分之百不重疊的數據集：

```bash
.venv\Scripts\python.exe scratch\prepare_v1_data.py
```
*(非 Windows 或全域環境請使用 `python scratch/prepare_v1_data.py`)*

#### 切割成果確認 (`data/` 目錄)：
- `v1_train_full.pkl`：**77,999 筆** (全量訓練集，保留自然分佈，無欠採樣，保障 Exchangeability)
- `v1_val.pkl`：**2,000 筆** (獨立驗證集，早停 Early Stopping 監控)
- `v1_test1.pkl`：**2,000 筆** (獨立校正集，1D 保序迴歸 Isotonic Calibration 用)
- `v1_test2.pkl`：**3,000 筆** (獨立測試集，最終論文與 ECE 評估用)

---

### 🔹 步驟 2：全自動管線批次執行 (Unified Batch Execution - 推薦)

可以直接執行統整後的 V1 全流程一鍵批次檔：

```cmd
.\run_full_v1.bat
```

該腳本會依序完成：模型訓練 ➔ 雙軌保序機率校正 ➔ 自訂圖表繪製 ➔ 組合大圖拼接。

---

### 🔹 步驟 3：分步手動執行指令 (Step-by-Step Manual Execution)

如需拆解步驟個別執行，指令如下（全專案已完全移除 PCA，直接對 1,024 維特徵進行處理）：

#### 1. 第一階段探針全量訓練 (Stage 1 Probe Training)
```bash
.venv\Scripts\python.exe pipeline_v1_baseline\unified_train.py --train_data data\v1_train_full.pkl --val_data data\v1_val.pkl --output_suffix all_models_y2_78k
```
> **關鍵模型設定**：
> - 模型：MLP, LightGBM, LogisticRegression × 6 個 Hidden Layer
> - 特徵空間：1,024 維原始特徵 (Raw Hidden State Features, 無 PCA 降維)
> - 類別分佈：自然真實分佈 (無 RandomUnderSampler)

#### 2. 第二階段統一機率校正 (Stage 2 Probability Calibration)
```bash
.venv\Scripts\python.exe pipeline_v1_baseline\step2_calibrate.py
```
> **校正模型設定**：
> - 採用統一 Isotonic Regression 擬合標籤 $y_3 = \mathbb{I}(y_1 == y_2)$，保證校正曲線單調遞增。

#### 3. 第三階段圖表繪製 (Stage 3 Plotting & Visualization)
```bash
# 生成核心自訂診斷圖 (Trends, Reliability, Joint Calibration)
.venv\Scripts\python.exe pipeline_v1_baseline\plot_v1_custom.py

# 生成標準大圖拼接
.venv\Scripts\python.exe pipeline_v1_baseline\step4_combine_plots.py
```

---

## 📁 成果與輸出目錄說明

- **模型權重與訓練日誌**：`models/v1_baseline/unified_training/lgb_y2_all_models_y2_78k/`
- **校正快取與指標檔**：`cache/v1_baseline/calibration/`
- **視覺化產物目錄**：`results/v1_baseline/plots_custom/`
