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
3. **確認 Python 環境與依據套件**：
   建議使用 `.venv` 虛擬環境，需包含 `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `imbalanced-learn`, `joblib`, `pyarrow`。

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

### 🔹 步驟 2：第一階段五大模型全量訓練 (Stage 1 Probe Training)
執行 5 種分類演算法 (`SGD`, `MLP`, `LGB`, `LR`, `RF`) × 6 個特徵層的全量訓練：

**方法 A：直接執行批次腳本 (推薦)**
```cmd
.\run_v1_lgb_expansion.bat
```

**方法 B：手動指令執行**
```bash
.venv\Scripts\python.exe pipeline_v1_baseline\unified_train.py --no_pca --train_data data\v1_train_full.pkl --val_data data\v1_val.pkl --output_suffix all_models_y2_78k
```

> **關鍵模型設定**：
> - LightGBM: `n_estimators=1000`, `learning_rate=0.03`, `early_stopping=50`
> - 特徵空間: 1,024 維全量特徵 (Without PCA)
> - 類別分佈: 自然真實分佈 (無 RandomUnderSampler)
> - 預計耗時: 約 25 ~ 30 分鐘

---

### 🔹 步驟 3：第二階段雙軌機率校正 (Stage 2 Probability Calibration)
使用 `v1_test1.pkl` (2,000 筆) 訓練 Isotonic Regression / Platt Scaling 1D 機率校正器：

```bash
.venv\Scripts\python.exe pipeline_v1_baseline\step2_calibrate.py --no_pca
```

---

### 🔹 步驟 4：第三與第四階段元評估與圖表繪製 (Meta-Evaluation & Plotting)
使用 `v1_test2.pkl` (3,000 筆) 與獨立評估集 `experiment_results_eval.pkl` 生成最終校正前後之 ECE、ROC-AUC、Brier Score 與安全防護邊界圖表：

```bash
.venv\Scripts\python.exe pipeline_v1_baseline\step3_plot.py --no_pca
.venv\Scripts\python.exe pipeline_v1_baseline\step4_combine_plots.py --no_pca
```

---

## 📁 成果與輸出目錄說明

- **模型檔與訓練對比圖**：`results/v1_baseline/unified_training/lgb_y2_all_models_y2_78k/without_pca/`
- **校正模型與預測快取**：`results/v1_baseline/safety_guardrails_evaluation/without_pca/`
- **報告簡報檔**：[`LLM 隱藏狀態機率校正與元評估框架 8月11日.md`](file:///c:/Users/weiwe/Safety-training-dataset/Safety-training-dataset/LLM%20%E9%9A%B1%E8%97%8F%E7%8B%80%E6%85%8B%E6%A9%9F%E7%8E%87%E6%A0%A1%E6%AD%A3%E8%88%87%E5%85%83%E8%A9%95%E4%BC%B0%E6%A1%86%E6%9E%B6%208%E6%9C%8811%E6%97%A5.md)
