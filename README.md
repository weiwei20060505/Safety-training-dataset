# 安全訓練數據集 (Safety Training Dataset)

一個完整的機器學習框架，用於訓練和評估多個模型，以檢測提示詞的有害性以及模型回應的安全一致性。

## 📋 目錄

- [項目簡介](#項目簡介)
- [快速開始](#快速開始)
- [項目結構](#項目結構)
- [數據格式](#數據格式)
- [使用指南](#使用指南)
- [模型介紹](#模型介紹)
- [評估指標](#評估指標)
- [預測任務](#預測任務)
- [輸出結果](#輸出結果)
- [機率校正與校正曲線](#機率校正與校正曲線)
- [統計陷阱修正說明](#統計陷阱修正說明)
- [進階功能](#進階功能)
- [故障排查](#故障排查)

## 🎯 項目簡介

本項目提供一個統一的機器學習訓練框架，用於：

1. **提示詞安全性檢測（Y1）** - 預測原始提示詞是否有害
2. **一致性檢測（Y3）** - 預測提示詞類型與模型回應的安全性是否一致

框架包含 5 種機器學習模型，在 6 層神經網絡隱藏狀態上進行訓練和評估。

### 核心特性

✅ **5 種機器學習模型**
- SGD（隨機梯度下降）
- LR（邏輯迴歸）
- MLP（多層感知器）
- RF（隨機森林）
- LGB（LightGBM）

✅ **完整的評估指標**
- Accuracy（準確率）
- Precision（精確率）
- Recall（召回率）
- F1 Score
- AUC（ROC 曲線下的面積）

✅ **6 層獨立訓練**
- 針對每層隱藏狀態（768 維）分別訓練模型
- 統一的數據處理流程

✅ **自動化流程**
- 數據加載和預處理
- 特徵標準化
- 模型訓練和評估
- 結果可視化
- 日誌記錄

## 🚀 快速開始

### 前置要求

- Python >= 3.14
- pip 或 uv（推薦）

### 安裝依賴

```bash
# 使用 uv（推薦）
uv sync

# 或使用 pip
pip install -r requirements.txt
```

### 基本用法

#### 步驟 1：生成測試數據

```bash
# 產生基本測試數據 (500筆模擬樣本)
python old_python/generate_test_data.py

# 或產生修正統計陷阱後的改進數據 (2000筆模擬樣本，包含交叉驗證預測)
python old_python/generate_improved_data.py
```

輸出：`test_experiment_results.pkl` 或 `improved_experiment_results.pkl`

#### 步驟 2：執行訓練框架

```bash
python unified_train.py
```

輸出：`results/unified_training/` 目錄

#### 步驟 3：查看結果

檢查 `results/unified_training/` 目錄中的：
- `training_log.txt` - 完整訓練日誌
- `model_comparison_y1.png` - Y1 任務性能對比圖
- `model_comparison_y3.png` - Y3 任務性能對比圖
- `layer_*/` - 各層的模型檔案

## 📁 項目結構

```
.
├── unified_train.py                 # 統一訓練框架（主程序）
├── Isotonic_Regression_corrected.py # 機率校正與校正曲線繪製（使用 Isotonic Regression）
├── uncorrected_graph.py             # 未校正 Calibration Curve 與直方圖繪製
├── Async_run_experiment_train.py    # 異步運行訓練實驗
├── Async_run_experiment_eval.py     # 異步運行評估實驗
├── check_data.py                    # 檢查數據工具
├── test_dataset.py                  # 數據集測試
├── pyproject.toml                   # 項目配置
├── .env.example                     # 環境變數配置模板（請複製為 .env 並自定義，嚴禁將 .env 上傳 Git）
├── .gitignore                       # Git 忽略文件列表
├── old_python/                      # 舊版獨立訓練腳本與數據生成工具
│   ├── generate_test_data.py        # 生成測試數據（500筆模擬數據）
│   ├── generate_improved_data.py    # 交叉驗證數據生成工具（解決數據洩漏與維度災難）
│   ├── train_*.py                   # 各模型舊版獨立訓練腳本
│   └── evaluate_all_models.py       # 舊版模型評估腳本
└── results/                         # 訓練與校正結果輸出目錄
    └── unified_training/
        ├── training_log.txt
        ├── model_comparison_y1.png
        ├── model_comparison_y3.png
        ├── layer_1/
        │   ├── sgd_y1.pkl, ... (訓練好的模型檔案)
        │   ├── calibration_uncorrected_y1.png (未校正 Calibration Curve)
        │   ├── calibration_uncorrected_y3.png
        │   ├── calibration_corrected_y1.png   (Isotonic 校正後的 Calibration Curve)
        │   └── calibration_corrected_y3.png
        ├── ...
        └── layer_6/
```

## 📊 數據格式

### 輸入數據格式

訓練數據應為 pickle 文件（`.pkl`），包含以下結構：

```python
{
    'hidden_states': np.ndarray,  # Shape: (num_samples, num_layers, hidden_dim)
                                   # Example: (500, 6, 768)
    'data_type': list,            # 'harmful' 或 'benign'
    'model_output': list,         # 'unsafe' 或 'safe'
}
```

### 數據統計

- **總樣本數**：通常 500-5000
- **層數**：6 層隱藏狀態
- **隱藏維度**：768 維
- **標籤分布**：harmful vs benign

## 📖 使用指南

### 使用測試數據

```bash
# 1. 生成測試數據（推薦使用改進版本以避免統計陷阱）
python old_python/generate_improved_data.py

# 2. 訓練模型
python unified_train.py

# 3. 查看日誌
cat results/unified_training/training_log.txt
```

### 使用真實數據

將真實數據放在項目目錄中，並在 `unified_train.py` 中進行設定（預設為 `experiment_results_train_1000.pkl` 或 `experiment_results.pkl`）：

```python
# 第 418 行左右
DATA_PATH = "your_real_dataset.pkl"  # 改成你的檔案名
```

然後執行：

```bash
python unified_train.py
```

### 自定義超參數

編輯 `unified_train.py` 中的 `UnifiedModelTrainer` 類，修改各模型的超參數：

```python
class UnifiedModelTrainer:
    def train_sgd(self, X_train, y_train):
        model = SGDClassifier(
            loss='log_loss',
            max_iter=1000,  # 修改此值
            random_state=42
        )
        # ...
```

### 添加新模型

在 `unified_train.py` 中：

1. 實現新的 `train_new_model()` 方法
2. 在 `train_all_layers()` 的 `models_to_train` 列表中添加

```python
models_to_train = [
    ('sgd', self.train_sgd),
    ('lr', self.train_lr),
    # ... 其他模型
    ('new_model', self.train_new_model),  # 添加你的模型
]
```

## 🤖 模型介紹

### SGD（隨機梯度下降）

- **特點**：線性模型，快速訓練
- **適用**：中等規模數據，需要快速推理
- **超參數**：`loss='log_loss'`，`max_iter=1000`

### LR（邏輯迴歸）

- **特點**：線性模型，可解釋性強
- **適用**：特徵線性可分的問題
- **超參數**：`max_iter=1000`，`solver='lbfgs'`

### MLP（多層感知器）

- **特點**：非線性模型，表達能力強
- **適用**：複雜的非線性關係
- **超參數**：`hidden_layer_sizes=(100,)`，`max_iter=1000`

### RF（隨機森林）

- **特點**：集合方法，魯棒性強
- **適用**：特徵交互複雜，容易過擬合的場景
- **超參數**：`n_estimators=100`，`max_depth=10`

### LGB（LightGBM）

- **特點**：梯度提升，速度快，準確率高
- **適用**：大規模數據，需要高準確率
- **超參數**：`n_estimators=100`，`max_depth=5`

## 📊 評估指標說明

### 1. Accuracy（準確率）

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

- **含義**：整體預測正確率
- **範圍**：0 到 1（越接近 1 越好）

### 2. Precision（精確率）

$$\text{Precision} = \frac{TP}{TP + FP}$$

- **含義**：預測為正的樣本中，實際為正的比例
- **使用場景**：誤報代價高的應用

### 3. Recall（召回率）

$$\text{Recall} = \frac{TP}{TP + FN}$$

- **含義**：實際為正的樣本中，被正確預測的比例
- **使用場景**：漏報代價高的應用

### 4. F1 Score

$$F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

- **含義**：Precision 和 Recall 的調和平均數
- **用途**：平衡兩者的綜合指標

### 5. AUC（Area Under Curve）

- **含義**：ROC 曲線下的面積
- **範圍**：0 到 1
- **解釋**：
  - 0.5 = 隨機分類
  - 0.7-0.8 = 中等分類能力
  - 0.8-0.9 = 良好分類能力
  - 0.9+ = 優秀分類能力

## 🎯 預測任務

### Y1 - 提示詞有害性檢測

**目標**：預測原始提示詞是否有害

**標籤來源**：`data_type` 欄位
- `harmful = 1`
- `benign = 0`

**應用**：
- 輸入提示詞安全性檢測
- 有害內容過濾
- 內容審核系統

**示例**：
```
輸入: "How to make a bomb?"
Y1 預測: 1 (harmful)
```

### Y3 - 一致性檢測

**目標**：預測提示詞類型與模型回應的安全性是否一致

**定義**：
$$y3 = (y1 == y2)$$

其中：
- `y1` = 提示詞類型（harmful=1 或 benign=0）
- `y2` = 模型回應（unsafe=1 或 safe=0）

**一致性判斷**：
- ✅ 一致（y3=1）：
  - harmful 提示詞 → unsafe 回應
  - benign 提示詞 → safe 回應
  
- ❌ 不一致（y3=0）：
  - harmful 提示詞 → safe 回應
  - benign 提示詞 → unsafe 回應

**應用**：
- 檢測模型判斷的一致性
- 識別模型漏洞
- 評估安全對齐情況

**示例**：
```
輸入: "How to make a bomb?" (harmful)
模型回應: "I can't help with that." (safe)
Y3 預測: 0 (inconsistent - harmful input but safe output)
```

## 📈 輸出結果

### 目錄結構

```
results/unified_training/
├── training_log.txt           # 完整訓練日誌
├── model_comparison_y1.png    # Y1 性能對比圖
├── model_comparison_y3.png    # Y3 性能對比圖
├── layer_1/
│   ├── sgd_y1.pkl             # 各模型訓練好的 weights
│   ├── sgd_y3.pkl
│   ├── lr_y1.pkl
│   ├── lr_y3.pkl
│   ├── mlp_y1.pkl
│   ├── mlp_y3.pkl
│   ├── rf_y1.pkl
│   ├── rf_y3.pkl
│   ├── lgb_y1.pkl
│   ├── lgb_y3.pkl
│   ├── calibration_uncorrected_y1.png # 未校正 Calibration Curve 與直方圖
│   ├── calibration_uncorrected_y3.png
│   ├── calibration_corrected_y1.png   # Isotonic 校正後的 Calibration Curve
│   └── calibration_corrected_y3.png
├── layer_2/
│   └── ... (同上)
└── ... (layer_3 到 layer_6)
```

### 日誌檔案內容

`training_log.txt` 包含：

```
[配置]
  數據檔案：experiment_results.pkl
  輸出目錄：results/unified_training/

[數據統計]
  總樣本數：500
  Y1 分布：harmful=250, benign=250
  Y2 分布：unsafe=240, safe=260
  Y3 分布：consistent=420, inconsistent=80

[分層訓練]
Layer 1:
  訓練集大小：300（60%）
  驗證集大小：100（20%）
  測試集大小：100（20%）
  
  SGD-Y1:
    Accuracy: 0.850
    Precision: 0.860
    Recall: 0.840
    F1 Score: 0.850
    AUC: 0.920
  
  ... (其他模型的指標)

[生成的圖表]
  模型對比圖已保存至：
  - model_comparison_y1.png
  - model_comparison_y3.png

[完成]
所有 60 個模型已成功訓練和保存
```

### 性能對比圖

生成的 PNG 圖表展示：

- X 軸：5 種模型（SGD, LR, MLP, RF, LGB）
- Y 軸：指標值（0-1）
- 5 條線：5 個評估指標（Accuracy, Precision, Recall, F1, AUC）
- 分開生成：Y1 任務和 Y3 任務

## 📈 機率校正與 Calibration Curve

本框架支援對訓練好的分類模型進行機率校正，以評估模型預測機率與真實正確率的一致性。

### 1. 繪製未校正 Calibration Curve (Reliability Diagram)
未校正的模型預測機率可能存在高估或低估的情況。使用以下指令繪製各層模型在測試集上的未校正校正曲線與預測分數直方圖：
```bash
python uncorrected_graph.py
```
- **輸出圖片**：`results/unified_training/layer_*/calibration_uncorrected_y1.png` 與 `calibration_uncorrected_y3.png`
- **圖表內容**：包含 10 個 Bins 的 Calibration Curve（相較於理想 45 度線）以及預測機率分佈直方圖。

### 2. 使用 Isotonic Regression 進行機率校正 (Calibrated)
若模型預測機率與實際機率有較大偏差，可使用 Isotonic Regression (保序回歸) 對預測機率進行重新校正：
```bash
python Isotonic_Regression_corrected.py
```
- **工作原理**：使用驗證集 (`X_val`, `y_val`) 擬合 `CalibratedClassifierCV` (Isotonic)，並在測試集 (`X_test`, `y_test`) 上評估與繪圖。
- **輸出圖片**：`results/unified_training/layer_*/calibration_corrected_y1.png` 與 `calibration_corrected_y3.png`

## 🛡️ 統計陷阱修正說明

本項目在數據生成與模型訓練上針對以下三個常見的統計陷阱進行了修復與優化（詳細細節參閱 `TRAPS_CORRECTION_SUMMARY.txt`）：

### 1. 維度災難 (Curse of Dimensionality)
- **問題**：若樣本數過少（例如 N=50）而特徵維度過高（d=768 或 1024），模型極易過擬合，導致訓練集 100% 準確但測試集僅有 50% 左右。
- **解決**：在 `old_python/generate_improved_data.py` 中將樣本數擴展至 2000 個，並在 `unified_train.py` 中對特徵進行 `StandardScaler` 標準化，並對 Logistic Regression 與 SGDClassifier 加上強正則化限制 (`C=0.01` 及 `alpha=0.01`)。

### 2. 類別嚴重不平衡 (Severe Class Imbalance)
- **問題**：例如在一致性任務中，不一致 (inconsistent) 的樣本數量遠少於一致樣本。若直接訓練，模型會傾向於盲目預測多數類，導致召回率 (Recall) 為 0。
- **解決**：在 `unified_train.py` 中為所有模型啟用 `class_weight='balanced'` (或對 LightGBM 設置相應平衡參數)，自動提升稀有類別的權重，從而大幅提升 Recall 與 ROC AUC。

### 3. 數據洩漏 (Data Leakage)
- **問題**：如果使用同批資料訓練模型並直接提取其預測結果作為一致性標籤的特徵，模型會因為「背答案」而產生高度虛假的超高準確率，但無法泛化到新數據。
- **解決**：採用 5-Fold 交叉驗證 (`cross_val_predict`) 來產生客觀的預測標籤，確保生成特徵時，模型從未見過該預測目標，杜絕數據洩漏。

## 🔧 進階功能

### 1. 修改特徵標準化方法

```python
# 在 DataSplitter 中修改
from sklearn.preprocessing import StandardScaler, RobustScaler
scaler = RobustScaler()  # 改用 RobustScaler
```

### 2. 自定義資料分割比例

```python
# 在 unified_train.py 中修改
train_size, val_size, test_size = 0.7, 0.15, 0.15  # 改成 70/15/15
```

### 3. 修改模型評估指標

在 `MetricsCalculator.calculate_metrics()` 中添加新指標：

```python
def calculate_metrics(self, y_true, y_pred, y_pred_proba):
    # ... 現有指標
    metrics['custom_metric'] = custom_metric_func(y_true, y_pred)
    return metrics
```

### 4. 自定義圖表樣式

在 `PlotGenerator.plot_model_comparison()` 中修改：

```python
plt.figure(figsize=(12, 8))  # 改變圖表大小
plt.rcParams['font.size'] = 14  # 改變字體大小
```

## 🔍 故障排查

### 問題 1：找不到數據文件

```
FileNotFoundError: [Errno 2] No such file or directory: 'experiment_results.pkl'
```

**解決**：
1. 確認文件在項目根目錄
2. 修改 `unified_train.py` 中的 `DATA_PATH` 變數
3. 或先執行 `python generate_test_data.py` 生成測試數據

### 問題 2：依賴安裝失敗

```
ERROR: Could not find a version that satisfies the requirement
```

**解決**：
```bash
# 更新 pip
pip install --upgrade pip

# 使用 uv 安裝（推薦）
uv sync

# 或重新安裝依賴
pip install -r requirements.txt --no-cache-dir
```

### 問題 3：記憶體不足

**解決**：
1. 減少樣本數：修改 `generate_test_data.py` 中的 `NUM_SAMPLES`
2. 減少模型複雜度：修改超參數（如 `max_iter`）
3. 逐層訓練：修改 `num_layers` 只訓練部分層

### 問題 4：模型訓練非常慢

**優化方法**：
1. 使用 LightGBM 替代隨機森林（更快）
2. 減少模型複雜度（如 `n_estimators`）
3. 檢查機器是否充分利用（CPU/GPU）

### 問題 5：結果無法重現

**確保重現性**：
```python
# 在 unified_train.py 開頭設置隨機種子
import random
import numpy as np
random.seed(42)
np.random.seed(42)
```

## 📝 配置文件

### .env 環境變數

本地運行時需要建立 `.env` 檔案（此檔案已被 `.gitignore` 排除以防止敏感 Token 洩漏到 GitHub）。請參考 `.env.example` 創建並修改內容：

```env
HF_TOKEN="your_huggingface_token_here"
BASE_URL="your_custom_api_base_url_here"
API_KEY="your_custom_api_key_here"
MODEL_NAME="xecguard_experimental"
```

### pyproject.toml

```toml
[project]
name = "safety-training-dataset"
version = "0.1.0"
description = "ML framework for safety training dataset"
requires-python = ">=3.14"

[project.dependencies]
datasets = ">=5.0.0"
lightgbm = ">=4.6.0"
matplotlib = ">=3.11.0"
openai = ">=2.41.1"
python-dotenv = ">=1.2.2"
scikit-learn = ">=1.9.0"
tqdm = ">=4.68.2"
```

## 💡 最佳實踐

### 1. 數據準備

- ✅ 確保數據格式正確（numpy array）
- ✅ 檢查標籤分布（避免嚴重不平衡）
- ✅ 驗證隱藏狀態維度

### 2. 模型訓練

- ✅ 使用統一的隨機種子確保可重現性
- ✅ 監控訓練進度和日誌
- ✅ 在測試集上評估最終性能

### 3. 結果分析

- ✅ 比較不同模型的 ROC 曲線
- ✅ 分析各層的性能差異
- ✅ 檢查混淆矩陣了解常見錯誤類型

### 4. 生產部署

- ✅ 保存最佳模型到版本控制
- ✅ 記錄所有超參數和配置
- ✅ 設置監控和告警機制

## 📚 相關資源

- [scikit-learn 文檔](https://scikit-learn.org/)
- [LightGBM 文檔](https://lightgbm.readthedocs.io/)
- [機器學習評估指標](https://en.wikipedia.org/wiki/Confusion_matrix)

## 📄 許可證

MIT License

## 📧 支持

如有問題，請檢查：
1. `UNIFIED_FRAMEWORK_GUIDE.md` - 詳細框架文檔
2. `training_log.txt` - 訓練日誌
3. 項目的 Issue 頁面

---

**版本**：0.1.0  
**最後更新**：2026-06-23
