# 機器學習安全防護特徵分析：執行摘要 (Execution Summary)

本執行摘要詳細記錄了利用大型語言模型（LLM）內部特徵（Hidden States）訓練安全探針（Probes）的完整實驗設計、模型表現、優化參數以及深層數學原理。

---

## 1. 實驗目標與背景

為了評估和增強 LLM 在對抗性攻擊與常規場景下的安全表現，本專案基於 **WildJailbreak 資料集**（包含 Vanilla 原始樣本與 Adversarial 對抗性樣本）進行特徵預測與機率校正實驗。我們從 LLM 的 6 個特徵層中提取了輸入序列最後一個 Token 的隱藏狀態（`last_input_hidden_state`）作為模型特徵 $X$，特徵維度為 1,024 維（全量原始特徵，無 PCA 降維）。

實驗包含三個核心分類任務：
1. **$y_1$ 任務 (Model Reply Safety)**：預測模型回覆是否包含 `unsafe` 標籤（Unsafe = 1, Safe = 0）。
2. **$y_2$ 任務 (Prompt Harmfulness)**：預測輸入 Prompt 是否有害（Harmful = 1, Benign = 0）。
3. **$y_3$ 任務 (Consistency Classification)**：預測 LLM 的安全判定是否與輸入 Prompt 的真實有害性一致（Consistent = 1, Inconsistent = 0）。一致性的定義為：
   $$y_3 = \mathbb{I}(y_1 == y_2)$$
   其中 $\mathbb{I}$ 為指示函數（Indicator Function）。當輸入提示詞有害且模型成功攔截為 unsafe，或提示詞無害且模型放行為 safe 時，$y_3$ 為 1，代表判定一致。

---

## 2. 精準化的管線架構 (Evaluation Pipeline)

我們將工作流精簡為統一、高效的管道：

```mermaid
graph TD
    A[LLM Hidden States X_3d: 78k x 6 x 1024] --> B[層特徵提取 X_2d: 78k x 1024]
    B --> C[StandardScaler 特徵標準化 (1024維全量特徵, 無 PCA)]
    C --> D[unified_train.py 訓練 3 大模型 MLP, LGB, LR 在 6 層隱藏狀態之權重]
    D --> E[step2_calibrate.py 統一 Isotonic 機率校正模型]
    E --> F[plot_v1_custom.py 生成 Trends, Reliability 與 Joint Calibration 診斷圖]
```

1. **`unified_train.py` (探針模型訓練)**：在 78,000 筆全量訓練集中完成 3 大模型（MLP, LGB, LR）在 6 層隱藏狀態下的訓練，自動保存 Validation Loss 最低點的最佳權重。
2. **`step2_calibrate.py` (統一條件機率校正)**：在 `test1` 上以統一 Isotonic Regression 擬合標籤 $y_3$，確保校正映射曲線單調遞增，並計算 `test1`、`test2` 與外部對抗評估集 `eval` 之指標與快取。
3. **`plot_v1_custom.py` (自訂視覺化與診斷)**：產出包含 Metrics Trends, Reliability Curves 以及 Joint Calibration（雙 Y 軸直方圖與保序階梯圖）的核心視覺化產物。

---

## 3. 特徵預處理與數據結構

每個 LLM 層所提取的隱藏狀態特徵具有 $1024$ 維的高維度。為了保留最完整的神經元表徵訊息，專案已完全移除 PCA 降維處理：

### A. 標準化 (StandardScaler)
將特徵中心化並縮放至單位變異數。對於特徵矩陣中的每一個特徵分量 $x$，其轉換公式為：
$$\hat{x} = \frac{x - \mu}{\sigma}$$
其中 $\mu$ 是訓練特徵的均值，$\sigma$ 是標準差。

---

## 4. 機率校正與階梯映射渲染優化

### A. 統一保序迴歸 (Isotonic Regression)
保序迴歸的目標是尋找一個非遞減的保序映射函數 $f(S)$，最小化均方誤差（MSE）：
$$\min_{f} \sum_{i=1}^{M} (y_i - f(S_i))^2 \quad \text{subject to } f(S_a) \le f(S_b) \text{ whenever } S_a \le S_b$$
本系統採用 **PAVA (Pool Adjacent Violators Algorithm)** 求解，採統一擬合，解決了先前因分拆 $y_1$ 子群導致校正曲線出現上下鋸齒非單調的結構性問題。

### B. 階梯曲線轉折點遮罩 (Turning Point Mask)
在繪製 Isotonic Regression 階梯曲線時，透過計算 `np.diff(y_step) != 0` 遮罩過濾掉 Y 值不變的冗餘過渡點，僅保留真正發生跳躍的轉折點，解決了數千個微小平移點重疊導致 Matplotlib 產生灰色鋸齒粗塊的視覺渲染瑕疵。

---

## 5. 圖表目錄結構 (`results/v1_baseline/plots_custom/`)

頂層資料夾包含三大核心類別：
* `01_Metrics_Trends`: Brier Score 與 Log Loss 隨層數變化趨勢圖。
* `02_Reliability_Curves`: 校正前後之可靠度對比曲線。
* `07_Joint_Calibration`: 置信度直方圖 (Correct/Incorrect) 與校正階梯曲線雙 Y 軸聯合校正圖。
