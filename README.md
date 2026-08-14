# LLM 安全防護特徵分析、特徵探針與 post-hoc 機率校正系統

本項目是一個完整的機器學習特徵探針（Representation Probing）與後驗（post-hoc）機率校正框架。項目旨在利用大型語言模型（LLM）內部的激活特徵（Hidden States），在模型尚未輸出文字前判定輸入提示詞（Prompt）是否有害、模型回覆是否安全，以及安全判定是否一致，並導入保序迴歸（Isotonic Regression）機率校正，還原真實的預測信賴度。

---

## 📌 專案目錄結構與批次腳本

專案採用模組化的目錄設計，已將全專案完全重構為 **1,024 維全量特徵 (無 PCA 降維)**，且目錄結構全面消除 `with_pca` / `without_pca` 二分法：

```text
Safety-training-dataset/
├── data/                          # 基準數據與測試集 (pkl, csv)
├── cache/                         # 快取資料夾 (包含校正快取與指標檔)
│   └── v1_baseline/calibration/   # V1 機率校正預測快取與指標 CSV
├── models/                        # 模型權重儲存目錄
│   ├── v1_baseline/unified_training/ # V1 探針模型權重
│   └── v2_framework/framework_training/ # V2 框架模型權重
├── results/                       # 視覺化圖表與實驗紀錄
│   ├── v1_baseline/               # 🔵 V1 Baseline 管線輸出結果
│   │   ├── plots_custom/          # V1 自訂診斷圖 (01, 02, 07 雙軸圖)
│   │   └── plots/                 # V1 標準繪圖產物
│   └── v2_framework/              # 🟢 V2 Framework 管線輸出結果
│       ├── framework_calibration/ # V2 框架校正與快取結果
│       └── plots_framework_stage2/# V2 階段二診斷圖與聯合校正圖
│
├── pipeline_v1_baseline/          # 🔵 V1 Baseline 管線程式碼
│   ├── unified_train.py           # V1 探針模型訓練 (全量 1024 維)
│   ├── step2_calibrate.py         # V1 統一 Isotonic 機率校正
│   ├── plot_v1_custom.py          # V1 自訂診斷圖表繪製 (含轉折點遮罩)
│   ├── step3_plot.py              # V1 標準繪圖
│   └── step4_combine_plots.py     # V1 大圖拼接
│
├── pipeline_v2_framework/         # 🟢 V2 Framework 管線程式碼
│   ├── train_framework_models.py  # V2 16 個模型訓練入口 (全量 1024 維)
│   ├── calibrate_framework_models.py # V2 子群條件校正 (PAVA)
│   ├── conditional_models.py      # V2 條件模型架構 (RootSplit, Y-Head)
│   ├── plot_framework_stage2_v2.py# V2 聯合校正診斷圖
│   └── plot_conditional_roc.py    # V2 條件 ROC 繪圖
│
├── run_full_v1.bat                # ⚡ V1 管線一鍵全自動批次執行檔
├── run_full_v2.bat                # ⚡ V2 管線一鍵全自動批次執行檔
├── README.md                      # 本說明文件
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
    *   特徵維度：$X \in \mathbb{R}^{M \times 6 \times 1024}$ （全量原始特徵空間，已全面移除 PCA 降維）。
3.  **資料預處理流水線 (Pipeline)**：
    *   **標準化 (StandardScaler)**：將特徵縮放為均值為 $0$、變異數為 $1$。
    *   **機率校正 (Isotonic Regression)**：採用 PAVA 演算法針對校正集獨立擬合單調保序映射。

---

## 🚀 快速執行 (Quick Start)

全專案提供兩個統整後的單一批次檔：

### 執行 V1 全套管線
```cmd
.\run_full_v1.bat
```

### 執行 V2 全套管線
```cmd
.\run_full_v2.bat
```
