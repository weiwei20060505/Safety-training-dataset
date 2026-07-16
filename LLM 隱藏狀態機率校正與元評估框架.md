---
marp: true
theme: default
paginate: true
size: 16:9
style: |-
  section {
    font-family: 'Microsoft JhengHei', 'sans-serif';
    font-size: 28px;
    line-height: 1.5;
    padding: 50px 70px;
  }
  h1 { font-size: 48px; color: #1a365d; margin-bottom: 20px; }
  h2 { font-size: 38px; color: #2b6cb0; margin-bottom: 30px; }
  h3 { font-size: 30px; color: #c53030; }
  ul { margin-top: 10px; margin-bottom: 20px; }
  li { margin-bottom: 15px; }
  .highlight { color: #d69e2e; font-weight: bold; }
  .footer { font-size: 16px; color: #718096; position: absolute; bottom: 20px; }
---
# LLM 隱藏狀態機率校正與安全評估框架
## 全管線特徵探針與可視化診斷突破

**報告人：** 馬浩瑋 (國立臺灣師範大學 數學系)
**近期研究進度報告**

---

## 🎯 研究核心背景與挑戰

* **傳統 LLM 安全防護的痛點：**
  * 外部文字審查 API 耗時過長，且容易被暗語或越獄模版繞過。
  * 機器學習探針 (Probes) 在下採樣訓練後，輸出機率**嚴重扭曲失真**。
* **本研究的終極解法：**
  * 讀取 LLM 內部 $1 \sim 6$ 層隱藏狀態 (Hidden States)，建立在線Guardrail。
  * 導入 **V1 保序迴歸 (Isotonic Regression)**，還原真實預測置信度。

---

## 🗺️ 簡報大綱與近期突破總覽

1. **模組一 (run_06)：** 資料分佈對齊策略 (data_align)
2. **模組二 (run_07)：** 校正基準與依 $y_i$ 拆分可靠度圖
3. **模組三 (run_08)：** 四象限直方圖與 Brier Score 嚴格分解
4. **關鍵發現：** Layer 4 表徵可分性飽和點與提前離開機制

---

## 1️⃣ 挑戰一：資料分佈偏移 (Covariate Shift)

* **為什麼標準校正會在上線時失效？**
  * 離線訓練集 (Train) 與外部越獄評估集 (Eval) 的**先驗機率大不相同**。
  * 評估集充滿了未見過的對抗性攻擊，導致貝氏決策邊界漂移。
* **我們的雙軌實驗設計：**
  * **資料擴增 (data_aug)：** 隨機擴充至 10k，維持原有先驗比例。
  * **分佈對齊 (data_align)：** 強制對齊外部集的正例比例，從數學上修正先驗。

---

## 2️⃣ 突破一： 裁判置信度轉換邏輯

* **如何正確評估 Guardrail 的可信度？**
  * 我們不看單純的分類機率，而是量測**「Guardrail判斷正確的信心分數」**。
* **數學轉換邏輯：**
  * **離線研究期 (有 GT)：** 若真為 Safe ($0$) 取 $1-p$；若為 Unsafe ($1$) 取 $p$。
  * **在線部署期 (無 GT)：** 直接假定系統維持安全一致 (代入 $y_3=1$)，輸入衛兵當下的置信度 $\max(p, 1-p)$ 查表。

---

## 3️⃣ 突破二：廢除 ECE，導入 Brier 嚴格分解

* **為什麼教授直言「把 ECE 刪掉」？**
  * ECE (期望校正誤差) 高度受限於等寬 Binning 的切分方式，統計上不夠穩健。
* **改用Brier Score分解公式：**
  $$\text{Brier Score} = \text{Reliability} - \text{Resolution} + \text{Uncertainty}$$
  * **Reliability (可靠度/偏差 ↓)：** 越接近 $0$，代表越準。
  * **Resolution (鑑別度/解析度 ↑)：** 越大越好，代表把 Safe 與 Unsafe 分開的能力。

---

# 📊 核心視覺化診斷成果
## 從整體趨勢到微觀決策群組

---

## 4️⃣ 宏觀趨勢：特徵層數與收斂走勢

![bg right:45% fit](./results/safety_guardrails_evaluation/data_align/01_Metrics_Trends/metrics_comparison_eval.png)

* **跨特徵層 (Layer 1~6) 全指標評估：**
  * 不論哪种模型，折線圖皆呈現完美的**「左上向右下俯衝」**收斂。
  * 證明安全語意是隨深度逐步成形。
* **最佳模型標註 (金色五角星 ⭐)：**
  * **SGD / MLP** 在深層展現了統治級的 Brier Score (低破 $0.108$)。

---

## 5️⃣ 嚴謹檢定：依 $y_i$ 拆分之可靠度對比 (方案 A)

![bg right:43% fit](./results/safety_guardrails_evaluation/data_align/02_Reliability_Curves_split_y/eval/layer_4_reliability.png)

* **為什麼要將 Safe 與 Unsafe 拆開畫？**
  * 破解模型的**「非對稱過度自信」**。
* **看圖重點 (Layer 4 實證)：**
  * **實線 ($y_i=1$，危險組)**：攔截越獄時的校正表現。
  * **虛線 ($y_i=0$，安全組)**：日常放行時的校正表現。
  * 兩條線皆極致貼合 $45^\circ$ 對角線！

---

## 6️⃣ 診斷利器：四象限直方圖與 Brier 解剖

![bg right:43% fit](./results/safety_guardrails_evaluation/data_aug/03_Quadrant_Histograms/eval/layer_4/MLP_layer_4_histogram.png)

* **將預測與真實配對為 4 個象限：**
  * 藍色長條：校正前 (Pre-cal Raw)
  * 橘色長條：校正後 (Post-cal Isotonic)
* **動態標題標註：**
  * 子圖上方直接即時運算顯示該群組的 `Rel(↓)` 與 `Res(↑)`。
  * 徹底透視分類器的底層行為！

---

## 🔍 關鍵診斷 (一)：Group 2 漏網之魚的壓制

![bg right:43% fit](./results/safety_guardrails_evaluation/data_aug/03_Quadrant_Histograms/eval/layer_4/MLP_layer_4_histogram.png)

* **聚焦右上角 Group 2 (偽陰性 FN / 漏網之魚) 🚨：**
  * 越獄攻擊成功騙過了衛兵，導致誤判為 Safe。
* **過度自信的修正：**
  * **校正前 (藍)：** 猜錯了卻還給出 $0.6 \sim 0.8$ 的高把握度。
  * **校正後 (橘)：** PAV 演算法將虛假信心強制**平滑向左壓縮至 0.6 以下**，大幅降低安全風險！

---

## 🔍 關鍵診斷 (二)：Group 4 成功攔截與 PAV 池化

![bg right:43% fit](./results/safety_guardrails_evaluation/data_aug/03_Quadrant_Histograms/eval/layer_4/MLP_layer_4_histogram.png)

* **聚焦右下角 Group 4 (真陽性 TP / 成功攔截) 🛡️：**
  * 衛兵英勇抓到了惡意越獄攻擊。
* **保序迴歸的「池化平滑 (PAV Pooling)」現象：**
  * 原本分散在 $0.8 \sim 1.0$ 的藍色柱子。
  * 被演算法智能池化為一座**高聳的橘色尖峰 ($0.7 \sim 0.75$)**。
  * 可靠度誤差 `Rel` 降至極低的 $0.0349$！

---

## 7️⃣ 論文核心論點：Layer 4 表徵可分性飽和點

* **為什麼不用等到最深層 (Layer 6)？**
  * 實驗證明，LLM 的安全語意在**中段網路 (Layer 4)** 即達到可分性高峰！
  * MLP 於 Layer 4 取得 $AUC = 0.9392$。
* **理論發現：**
  * Layer 6 (輸出映射前) 在越獄對抗集上反而易產生 OOD 數值碰撞與泛化退化。
  * **Layer 4 才是層級特徵的「效能甜蜜點」！**

---

## 8️⃣ 系統實務落地：Layer 4 提前離開機制 (Early Exiting)


```
[輸入提示詞] ---> [LLM L1~L3] ---> [LLM Layer 4] --- (提取 1024維特徵) 
│ 
▼ 
[PCA(128) + MLP 探針]
 │ 
 ┌───────────────────────┴───────────────────────┐ 
▼ (安全置信度 > 0.7)                             ▼ (低信度或有害) 
 [放行：繼續執行 L5~L6]                           [攔截：立即拒答 / 終止推論]
```


* **極致的工程價值：** 為安全防禦系統省下 **$33\%$ 以上** 的推論算力與時間！

---

## 🏆 總結：學術與系統工程貢獻

1. **統計理論實證：** 證實「分佈對齊 (data_align)」在跨域部署的必要性，配合 V1 保序迴歸大幅降低 Brier Score 預測誤差。
2. **診斷工具創新：** 首創 1x5 依類別拆分可靠度圖與四象限置信度直方圖，解決了過度自信與對抗評估的黑箱問題。
3. **落地防護最佳化：** 確立 **Layer 4 + MLP/LightGBM** 為在線安全動態路由的黃金配置，兼具極高攔截率與推論效率！



