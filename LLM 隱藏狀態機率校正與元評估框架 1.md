---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section {
    font-family: 'Microsoft JhengHei', 'sans-serif';
    font-size: 26px;
    line-height: 1.6;
    padding: 45px 65px;
    color: #2d3748;
  }
  h1 { font-size: 44px; color: #1a365d; margin-bottom: 15px; font-weight: 700; }
  h2 { font-size: 36px; color: #2b6cb0; margin-bottom: 25px; font-weight: 700; }
  h3 { font-size: 28px; color: #9b2c2c; font-weight: 600; }
  ul { margin-top: 5px; margin-bottom: 15px; }
  li { margin-bottom: 12px; }
  .highlight { color: #d69e2e; font-weight: bold; }
  .footer { font-size: 14px; color: #718096; position: absolute; bottom: 20px; }
  pre { background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 18px; }
---

# LLM 隱藏狀態機率校正與安全評估框架
## 全管線特徵探針與可視化診斷突破

**報告人：** 馬浩瑋（國立臺灣師範大學 數學系）  
**研究進度與理論架構匯報**

---

## 🎯 研究核心背景與技術挑戰

* **大語言模型安全部署之既有侷限：**
  * 傳統基於生成後之文字匹配與外部 API 審查，推論延遲顯著，且難以抵禦越獄提示詞（Jailbreak Prompts）之對抗性攻擊。
  * 改採內部隱藏狀態（Hidden States）訓練輕量化特徵探針雖能提早預警，惟在經歷重抽樣下採樣（Under-sampling）後，模型預測概率呈現**嚴重之 Sigmoid 數值扭曲**。
* **本研究之核心解決方案：**
  * 萃取 LLM 第 $1 \sim 6$ 層激活特徵，構建非同步前哨防護探針。
  * 導入 **V1 保序迴歸（Isotonic Regression）**，將原始置信度映射為嚴格符合統計物理意義之經驗期望準確率。

---

## 🗺️ 簡報大綱與核心貢獻總覽

1. **模組一（run_06）：** 跨域分佈偏移之先驗對齊策略（data_align）
2. **模組二（run_07）：** V1 置信度轉換基準與類別條件可靠度圖
3. **模組三（run_08）：** 預測混淆矩陣四象限直方圖與 Brier 分解診斷
4. **核心發現：** Layer 4 表徵可分性飽和點與動態路由終止機制

---

## 1️⃣ 挑戰一：跨域對抗集之共變異數偏移

* **標準機率校正模型於實務部署失效之成因：**
  * 校正擬合集（Train/Test1）與外部對抗評估集（Eval）之間存在劇烈的**共變異數偏移（Covariate Shift）**。
  * 對抗集中暗語與越獄模版觸發貝氏決策邊界（Bayesian Decision Boundary）之非線性漂移。
* **雙軌對比實驗設計：**
  * **資料擴增組（data_aug）：** 隨機擴充至 10k 規模，維持原始母體先驗比例。
  * **分佈對齊組（data_align）：** 依循外部對抗集之正負例比例進行嚴格配比，**從數學公式層面等價對齊貝氏先驗概率**。

---

## 2️⃣ 突破一：V1 探針預測置信度轉換架構

* **後驗機率校正之真諦：**
  * 評估核心並非終端類別概率 $P(Y=1)$，而為量測**「前哨探針該次預測結果為真實正確之條件機率」**。
* **嚴謹之特徵轉換數學邏輯：**
  * **離線研究與校正期（持有 GT）：** 針對真實標籤 $y_i^{gt}$，若為 Safe ($0$) 取 $1-p_1$；若為 Unsafe ($1$) 取 $p_1$，精準量測模型對正確答案之估計把握度。
  * **在線實際部署期（無 GT）：** 根據最大後驗決策假定系統維持一致性（代入 $y_3=1$），直接輸入探針當下置信度 $\max(p_1, 1-p_1)$ 查表，完成概率映射之邏輯閉環。

---

## 3️⃣ 突破二：廢除 ECE，導入 Brier Score 嚴格分解

* **期望校正誤差（ECE）之統計侷限：**
  * ECE 對等寬分箱（Uniform Binning）之數量與邊界高度敏感，極易產生統計偏差。
* **採用機率評估之嚴格標準分解公式：**
  $$\text{Brier Score} = \text{Reliability} - \text{Resolution} + \text{Uncertainty}$$
  * **Reliability（可靠度誤差 ↓）：** 預測確率與經驗準確率之均方偏差，數值愈近 $0$ 代表模型無過度自信。
  * **Resolution（特徵鑑別度 ↑）：** 分類器將安全與危險特徵空間有效剝離之解像能力，數值愈大愈優。

---

# 📊 核心可視化診斷與實證成果
## 從宏觀層級收斂到微觀決策群組

---

## 4️⃣ 宏觀趨勢：特徵層數與綜合指標收斂走勢

![bg right:46% fit](./results/safety_guardrails_evaluation/data_align/01_Metrics_Trends/metrics_comparison_eval.png)

* **跨隱藏層（Layer $1 \sim 6$）表徵分析：**
  * 隨網路層數加深，五大機器學習模型之 Brier Score 與 Log Loss 皆呈現單調下降趨勢。
  * 實證了 LLM 對於安全規範與危險語意之表徵，乃基於深層抽象幾何特徵逐漸構建。
* **最優估計模型標註（金色五角星 ⭐）：**
  * **SGD** 與 **MLP** 探針於深層取得最優 Brier Score（收斂至 $0.108$ 以下）。

---

## 5️⃣ 嚴謹檢定：依 $y_i$ 條件拆分之可靠度對比

* **類別條件分佈（Class-Conditional）檢定之必要性：**
  * 旨在破除一般混合可靠度圖對**「非對稱過度自信」**之掩蓋。
* **Layer 4 解耦校正實證：**
  * **實線（$y_i=1$，危險對抗群組）：** 探針於攔截越獄提示詞時之校正穩定度。
  * **虛線（$y_i=0$，常規安全群組）：** 探針於放行無害對話時之信賴水準。
  * 兩條獨立映射曲線皆極致貼合 $45^\circ$ 理想對角線！

---

## 6️⃣ 診斷利器：混淆矩陣四象限直方圖與 Brier 解剖

![bg right:43% fit](./results/safety_guardrails_evaluation/data_aug/03_Quadrant_Histograms/eval/layer_4/MLP_layer_4_histogram.png)

* **基於預測決策 $\hat{y}$ 與母體標籤 $y_1$ 之四象限分割：**
  * 淺藍色分布：校正前之原始置信度（Pre-cal Raw）。
  * 深橘色分布：保序迴歸映射後之真實準確率（Post-cal Isotonic）。
* **動態 Brier 指標分解標註：**
  * 於各子圖上方即時呈現專屬之 `Rel(↓)` 與 `Res(↑)`。
  * 徹底透視特徵空間中過度自信樣本之流向。

---

## 🔍 深度診斷（一）：對抗性偽陰性（Group 2）之概率壓制

![bg right:43% fit](./results/safety_guardrails_evaluation/data_aug/03_Quadrant_Histograms/eval/layer_4/MLP_layer_4_histogram.png)

* **聚焦右上象限 Group 2（偽陰性 FN / 越獄漏報）🚨：**
  * 對抗性暗語成功誘導 LLM 產生違規回覆，探針誤判為 Safe。
* **過度自信之非線性壓制：**
  * **校正前（淺藍）：** 發生誤判仍給予 $0.6 \sim 0.8$ 之虛假高置信度。
  * **校正後（深橘）：** PAV 演算法將虛高分數**強制壓縮至 $0.55 \sim 0.65$ 之低信度區間**，成功消除過度自信膨脹！

---

## 🔍 深度診斷（二）：真陽性攔截（Group 4）與 PAV 池化現象

![bg right:43% fit](./results/safety_guardrails_evaluation/data_aug/03_Quadrant_Histograms/eval/layer_4/MLP_layer_4_histogram.png)

* **聚焦右下象限 Group 4（真陽性 TP / 越獄攔截）🛡️：**
  * 探針成功識別並攔截惡意對抗提示詞。
* **保序迴歸之「數值池化平滑（PAV Pooling）」物理現象：**
  * 校正前分散於 $0.8 \sim 1.0$ 之離散置信度。
  * 經 PAV 演算法單調限制，自動合併為**高度均一之經驗準確率尖峰（$0.7 \sim 0.75$）**。
  * 該象限之可靠度誤差 `Rel` 大幅收斂至 $0.0349$。

---

## 7️⃣ Layer 4 表徵可分性飽和點

* **特徵層級動力學之關鍵發現：**
  * 實證指出，LLM 內部對於危險語意之分類邊界，於**中段網路（Layer 4）即達到非線性可分性之頂峰**。
  * MLP 探針於 Layer 4 取得最高之鑑別度（$AUC = 0.9392$）。
* **深層泛化退化機制剖析：**
  * 最深層（Layer 6）近乎終端文字映射，在對抗集上易引發過度擬合與數值碰撞，導致 $\text{Resolution}$ 下降。
 










