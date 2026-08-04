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
### **目前核心目標** 

- 基於大型語言模型（LLM）內部的隱藏狀態特徵（Hidden States）訓練安全探針（Probes），在已知**LLM回覆(y1)** 的情況，針對性預測**提示詞有害性(y2)** 。
- 經由之前的測試，此次模型選擇之前表現較優的LGB和MLP，並且此次將會著重於3~6層(因為前兩層表現較差)。

---
### 基本定義

1. **$y_1$  (Model Reply Safety)**：LLM回覆是否包含 `unsafe` 標籤（Unsafe = 1, Safe = 0）。
2. **$y_2$  (Prompt Harmfulness)**：輸入 Prompt 是否有害（Harmful = 1, Benign = 0）。
3. **$y_3$  (Consistency Classification)**： LLM 的安全判定是否與輸入 Prompt 的真實有害性一致（Consistent = 1, Inconsistent = 0）。一致性的定義為：
   $$y_3 = \mathbb{I}(y_1 = y_2)$$
   其中 $\mathbb{I}$ 為指示函數（Indicator Function）

---
###  系統架構
**分成兩階段**:
##### **階段一：底層分數預測**

在這個階段，我們的目標是生出具有排序意義的預測分數 $S$，並且加入 $y_1$ 的分流機制：

1. **LightGBM (Tree-based 分支)：**
    取代建立兩個模型，我們會訓練一個 LGB 模型，並利用樹狀模型的特性，強迫樹的根節點（Root Node）或最淺層優先對 $y_1$ 進行切割（例如將 $y_1$ 視為最高權重的類別特徵，或使用強制特徵分裂設定）。這樣一來，資料一進到樹裡，就會根據 $y_1=0$ 或 $y_1=1$ 走入完全不同的決策路徑，產出各自的分數。
    
2. **MLP (神經網路多頭架構 Multi-head Architecture)：**
    
    我們會設計一個 Y 字型的神經網路。前面的隱藏層共用，負責把輸入特徵 $X$ 轉換成高維度的抽象表徵（Latent Representation）；到了最後一層（輸出層），網路直接分叉成兩個獨立的 Heads，一個專門輸出 $y_1=0$ 的分數，另一個專門輸出 $y_1=1$ 的分數。
    
---
##### 階段二：子群機率校準 (Isotonic Regression / PAVA)

不管前面是用 LGB 還是 MLP，只要分數 $S$ 算出來了，我們就進入校準階段：

1. **第一層分流（確保基準一致）：** 把產生的分數依照 $y_1=0$ 和 $y_1=1$ 拆開。
    
2. **第二層分流（文字條件劃分）：** 在各自的 $y_1$ 世界裡，進一步依照指定的「輸入X的種類」切分出不同的獨立子空間。
    
3. **動態區間生成（PAVA）：** 在每一個「文字子空間」內，各自獨立跑 PAVA 演算法。讓演算法自動找出非遞減的區間斷點，把分數 $S$ 壓縮、映射成一階一階的真實條件機率。
**(上述也可改成手動切分區間)**


---

### 📊 目前資料集規模與類別占比摘要

**【全量訓練集】評估集資訊（共 84,999 筆）**：

- **任務 Y1** (模型安全性)：安全 (0) 占 `36.19%`（30,758 筆）/ 不安全 (1) 占 `63.81%`（54,241 筆）
- **任務 Y2** (提示詞有害性)：無害 (0) 占 `50.79%`（43,169 筆）/ 有害 (1) 占 `49.21%`（41,830 筆）
- **任務 Y3** (判定一致性)：不一致 (0) 占 `23.02%`（19,569 筆）/ 一致 (1) 占 `76.98%`（65,430 筆）

**其他類別分布**:
1. **`adversarial_harmful`** (對抗性有害提示詞)：`32.97%`（28,028 筆）
2. **`adversarial_benign`** (對抗性無害提示詞)：`31.20%`（26,521 筆）
3. **`vanilla_benign`** (一般無害提示詞)：`19.59%`（16,648 筆）
4. **`vanilla_harmful`** (一般有害提示詞)：`16.24%`（13,802 筆）
---
###  y1​ 模型安全性 vs. 樣本類型 2×2 交叉統計表

| 樣本類型 (Sample Type) \ y1​ 狀態 | y1=0 (SAFE )                 | y1=1 (UNSAFE )               | **列總計 (Row Total)**           |
| --------------------------- | ---------------------------- | ---------------------------- | ----------------------------- |
| **Adversarial (對抗樣本)**      | **15,860 筆**  <br>           | **38,689 筆**  <br>           | **54,549 筆**  <br>_(64.18%)_  |
| **Vanilla (一般樣本)**          | **14,898 筆**  <br>           | **15,552 筆**  <br>           | **30,450 筆**  <br>_(35.82%)_  |
| **行總計 (Column Total)**      | **30,758 筆**  <br>_(36.19%)_ | **54,241 筆**  <br>_(63.81%)_ | **84,999 筆**  <br>_(100.00%)_ |

---
### 目前問題
1. 輸入X要用甚麼種類做區分
2. 

---
### 可視化診斷工具

待商榷


---










