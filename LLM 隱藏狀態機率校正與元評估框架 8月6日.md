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
### 系統架構
##### **階段一：底層分數預測 (樹狀模型策略)** 
在這個階段，我們的目標是生出具有排序意義的預測分數 $S$，並且測試不同的 $y_1$ 整合機制： 

1. **LightGBM (Tree-based 強迫分支)：** 取代建立兩個獨立模型，我們訓練一個 LGB 模型，並利用樹狀模型的特性，強迫樹的根節點（Root Node）優先對 $y_1$ 進行切割。這樣一來，資料一進到樹裡，就會根據 $y_1=0$ 或 $y_1=1$ 走入完全不同的決策路徑，產出各自的條件分數。
2. **LightGBM (129維特徵全域探索)：** 另一種對照策略是將 $y_1$ 作為第 129 維度，與 128 維的隱藏狀態特徵 (Hidden States) 合併後一起放入標準 LGB 模型。把決策權交還給演算法，讓模型透過資訊增益 (Information Gain) 自由尋找 $y_1$ 與其他特徵的最佳交互作用點。 
--- 
### 系統架構 
##### **階段一：底層分數預測 (神經網路策略)** 
針對神經網路，我們同樣設計了兩種架構來比較「條件限制」與「資料驅動」的表現:

3. **MLP (神經網路多頭架構 Multi-head Architecture)：** 設計一個 Y 字型的神經網路。前面的隱藏層共用，到了最後一層（輸出層），網路直接分叉成兩個獨立的 Heads，一個專門輸出 $y_1=0$ 的分數，另一個專門輸出 $y_1=1$ 的分數。 
4. **MLP (129維單頭標準架構 Single-head Architecture)：** 採用標準的單一神經網路。直接將 $y_1$ 作為第 129 維輸入，網路的權重矩陣會將 $y_1$ 轉換並與其他 128 維特徵進行線性組合與非線性運算。藉此驗證神經網路是否能自行從全域視角中捕捉 $X$ 與 $y_1$ 的連動關係，不須人工強制分流。
---
##### 階段二：子群機率校準 (Isotonic Regression / PAVA)

不管前面是用 LGB 還是 MLP，只要分數 $S$ 算出來了，我們就進入校準階段：

1. **第一層分流（確保基準一致）：** 把產生的分數依照 $y_1=0$ 和 $y_1=1$ 拆開。
    
2. **第二層分流（文字條件劃分）：** 在各自的 $y_1$ 世界裡，進一步依照指定的「輸入X的種類」切分出不同的獨立子空間。
    
3. **動態區間生成（PAVA）：** 在每一個「文字子空間」內，各自獨立跑 PAVA 演算法。讓演算法自動找出非遞減的區間斷點，把分數 $S$ 壓縮、映射成一階一階的真實條件機率。


---
### PAVA 演算法的完整迭代過程


- **初始化：** 將每一個觀測值視為獨立的區塊 $B_i = \{y_i\}$，且其初始權重 $w_i = 1$。定義每個區塊的均值為 $m_i = y_i$。
    
- **掃描與合併（While 迴圈）：**
    
    由左至右檢查相鄰的區塊 $B_i$ 與 $B_{i+1}$：
    
    如果發現 $m_i > m_{i+1}$（破壞了單調性），則將這兩個區塊合併為一個新區塊 $B_{new} = B_i \cup B_{i+1}$。
    
- **更新區塊均值：**
    
    新區塊的均值 $m_{new}$ 會是原本兩個區塊均值的加權平均：
    
    $$m_{new} = \frac{\vert{}B_i\vert{} \cdot m_i + \vert{}B_{i+1}\vert{} \cdot m_{i+1}}{\vert{}B_i\vert{} + \vert{}B_{i+1}\vert{}}$$
---
- **向後檢查（Backtracking）：**
    
    合併出新區塊後，因為新的均值 $m_{new}$ 可能又會小於它左邊的區塊 $m_{i-1}$，所以必須往回檢查。如果再次發生違規，就繼續將左邊的區塊也合併進來，並重新計算大區塊的平均值。
    
- **終止條件：**
    
    不斷重複上述過程，直到序列中所有的區塊均值都滿足 $m_1 \leq m_2 \leq \dots \leq m_k$。此時，每個資料點對應的最終區塊均值，就是我們要找的全局最佳解 $\hat{p}$。
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
### 數據集使用
##### 階段一:
- **資料來源**：`data/experiment_results_train_10000.pkl`
- **資料集切分**：採用 `60% Train : 20% Val : 20% Test` 的分層抽樣切分：
    - **訓練 (`60% Train`, 6,000 筆)**
    - **畫圖 (`20% val`, 2,000 筆)**
    - **留給階段二 (`20% Test`, 2,000 筆)**
##### 階段二
- **`Test 1` (`test1.pkl`)**：原 2,000 筆測試集(`Test`)的前 1,000 筆擴充至 10,000 筆，用來做保序迴歸。
- **`Test 2` (`test2.pkl`)**：原 2,000 筆測試集(`Test`)的後 1,000 筆擴充至 10,000 筆，用來評估。
- **`Eval` (`experiment_results_eval.pkl`)**：獨立真實評估集 (2,210 筆)，記做`Test3`，用來做最終評估。


---
### 可視化診斷工具

##### **階段一：底層分數預測與探針性能診斷**

為評估 4 種模型策略（`RootSplit-LGBM`, `Feature129-LGBM`, `YHead-MLP`, `SingleHead129-MLP`）在 Layer 3~6 的分類品質，建立兩套核心診斷圖表：

1. **1×4 跨層數 ROC 曲線對比圖 (`roc_curves_1x4_val.png`)**：
   - 橫向排開 Layer 3 ~ 6 在 Validation Set 上的 ROC 曲線與 AUC 數值。
   - **診斷發現**：神經網路策略（`YHead-MLP` 與 `SingleHead129-MLP`）顯著優於樹模型，在 Layer 4 達到最高 AUC **0.9576**。

2. **6 指標模型性能對比條形圖 (`model_comparison_layer_X.png`)**：
   - 針對各層同步呈現 `Accuracy`、`Balanced Acc`、`Precision`、`Recall`、`F1`、`ROC AUC` 柱狀對比，並標註具體數值。
   - **診斷發現**：全量訓練下神經網路策略 Precision 與 Recall 均持穩於 **>90%**，具備優異平衡性與顯著區分度。
---
##### **階段二：子群機率校準 (PAVA) 與可視化診斷**

為評估 PAVA 保序回歸 (Isotonic Regression) 與 $y_1$ 子群分流校準在 3 個獨立測試集（`test1`, `test2`, `eval`）上的校正效果，建立四套獨立診斷圖表：

1. **07_Joint_Calibration 聯合校正圖 (`joint_calibration_layer_X_Model_Dataset.png`) **：
   - **繪製條件與情境**：針對特定測試集、隱藏層 (Layer 3~6) 與模型，展示校準前 Raw Score $S$ 與 PAVA 校準概率 $P_{\text{cal}}$ 的聯合對比。
   - **繪圖流程細節**：上半部繪製分數密度分佈直方圖 (Density Histogram)，呈現 Raw vs Calibrated 機率壓縮；下半部繪製單點映射散點、PAVA 非遞減曲線與 45 度對角線，標註 Brier Score 變化 ($S_{\text{raw}} \rightarrow P_{\text{cal}}$)。

2. **02_Reliability_Curves 1×4 可靠度對比圖 (`reliability_1x4_Dataset.png`)**：
   - **繪製條件與情境**：在獨立測試集上橫向排開 Layer 3~6，繪製 4 種模型經過 PAVA 校準後的 10-bin 可靠度曲線，與 45 度對角理想線對比。
   - **繪圖流程細節**：PAVA 機率校正後，神經網路（`YHead-MLP` 與 `SingleHead129-MLP`）曲線極為貼近 45 度對角線，Brier Score 顯著降至 **0.1158**（Layer 6 Test 2）。
---
3. **05_Brier_Components 條形圖 (`brier_components_Dataset.png`)**：
   - **繪製條件與情境**：將 Brier Score 分解為 **Reliability (可靠度誤差，越低越好)** 與 **Resolution (區分度，越高越好)**，展示校準後的品質。
   - **繪圖流程細節**：1×2 子圖條形對比，展現神經網路保持高 Resolution (~0.20) 的同時，將 Reliability 誤差壓低至 **<0.005**。

4. **06_Step_Mappings 階梯映射條形圖 (`step_mapping_layer_X_Model_Dataset.png`)**：
   - **繪製條件與情境**：呈現 Raw Score 10 個標準區間映射至 PAVA 機率後的數值變化。
   - **繪圖流程細節**：條形圖呈現 PAVA 保序演算法產出的非遞減動態階梯區間與預測機率。









