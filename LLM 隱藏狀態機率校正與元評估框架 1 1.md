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

### 1. 📅 校正用的「資料集來源」

不論是 y1,y2​ 還是 y3​ 模型，校正器（Isotonic Regression）**只在 `test1`（校正集）上進行 `.fit()` 擬合**：

- **資料對齊組 (`data_align`)**：使用 **`aligned_test1.pkl`** 進行校正擬合。 

---

### 2. 🔹 Baseline 模式：擬合資料與標籤明細

在 `baseline` 模式下，將 `test1` 的全體樣本放在一起，擬合 **單一 Isotonic 校正器 (`iso`)**：

| 模型任務                | 擬合輸入 X (Feature)                                                       | 擬合目標 Y (Target)                 |
| ------------------- | ---------------------------------------------------------------------- | ------------------------------- |
| **y1 模型** (模型回應安全性) | `pre_cal_test1`  <br>`= np.where(y1_test1 == 1, p_test1, 1 - p_test1)` | **`y3_test1`**  <br>(判定一致性真實標籤) |
| **y2 模型** (提示詞有害性)  | `pre_cal_test1`  <br>`= np.where(y1_test1 == 1, p_test1, 1 - p_test1)` | **`y3_test1`**  <br>(判定一致性真實標籤) |
| **y3 模型** (判定一致性)   | `pre_cal_test1`  <br>`= p_test1` (原始 y3 預測機率)                          | **`y3_test1`**  <br>(判定一致性真實標籤) |

---

### 3. ✂️ Split 模式：擬合資料與標籤明細

在 `split` 模式下，**將 `test1` 資料依據 y1​ 的真實標籤（`y1_test1`）拆分為兩組子集**，分別訓練兩個專屬校正器：

#### ① 校正器 1：`iso_0`（專門處理 y1=0，即模型回應 Safe 的樣本）

- **校正資料範圍**：`test1` 中所有 y1=0 的樣本 (`y1_test1 == 0`)。
- **輸入 X**：`pre_cal_test1[y1_test1 == 0]`
- **目標 Y**：**`y3_test1[y1_test1 == 0]`**

#### ② 校正器 2：`iso_1`（專門處理 y1=1，即模型回應 Unsafe 的樣本）

- **校正資料範圍**：`test1` 中所有 y1=1 的樣本 (`y1_test1 == 1`)。
- **輸入 X**：`pre_cal_test1[y1_test1 == 1]`
- **目標 Y**：**`y3_test1[y1_test1 == 1]`**
---

 










