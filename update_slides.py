import re

file_path = r'LLM 隱藏狀態機率校正與元評估框架 8月20日.md'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

slides_raw = re.split(r'\n---\s*\n', content)
slide_dict = {}
frontmatter_and_title = []

for i, slide in enumerate(slides_raw):
    match = re.search(r'#{1,3}\s+(.*)', slide)
    if 'marp: true' in slide or '# LLM 隱藏狀態機率校正與安全評估框架' in slide:
        frontmatter_and_title.append(slide)
    elif match:
        header = match.group(1).strip()
        slide_dict[header] = slide
    else:
        if slide.strip():
            slide_dict[f'unknown_{i}'] = slide

slide_3_new = '''### 核心研究目標 (Core Research Objectives)

- **內部表徵透視 (Hidden States Probing)**： 
	- 提取 LLM 內部 Transformer 各層隱藏狀態（Hidden States）  \in \mathbb{R}^d$，建構輕量化且高效之白箱安全探針 \theta$。 

- **從預測到校正的兩階段策略 (Prediction & Calibration Pipeline)**：
    - **經驗驅動之任務轉換**：前期測試發現，直接預測模型回覆安全性 ($) 或安全判定一致性 ($) 的效果較差，而預測輸入 Prompt 的真實有害性 ($) 能達到最高準確度。
    - **管線設計**：因此本研究確立了**先預測 $，再校正為 $** 的兩階段框架：
        - 階段一：訓練探針模型預測輸入有害性 $。
        - 階段二：透過保序迴歸 (Isotonic Regression) 將輸出分數校正為 $ (安全判定是否一致) 的真實機率。

- **數學表示與雙版本架構 (V1 vs V2 Evolution)**：
	- **階段一 (分類任務預測 $)**：
        g_\phi(h_L, y_1) \approx P(y_2 = 1 \mid h_L, y_1)
	- **階段二 (機率校正映射為 $)**：
        \hat{p} = \mathcal{C}\Big(g_\phi(h_L, y_1)\Big) \approx P(y_3 = 1 \mid h_L, y_1)
	- **V1 (全域基準探針)**：$\hat{p}_{\text{v1}} = \mathcal{C}\Big(g_\phi(h_L)\Big)$，無視 $ 狀態，建立全域機率校正標準管線。 
	- **V2 (條件分流架構)**：$\hat{p}_{\text{v2}} = \mathcal{C}\Big(g_\phi(h_L,y_1),y_1\Big)$，探索樹狀強迫分支與雙頭神經網路 (YHead-MLP)，在已知 $ 條件下預測 $ 並實現精細化子群機率校準。'''

slide_np_where = '''### 任務轉換核心：從 $ 到 $ 的機率映射

- **任務等價性 ( = \mathbb{I}(y_1 == y_2)$)**：
  當探針預測輸入提示詞有害的機率為 (y_2=1) = p$ 時，我們如何反推安全判定的一致性 ($)？
  
  1. **當 LLM 回覆不安全 (=1$)**：
     判定一致代表提示詞真的有害，因此 (y_3=1) = p$
  2. **當 LLM 回覆安全 (=0$)**：
     判定一致代表提示詞是無害的，因此 (y_3=1) = 1 - p$

- **程式碼實作 (邏輯翻轉)**：
  `python
  pre_cal_score = np.where(y1 == 1, p, 1 - p)
  `
- **核心意義**：
  我們透過這個簡單且無損的數學轉換，完美將 $ 探針的輸出分數轉換為 $ 的先驗分數，接著再送入 Isotonic Regression 擬合真實信賴度。這是串聯第一階段與第二階段的關鍵橋樑。'''

v1_bottleneck_key = [k for k in slide_dict.keys() if 'V1 的瓶頸與洞察' in k][0]
slide_v1_bottleneck = slide_dict[v1_bottleneck_key]
slide_v1_bottleneck_new = slide_v1_bottleneck.replace(
    '當我們將 V1 全域模型按照 **LLM 回覆狀態 (=0$ Safe / =1$ Unsafe)** 拆開繪製各自的可靠性曲線 (Subgroup Reliability Curves) 時，發現了嚴重的**條件校正漂移 (Subgroup Calibration Drift)**：',
    '回顧前面的 
p.where 邏輯，我們將 =0$ 翻轉後的 -p$ 與 =1$ 的 $ **混在同一個池子裡進行全域校正**。\\n\\n如最新繪製的**「$ 分群可靠度診斷圖」**所示，當我們將 V1 全域模型按照 **LLM 回覆狀態 (=0$ Safe / =1$ Unsafe)** 拆開檢視時，發現了嚴重的**條件校正漂移 (Subgroup Calibration Drift)**：'
)
slide_v1_bottleneck_new = slide_v1_bottleneck_new.replace(
    'V1 全域模型假設 (y_2 \mid h_L)$ 單一映射，無視了 (y_2 \mid h_L, y_1=0) \\neq P(y_2 \mid h_L, y_1=1)$ 的條件異質性。',
    'V1 全域模型假設 (y_2 \mid h_L)$ 單一映射，無視了 (y_2 \mid h_L, y_1=0) \\neq P(y_2 \mid h_L, y_1=1)$ 的條件異質性，導致在 PAVA 校正時，兩股方向不同的誤差互相拉扯，引發先驗機率偏差。'
)

ordered_headers = [
    '研究背景：黑箱輸出診斷的侷限性',
    '核心研究目標 (Core Research Objectives)',
    '基本定義',
    '目前資料集規模與類別占比摘要',
    'y1 模型安全性 vs. y2 提示詞有害性 2×2 交叉統計表',
    '實驗流程與二階段篩選機制',
    '數據集劃分 (84,999 筆數據庫+額外)',
    '訓練模型選擇介紹',
    '📌 基準模型 (Baseline Anchor)：Logistic Regression (LR)',
    '任務轉換核心：從 $ 到 $ 的機率映射',
    '第一(訓練)階段模型效能排名總表',
    '各層 AUC 詳細演進 (Layer 1 ~ Layer 6)',
    '第一(訓練)階段階段性成果',
    '核心痛點：模型「過度自信 (Overconfidence)」與「自信度失真」',
    '解決方案：保序機率校準 (Isotonic Calibration)',
    '第二(校正)階段全域機率校正 (Isotonic Calibration) 成果',
    'V1 的瓶頸與洞察：拆解 =0$ 與 =1$ 子群分佈',
    'V2 架構演進：條件分流',
    '系統架構：V2 六大模型策略 (6 Model Architectures)',
    'V2 條件探針模型效能排名總表',
    'V2 子群機率校正 (Subgroup Calibration) 成果突破',
    '🎯 未來工作與總結',
    '附錄 (Appendix)'
]

new_slides = frontmatter_and_title.copy()

for header in ordered_headers:
    if header == '核心研究目標 (Core Research Objectives)':
        new_slides.append(slide_3_new)
    elif header == '任務轉換核心：從 $ 到 $ 的機率映射':
        new_slides.append(slide_np_where)
    elif header == 'V1 的瓶頸與洞察：拆解 =0$ 與 =1$ 子群分佈':
        new_slides.append(slide_v1_bottleneck_new)
    else:
        match_key = None
        for k in slide_dict.keys():
            if k.startswith(header) or header in k:
                match_key = k
                break
        if match_key:
            new_slides.append(slide_dict[match_key])
        else:
            print(f"WARNING: Could not find slide for header: {header}")

appendix_started = False
for slide in slides_raw:
    if '附錄 (Appendix)' in slide:
        appendix_started = True
        continue
    if appendix_started and slide.strip():
        new_slides.append(slide)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write('\\n---\\n'.join(new_slides))
    
print("Successfully updated the presentation file.")
