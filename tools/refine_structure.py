"""
refine_structure.py
===================
在 v2_20k_organized 基礎上，對有多個 Y / 多個 Model 的資料夾
再往下加一層分類，對齊 safety_guardrails_evaluation 的風格。

最終結構:
  v2_20k_organized/
    00_Reports/                         ← PDFs
    01_Model_Training/
      01_Model_Comparison/layer_X/      ← 按 layer 再分
        y1/ y2/ y3/
      02_ROC_Curves/layer_X/
        y1/ y2/ y3/
    02_Reliability_Evaluation/
      baseline|split/
        01_Metrics_Trends/              ← 不動（4 張趨勢圖）
        02_Reliability_Diagrams/
          layer_X/
            y1/ y2/ y3/                 ← 每個 Y 各自放 (bars/line/split_y)
        03_Quadrant_Histograms/
          y1|y2|y3/
            test1|test2/
              layer_X/                  ← 最終層，3 個 Y 各一個圖（已是最細）
        04_Brier_Components/
          y1|y2|y3/
            test1|test2/
              layer_X/                  ← brier_bin/brier_rel 各一張
        05_Bimodal_KDE_Histograms/
          test1|test2/
            layer_X/
              y1|y2|y3/                 ← 每個 Y 再按 model 分
                LGB/ LR/ MLP/ RF/ SGD/
    cache/
"""

import os, shutil, re

ROOT = r"results/v2_20k_organized"

# ── 工具 ─────────────────────────────────────────────────────
def mv(src, dst):
    if not os.path.exists(src): return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.move(src, dst)

def re_sort(folder, pattern, key_fn, dry=False):
    """把 folder 內符合 pattern 的檔案依 key_fn(match) 搬到子目錄"""
    if not os.path.exists(folder): return 0
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    n = 0
    for fname in files:
        m = re.search(pattern, fname)
        if not m: continue
        subdir = key_fn(m)
        dst = os.path.join(folder, subdir, fname)
        if not dry:
            mv(os.path.join(folder, fname), dst)
        n += 1
    return n

total = 0

print("="*60)
print("  refine_structure: 加強分層")
print("="*60)

# ═══════════════════════════════════════════════════════════
# 1. 01_Model_Training  →  layer_X / y1|y2|y3
# ═══════════════════════════════════════════════════════════
print("\n[1] 01_Model_Training → layer_X/y1|y2|y3/")
for subfolder in ["01_Model_Comparison", "02_ROC_Curves"]:
    folder = f"{ROOT}/01_Model_Training/{subfolder}"
    if not os.path.exists(folder): continue
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    for fname in files:
        # model_comparison_layer_1_y2.png  /  roc_curve_layer_3_y1.png
        m = re.search(r'layer_(\d+)_(y[123])\.png', fname)
        if not m: continue
        layer_tag = f"layer_{m.group(1)}"
        y_tag     = m.group(2)
        dst = f"{folder}/{layer_tag}/{y_tag}/{fname}"
        mv(f"{folder}/{fname}", dst)
        total += 1
    print(f"  {subfolder}: moved {total} files")

# ═══════════════════════════════════════════════════════════
# 2. 02_Reliability_Diagrams  → layer_X / y1|y2|y3
#    原始: flat 108 files   layer_X_testX_yX_{bars|line|split_y}.png
#    目標: layer_X / y1|y2|y3 / testX / {bars|line|split_y}.png
# ═══════════════════════════════════════════════════════════
print("\n[2] 02_Reliability_Diagrams → layer_X/y1|y2|y3/testX/")
for mode in ["baseline", "split"]:
    folder = f"{ROOT}/02_Reliability_Evaluation/{mode}/02_Reliability_Diagrams"
    if not os.path.exists(folder): continue
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    n = 0
    for fname in files:
        # layer_1_test1_y2_bars.png
        m = re.search(r'(layer_\d+)_(test\d)_(y[123])_(.+)\.png', fname)
        if not m: continue
        layer_tag = m.group(1)
        test_tag  = m.group(2)
        y_tag     = m.group(3)
        dst = f"{folder}/{layer_tag}/{y_tag}/{test_tag}/{fname}"
        mv(f"{folder}/{fname}", dst)
        n += 1; total += 1
    print(f"  [{mode}] Reliability_Diagrams: {n} files")

# ═══════════════════════════════════════════════════════════
# 3. 03_Quadrant_Histograms
#    現狀: y{N}/other/test/layer/  (other 是 re-sort 時失誤)
#    其實已是 y1|y2|y3/test/layer/，只需把 other 修正
#    重新整理: 直接從 flat 的 Reliability_Diagrams 重做
#    (Quadrant files 已在 y/test/layer 下，可能路徑有 other 問題)
# ═══════════════════════════════════════════════════════════
print("\n[3] 03_Quadrant_Histograms: 修正 other/ 問題")
for mode in ["baseline", "split"]:
    base = f"{ROOT}/02_Reliability_Evaluation/{mode}/03_Quadrant_Histograms"
    if not os.path.exists(base): continue
    # 把 other/ 底下的檔案按 y tag 重新分
    other_dir = f"{base}/other"
    if os.path.exists(other_dir):
        for fname in [f for f in
                      [item for _, _, fls in os.walk(other_dir) for item in fls]
                      if f.endswith(".png")]:
            # find in tree
            for dirpath, _, fnames in os.walk(other_dir):
                if fname in fnames:
                    fpath = os.path.join(dirpath, fname)
                    # layer_X_testX_yX_...png
                    m_y  = re.search(r'_(y[123])_', fname)
                    m_t  = re.search(r'layer_\d+_(test\d)', fname)
                    m_l  = re.search(r'(layer_\d+)_', fname)
                    if m_y and m_t and m_l:
                        dst = f"{base}/{m_y.group(1)}/{m_t.group(1)}/{m_l.group(1)}/{fname}"
                        mv(fpath, dst)
                        total += 1
        # remove empty other/
        try: shutil.rmtree(other_dir)
        except: pass
        print(f"  [{mode}] fixed other/ folder")

# ═══════════════════════════════════════════════════════════
# 4. 04_Brier_Components → y1|y2|y3 / test / layer
#    現狀: test/layer/  檔名含 y tag
#    目標: y1|y2|y3/test/layer/
# ═══════════════════════════════════════════════════════════
print("\n[4] 04_Brier_Components → y1|y2|y3/test/layer/")
for mode in ["baseline", "split"]:
    base = f"{ROOT}/02_Reliability_Evaluation/{mode}/04_Brier_Components"
    if not os.path.exists(base): continue
    n = 0
    for dirpath, _, fnames in os.walk(base):
        for fname in fnames:
            if not fname.endswith(".png"): continue
            m_y = re.search(r'_(y[123])_', fname)
            m_t = re.search(r'layer_\d+_(test\d)_', fname)
            m_l = re.search(r'(layer_\d+)_', fname)
            if not (m_y and m_t and m_l): continue
            y_tag    = m_y.group(1)
            test_tag = m_t.group(1)
            layer_tag= m_l.group(1)
            # check not already in right place
            if y_tag in dirpath and test_tag in dirpath: continue
            dst = f"{base}/{y_tag}/{test_tag}/{layer_tag}/{fname}"
            src = os.path.join(dirpath, fname)
            if src != dst:
                mv(src, dst)
                n += 1; total += 1
    # remove empty test/ dirs that are now empty
    for item in os.listdir(base):
        item_path = f"{base}/{item}"
        if os.path.isdir(item_path) and item.startswith("test"):
            # might be empty now
            remaining = list(os.walk(item_path))
            if all(len(f)==0 for _,_,f in remaining):
                shutil.rmtree(item_path)
    print(f"  [{mode}] Brier_Components: {n} files reorganized")

# ═══════════════════════════════════════════════════════════
# 5. 05_Bimodal_KDE_Histograms → test/layer/y1|y2|y3/model/
#    現狀: test/layer/  含 y1_LGB_bimodal_kde.png 等
#    目標: test/layer/y1|y2|y3/model/
# ═══════════════════════════════════════════════════════════
print("\n[5] 05_Bimodal_KDE_Histograms → test/layer/y/model/")
for mode in ["baseline", "split"]:
    base = f"{ROOT}/02_Reliability_Evaluation/{mode}/05_Bimodal_KDE_Histograms"
    if not os.path.exists(base): continue
    n = 0
    for dirpath, _, fnames in os.walk(base):
        for fname in fnames:
            if not fname.endswith(".png"): continue
            # y1_LGB_bimodal_kde.png
            m = re.match(r'(y[123])_([A-Z]+)_bimodal_kde\.png', fname)
            if not m: continue
            y_tag    = m.group(1)
            model    = m.group(2)
            if y_tag in dirpath and model in dirpath: continue
            dst = os.path.join(dirpath, y_tag, model, fname)
            src = os.path.join(dirpath, fname)
            if src != dst:
                mv(src, dst)
                n += 1; total += 1
    print(f"  [{mode}] Bimodal_KDE: {n} files")

# ── 清理空目錄 ────────────────────────────────────────────
print("\n[6] 清理空目錄 ...")
cleaned = 0
for root_dir, dirs, files in os.walk(ROOT, topdown=False):
    if root_dir == ROOT: continue
    try:
        if not os.listdir(root_dir):
            os.rmdir(root_dir)
            cleaned += 1
    except: pass
print(f"  移除 {cleaned} 個空目錄")

print(f"\n{'='*60}")
print(f"  完成！共移動 {total} 個檔案")
print(f"{'='*60}")

# ── 印出最終結構 ──────────────────────────────────────────
print("\n最終資料夾結構:")
for dirpath, dirnames, filenames in os.walk(ROOT):
    depth = dirpath.replace(ROOT, '').count(os.sep)
    if depth > 5: continue
    indent = '  ' * depth
    rel = dirpath.replace(ROOT, '').lstrip(os.sep) or '.'
    n_files = len(filenames)
    n_dirs  = len(dirnames)
    print(f"{indent}[{rel}]  dirs={n_dirs}  files={n_files}")
