"""
tools/standardize_image_names.py
================================
1. 清理 04_Brier_Components 中的重複檔名 ({model}.png)
2. 將圖檔全面改為最詳細、含完整參數特徵的詳細描述命名
   例如: LGB_y1_aligned_test2_layer_1_brier_components_dual_y.png
"""

import os
import shutil

base_brier = r"C:\Users\weiwe\OneDrive\Desktop\Safety-training dataset\results\safety_guardrails_evaluation\data_align\split\04_Brier_Components"

targets = ['y1', 'y2', 'y3']
datasets = ['aligned_test1', 'aligned_test2', 'augmented_test2', 'eval']
layers = [f'layer_{i}' for i in range(1, 7)]
axis_modes = ['dual_y', 'single_y']
models = ['SGD', 'MLP', 'LGB', 'LR', 'RF']

def cleanup_brier_components():
    print("="*70)
    print("開始精細重命名與清理 04_Brier_Components 圖檔...")
    print("="*70)
    
    renamed_count = 0
    deleted_count = 0
    
    for t in targets:
        for d in datasets:
            for l in layers:
                layer_num = l.split('_')[1]
                for mode in axis_modes:
                    folder = os.path.join(base_brier, t, d, l, mode)
                    if not os.path.exists(folder):
                        continue
                        
                    for m in models:
                        short_file = os.path.join(folder, f"{m}.png")
                        full_file = os.path.join(folder, f"{m}_brier_components.png")
                        new_descriptive_name = f"{m}_{t}_{d}_layer_{layer_num}_brier_components_{mode}.png"
                        new_file_path = os.path.join(folder, new_descriptive_name)
                        
                        # Remove short duplicate file if exists
                        if os.path.exists(short_file) and os.path.exists(full_file):
                            os.remove(short_file)
                            deleted_count += 1
                            
                        # Rename full file to comprehensive descriptive name
                        target_src = full_file if os.path.exists(full_file) else (short_file if os.path.exists(short_file) else None)
                        if target_src and os.path.exists(target_src):
                            os.rename(target_src, new_file_path)
                            renamed_count += 1
                            
    print(f"清理完畢！已刪除重複簡短檔名 {deleted_count} 個，完成最詳細描述重命名 {renamed_count} 個圖檔。")

if __name__ == '__main__':
    cleanup_brier_components()
