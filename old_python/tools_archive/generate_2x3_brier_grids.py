import os
from PIL import Image, ImageDraw, ImageFont

base_dir = r'C:\Users\weiwe\OneDrive\Desktop\Safety-training dataset\results\safety_guardrails_evaluation\data_align\split\04_Brier_Components'

targets = ['y1', 'y2', 'y3']
models = ['SGD', 'MLP', 'LGB', 'LR', 'RF']

target_titles = {
    'y1': 'Y1 (Model Reply Safety)',
    'y2': 'Y2 (Prompt Harmfulness)',
    'y3': 'Y3 (Safety Consistency)'
}

sample_p = os.path.join(base_dir, 'y1', 'aligned_test2', 'layer_1', 'dual_y', 'LGB.png')
sample_img = Image.open(sample_p)
single_w, single_h = sample_img.size

top_banner_h = 160
col_header_h = 70
top_margin = top_banner_h + col_header_h

total_w = single_w * 3
total_h = top_margin + single_h * 2

try:
    font_banner = ImageFont.truetype('arial.ttf', 52)
    font_header = ImageFont.truetype('arial.ttf', 44)
except Exception as e:
    print('Font load error:', e)
    font_banner = font_header = ImageFont.load_default()

generated_files = []

for t in targets:
    t_dir = os.path.join(base_dir, t)
    
    for m in models:
        canvas = Image.new('RGB', (total_w, total_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        
        main_title = f'Brier Components (Dual-Axis 2x3 Grid) - Model: {m} | Target: {target_titles[t]} | Dataset: Aligned Test 2'
        bbox = draw.textbbox((0, 0), main_title, font=font_banner)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((total_w // 2 - tw // 2, top_banner_h // 2 - th // 2), main_title, fill=(0, 0, 0), font=font_banner)
        
        for layer_num in range(1, 7):
            r_idx = (layer_num - 1) // 3  # 0 for L1..3, 1 for L4..6
            c_idx = (layer_num - 1) % 3   # 0, 1, 2
            
            img_path = os.path.join(base_dir, t, 'aligned_test2', f'layer_{layer_num}', 'dual_y', f'{m}.png')
            img = Image.open(img_path)
            x_pos = c_idx * single_w
            y_pos = top_margin + r_idx * single_h
            canvas.paste(img, (x_pos, y_pos))
            
        out_path1 = os.path.join(t_dir, f'{m}_aligned_test2_dual_y_2x3_grid.png')
        out_path2 = os.path.join(t_dir, 'aligned_test2', f'{m}_dual_y_2x3_grid.png')
        
        canvas.save(out_path1, quality=95)
        canvas.save(out_path2, quality=95)
        generated_files.append(out_path1)
        print(f'Generated: {out_path1}')

print(f'Total {len(generated_files)} 2x3 Brier dual-axis grid images successfully created.')
