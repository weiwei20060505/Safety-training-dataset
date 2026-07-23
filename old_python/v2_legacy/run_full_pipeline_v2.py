import subprocess
import sys
import time

def run_script(script_cmd, description):
    print("\n" + "="*70)
    print(f"[v2_20k 完整流程] {description}")
    print("="*70)
    start_time = time.time()
    result = subprocess.run([sys.executable] + script_cmd, check=True)
    elapsed = time.time() - start_time
    print(f"[完成] {description} 完成！耗時: {elapsed:.2f} 秒\n")

def main():
    print("*"*70)
    print("開始執行 v2_20k 完整一鍵評估與繪圖工作流")
    print("*"*70)
    
    # 1. 訓練 Layer 1~6 的 5 種模型 (SGD, MLP, LGB, LR, RF)
    run_script(["unified_train_v2.py", "--layer", "0", "--model", "all", "--target", "all"], "步驟一：全量 6 層全模型訓練與階段一繪圖")
    
    # 2. Step 1: 機率校正與快取寫入 (Baseline + Split 模式)
    run_script(["step1_generate_scores_v2.py"], "步驟二：機率校正與快取運算 (Step 1)")
    
    # 3. Step 2: 可靠度圖表與指標趨勢圖
    run_script(["step2_evaluate_metrics_v2.py"], "步驟三：可靠度圖表與指標趨勢繪製 (Step 2)")
    
    # 4. Step 3: 四象限直方圖產出
    run_script(["step3_plot_histograms_v2.py"], "步驟四：四象限直方圖繪製 (Step 3)")
    
    # 5. Step 4: 真實標籤分群 KDE/Histogram 雙峰圖與詳細 Log 產出
    run_script(["step4_plot_bimodal_kde_v2.py"], "步驟五：雙峰 KDE/直方圖繪製與詳細數值日誌 (Step 4)")
    
    print("v2_20k 完整工作流全數執行完成！所有圖檔與快取皆已妥善存放於 results/v2_20k/。")

if __name__ == "__main__":
    main()
