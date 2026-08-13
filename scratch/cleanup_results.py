import os
import shutil

def cleanup():
    base_dir = r"C:\Users\weiwe\OneDrive\Desktop\Safety-training dataset\results"
    
    # 1. Move top-level conditional_training into v2_framework if needed
    top_cond = os.path.join(base_dir, "conditional_training")
    v2_cond = os.path.join(base_dir, "v2_framework", "conditional_training")
    
    if os.path.exists(top_cond):
        if not os.path.exists(v2_cond):
            print(f"Moving {top_cond} -> {v2_cond}")
            shutil.move(top_cond, v2_cond)
        else:
            print(f"Removing redundant top-level {top_cond}")
            shutil.rmtree(top_cond, ignore_errors=True)
            
    # 2. Remove redundant top-level folders (safety_guardrails_evaluation & unified_training)
    top_safety = os.path.join(base_dir, "safety_guardrails_evaluation")
    top_unified = os.path.join(base_dir, "unified_training")
    
    if os.path.exists(top_safety):
        print(f"Removing redundant top-level {top_safety}")
        shutil.rmtree(top_safety, ignore_errors=True)
        
    if os.path.exists(top_unified):
        print(f"Removing redundant top-level {top_unified}")
        shutil.rmtree(top_unified, ignore_errors=True)
        
    # 3. Remove standalone LGB folders under v1_baseline/unified_training
    v1_unified = os.path.join(base_dir, "v1_baseline", "unified_training")
    lgb_folders_to_delete = [
        "lgb_y2_6000",
        "lgb_y2_10000",
        "lgb_y2_15000",
        "lgb_y2_15000_tuned",
        "lgb_y2_30000_tuned",
        "lgb_y2_78k_ultimate"
    ]
    
    for f in lgb_folders_to_delete:
        target_path = os.path.join(v1_unified, f)
        if os.path.exists(target_path):
            print(f"Deleting standalone LGB folder: {target_path}")
            shutil.rmtree(target_path, ignore_errors=True)
            
    print("\n[OK] Results directory cleanup completed successfully!")

if __name__ == "__main__":
    cleanup()
