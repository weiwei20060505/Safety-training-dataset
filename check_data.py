import os
import sys
import io
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 確保在 Windows 控制台輸出時不會因為 Emoji 或 Unicode 字元產生 cp950 編碼錯誤
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 定義 Tee 類別，用來將輸出同時寫入控制台與 Log 檔案
class Tee:
    def __init__(self, filepath, mode="w", encoding="utf-8"):
        self.file = open(filepath, mode, encoding=encoding)
        self.stdout = sys.stdout

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def close(self):
        self.file.close()

def analyze_prediction(sub_df, group_name):
    total = len(sub_df)
    if total == 0:
        print(f"\n📊 【{group_name}】")
        print("   - 沒有資料 (無樣本)")
        return
        
    correct_count = sub_df['is_correct'].sum()  # True 會被當作 1 計算
    incorrect_count = total - correct_count
    accuracy = (correct_count / total) * 100
    
    print(f"\n📊 【{group_name}】")
    print(f"   - 總筆數: {total}")
    print(f"   - 預測正確數: {correct_count}")
    print(f"   - 預測錯誤數: {incorrect_count}")
    print(f"   - 分類準確度 (Accuracy): {accuracy:.2f}%")
    
    print("   - 混淆矩陣 (Confusion Matrix):")
    # 使用 crosstab 建立列聯表，能清晰看出 True Positive, False Positive 等
    confusion_matrix = pd.crosstab(
        index=sub_df['label'], 
        columns=sub_df['model_reply'], 
        margins=True, 
        margins_name="總計"
    )
    print(confusion_matrix)

def analyze_splits(df, target_col, target_display_name):
    print("\n" + "="*60)
    print(f"--- 基於 {target_display_name} 進行分層分割的 Train/Val/Test1/Test2 分布 ---")
    print("="*60)
    
    # 建立目標變數以進行分層 (stratify)
    if target_col == 'y1':
        y = df['model_reply'].str.lower().str.contains('unsafe').astype(int)
    elif target_col == 'y2':
        y = df['data_type'].str.contains('harmful').astype(int)
    elif target_col == 'y3':
        y1 = df['model_reply'].str.lower().str.contains('unsafe').astype(int)
        y2 = df['data_type'].str.contains('harmful').astype(int)
        y = (y1 == y2).astype(int)
    else:
        print(f"❌ 未知的目標欄位: {target_col}")
        return

    try:
        # 60% Train, 20% Val, 20% Test (Test 再平分為 Test1, Test2 各 10%)
        df_train_val, df_test, y_train_val, y_test = train_test_split(
            df, y, test_size=0.2, random_state=42, stratify=y
        )
        df_train, df_val, _, _ = train_test_split(
            df_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
        )
        df_test1, df_test2 = train_test_split(
            df_test, test_size=0.5, random_state=42
        )
        
        splits = {
            "Train (60%)": df_train,
            "Val (20%)": df_val,
            "Test1 (10%)": df_test1,
            "Test2 (10%)": df_test2
        }
        
        for split_name, split_df in splits.items():
            print(f"\n📌 【{split_name}】 - 總筆數: {len(split_df)}")
            
            # data_type 分布
            if 'data_type' in split_df.columns:
                print("   - data_type 分布:")
                counts = split_df['data_type'].value_counts()
                pcts = split_df['data_type'].value_counts(normalize=True) * 100
                for val in counts.index:
                    print(f"     * {val}: {counts[val]} 筆 ({pcts[val]:.2f}%)")
            
            # model_reply 分布
            if 'model_reply' in split_df.columns:
                print("   - model_reply 分布:")
                counts = split_df['model_reply'].value_counts()
                pcts = split_df['model_reply'].value_counts(normalize=True) * 100
                for val in counts.index:
                    print(f"     * {val}: {counts[val]} 筆 ({pcts[val]:.2f}%)")
                    
            # 該 split 的分類準確度 (LLM 預測與真實標籤)
            if 'data_type' in split_df.columns and 'model_reply' in split_df.columns:
                temp_df = split_df.copy()
                temp_df['label'] = temp_df['data_type'].apply(lambda x: x.split('_')[1] if isinstance(x, str) and '_' in x else 'unknown')
                temp_df['is_correct'] = ((temp_df['label'] == 'harmful') & (temp_df['model_reply'] == 'UNSAFE')) | \
                                       ((temp_df['label'] == 'benign') & (temp_df['model_reply'] == 'SAFE'))
                
                correct_count = temp_df['is_correct'].sum()
                accuracy = (correct_count / len(temp_df)) * 100 if len(temp_df) > 0 else 0
                print(f"   - LLM 分類準確度 (Accuracy): {accuracy:.2f}% (正確: {correct_count}/{len(temp_df)})")
                
    except Exception as e:
        print(f"❌ 進行 {target_display_name} 分割與分析時發生錯誤: {e}")

def check_file(file_name):
    if not os.path.exists(file_name):
        print(f"⚠️ 檔案 {file_name} 不存在，跳過檢查。")
        return

    # 設定 Log 檔案路徑
    log_dir = "results"
    os.makedirs(log_dir, exist_ok=True)
    log_name = f"check_{os.path.splitext(file_name)[0]}.log"
    log_path = os.path.join(log_dir, log_name)

    print(f"\n==========================================")
    print(f"正在檢查檔案: {file_name}")
    print(f"Log 將儲存至: {log_path}")
    print(f"==========================================\n")

    # 重導向 stdout
    tee = Tee(log_path, "w", encoding="utf-8")
    original_stdout = sys.stdout
    sys.stdout = tee

    try:
        # 1. 讀取 Pickle / Joblib 檔案
        print(f"正在讀取實驗資料: {file_name}...")
        try:
            import joblib
            data = joblib.load(file_name)
        except Exception:
            data = pd.read_pickle(file_name)

        def analyze_single_df(df, subtask_name=None):
            prefix = f" [{subtask_name}]" if subtask_name else ""
            
            # 2. 檢查總筆數
            print(f"\n---{prefix} 總共讀取了 {len(df)} 筆資料 ---")

            # 3. 檢查特徵矩陣 X (Hidden State) 的維度
            try:
                num_samples = len(df)
                if num_samples > 0:
                    first_val = df['hidden_state'].iloc[0]
                    if isinstance(first_val, np.ndarray):
                        element_shape = first_val.shape
                    elif isinstance(first_val, list):
                        if len(first_val) > 0 and isinstance(first_val[0], list):
                            element_shape = (len(first_val), len(first_val[0]))
                        else:
                            element_shape = (len(first_val),)
                    else:
                        element_shape = ()
                    
                    full_shape = (num_samples,) + element_shape
                    print(f"\n---{prefix} 特徵矩陣 X 的資訊 ---")
                    print(f"X 的維度 (Shape): {full_shape}")
                    if len(element_shape) == 1:
                        print(f"這代表我們有 {num_samples} 筆樣本，每個樣本有 {element_shape[0]} 個特徵 (維度)。")
                    elif len(element_shape) == 2:
                        print(f"這代表我們有 {num_samples} 筆樣本，每個樣本特徵結構為 {element_shape[0]}x{element_shape[1]} (如：{element_shape[0]} 層 x {element_shape[1]} 維)。")
                    else:
                        print(f"這代表我們有 {num_samples} 筆樣本。")
                else:
                    print(f"\n---{prefix} 特徵矩陣 X 的資訊 ---")
                    print("X 的維度 (Shape): (0,)")
            except Exception as e:
                print(f"\n❌{prefix} 檢查特徵矩陣維度時發生錯誤: {e}")

            # 4. 檢查目標變數 y1 (Data Type) 的分佈
            if 'data_type' in df.columns:
                print(f"\n---{prefix} 目標變數 y1 (Data Type) 的分佈 ---")
                print(df['data_type'].value_counts())
            else:
                print(f"\n⚠️{prefix} 找不到 'data_type' 欄位")

            # 5. 檢查目標變數 y2 (Model Reply) 的分佈
            if 'model_reply' in df.columns:
                print(f"\n---{prefix} 目標變數 y2 (Model Reply) 的分佈 ---")
                print(df['model_reply'].value_counts())
            else:
                print(f"\n⚠️{prefix} 找不到 'model_reply' 欄位")

            # 6. 分析模型預測與真實標籤的對應關係（合併與分組分析）
            if 'data_type' in df.columns and 'model_reply' in df.columns:
                print("\n" + "="*40)
                print(f"---{prefix} 模型預測與真實標籤的對應關係分析 ---")
                print("="*40)

                # 複製以避免對原 dataframe 修改產生 Warning
                df_copy = df.copy()
                df_copy['setting'] = df_copy['data_type'].apply(lambda x: x.split('_')[0] if isinstance(x, str) and '_' in x else 'unknown')
                df_copy['label'] = df_copy['data_type'].apply(lambda x: x.split('_')[1] if isinstance(x, str) and '_' in x else 'unknown')
                df_copy['is_correct'] = ((df_copy['label'] == 'harmful') & (df_copy['model_reply'] == 'UNSAFE')) | \
                                       ((df_copy['label'] == 'benign') & (df_copy['model_reply'] == 'SAFE'))

                analyze_prediction(df_copy, f"{subtask_name or '合併資料'} (Overall)")
                analyze_prediction(df_copy[df_copy['setting'] == 'adversarial'], f"{subtask_name or '合併資料'} (Adversarial 對抗樣本)")
                analyze_prediction(df_copy[df_copy['setting'] == 'vanilla'], f"{subtask_name or '合併資料'} (Vanilla 原始樣本)")
            else:
                print(f"\n⚠️{prefix} 欄位不全，無法進行預測與真實標籤的關係分析。")

            # 7. 基於目標變數分析 Train/Val/Test1/Test2 分割分布
            if "train" in file_name.lower():
                analyze_splits(df, 'y1', '目標變數 y1 (Model Reply Safety)')
                analyze_splits(df, 'y2', '目標變數 y2 (Prompt Harmfulness)')
                analyze_splits(df, 'y3', '目標變數 y3 (Safety Consistency)')
            else:
                print(f"\n📌 提示：{prefix} 本檔案為評估/測試資料，不進行 Train/Val/Test 分割分析。")

        if isinstance(data, dict):
            print(f"此檔案格式為 dict，包含多個子任務資料集：{list(data.keys())}")
            for target_key, sub_df in data.items():
                print(f"\n>>>>>>>> 展開子任務: {target_key} <<<<<<<<")
                analyze_single_df(sub_df, subtask_name=target_key)
        else:
            analyze_single_df(data)

    except Exception as e:
        print(f"\n❌ 處理 {file_name} 時發生未預期錯誤: {e}")
    finally:
        sys.stdout = original_stdout
        tee.close()
        print(f"✨ {file_name} 檢查完成，Log 已存至: {log_path}\n")

if __name__ == "__main__":
    files_to_check = [
       # "experiment_results_train_10000.pkl",
       # "experiment_results_train.pkl",
        #"experiment_results_eval.pkl",
        "augmented_test1.pkl",
        "augmented_test2.pkl",
        "aligned_test1.pkl",
        "aligned_test2.pkl"
    ]
    
    for file_name in files_to_check:
        check_file(file_name)