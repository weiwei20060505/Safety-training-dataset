import os
import sys
import argparse
import asyncio
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset
from openai import AsyncOpenAI
from tqdm import tqdm
import random

# ================= 1. 初始化與資料準備 =================
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
base_url = os.getenv("BASE_URL")
api_key = os.getenv("API_KEY")
model_name = os.getenv("MODEL_NAME")

client = AsyncOpenAI(api_key=api_key, base_url=base_url)

# ================= 2. 定義輔助與非同步工作 =================
sem = asyncio.Semaphore(30)

def save_results(new_results_list, existing_results_list, checkpoint=False):
    # 只將新結果中 is_valid 為 True 且 model_reply 為 safe/unsafe 的項目進行存檔
    valid_new = [
        r for r in new_results_list 
        if r.get('is_valid', False) or (r.get('model_reply') and str(r['model_reply']).strip().lower() in ['safe', 'unsafe'])
    ]
    
    # 合併新舊結果，使用 id 作為 key 避免重複
    all_results = {r['id']: r for r in existing_results_list}
    for r in valid_new:
        all_results[r['id']] = r
        
    combined_list = list(all_results.values())
    if not combined_list:
        tqdm.write(f"[{'暫存' if checkpoint else '完成'}] 無有效資料可儲存")
        return
        
    results_df = pd.DataFrame(combined_list)
    # 按 id 排序以維持順序
    results_df = results_df.sort_values(by="id").reset_index(drop=True)
    
    pkl_name = "experiment_results_train.pkl"
    csv_name = "experiment_results_train.csv"
    
    results_df.to_pickle(pkl_name)
    results_df.to_csv(csv_name, index=False, encoding='utf-8-sig')
    
    status = "暫存" if checkpoint else "完成"
    tqdm.write(f"[{status}] 已合併並儲存共 {len(results_df)} 筆有效資料至 {pkl_name} 與 {csv_name}")

async def process_row(index, row):
    # Train 資料的 prompt 選擇邏輯：優先使用 adversarial，否則用 vanilla
    # 這樣能保留原始提示詞資訊
    if pd.notna(row.get('adversarial')) and row['adversarial'] != "":
        prompt_text = row['adversarial']
        prompt_source = "adversarial"
    elif pd.notna(row.get('vanilla')) and row['vanilla'] != "":
        prompt_text = row['vanilla']
        prompt_source = "vanilla"
    else:
        prompt_text = ""
        prompt_source = "empty"

    max_retries = 3
    base_delay = 2.0

    async with sem:
        for attempt in range(max_retries):
            try:
                response = await client.completions.create(
                    model=model_name,
                    prompt=prompt_text,
                    max_tokens=1000,
                    temperature=0.0,
                    timeout=30.0
                )

                dumped_data = response.model_dump()
                model_reply = dumped_data['choices'][0]['message']['content']
                hidden_state = dumped_data.get('kv_transfer_params', {}).get('last_input_hidden_state', [])

                reply_clean = model_reply.strip().lower() if model_reply else ""
                if reply_clean not in ["safe", "unsafe"]:
                    return {
                        "id": index,
                        "data_type": row['data_type'],
                        "vanilla": row.get('vanilla', ''),
                        "adversarial": row.get('adversarial', ''),
                        "prompt_source": prompt_source,
                        "prompt": prompt_text,
                        "model_reply": f"INVALID: {model_reply}",
                        "hidden_state": [],
                        "is_valid": False
                    }

                return {
                    "id": index,
                    "data_type": row['data_type'],
                    "vanilla": row.get('vanilla', ''),
                    "adversarial": row.get('adversarial', ''),
                    "prompt_source": prompt_source,
                    "prompt": prompt_text,
                    "model_reply": model_reply,
                    "hidden_state": hidden_state,
                    "is_valid": True
                }

            except Exception as e:
                if attempt == max_retries - 1:
                    return {
                        "id": index,
                        "data_type": row['data_type'],
                        "vanilla": row.get('vanilla', ''),
                        "adversarial": row.get('adversarial', ''),
                        "prompt_source": prompt_source,
                        "prompt": prompt_text,
                        "model_reply": f"ERROR: {e}",
                        "hidden_state": [],
                        "is_valid": False
                    }

                sleep_time = (base_delay ** attempt) + random.uniform(0, 1)
                await asyncio.sleep(sleep_time)

# ================= 3. 主程式進入點 =================
async def main():
    # 解析參數
    parser = argparse.ArgumentParser(description="Async run experiment train")
    parser.add_argument("--all", action="store_true", help="使用全部未處理資料集")
    parser.add_argument("-n", "--n_samples", type=int, default=2000, help="隨機抽取的筆數 (預設: 2000)")
    
    # 互動式模式判斷：當無任何參數，且為終端機執行時
    if len(sys.argv) == 1 and sys.stdin.isatty():
        try:
            use_all_input = input("是否使用全部資料集？(y/N，預設 N): ").strip().lower()
            use_all = use_all_input in ('y', 'yes')
            n_samples = 2000
            if not use_all:
                n_input = input("請輸入隨機抽取的筆數 (預設 2000): ").strip()
                if n_input:
                    n_samples = int(n_input)
            args = argparse.Namespace(all=use_all, n_samples=n_samples)
        except Exception as e:
            print(f"互動式輸入解析失敗，將使用預設值 (隨機抽取 2000 筆): {e}")
            args = argparse.Namespace(all=False, n_samples=2000)
    else:
        args = parser.parse_args()

    pkl_name = "experiment_results_train.pkl"
    
    # 讀取已處理的 IDs 避免重覆抽樣
    processed_ids = set()
    existing_results = []
    if os.path.exists(pkl_name):
        try:
            existing_df = pd.read_pickle(pkl_name)
            # 排除 model_reply 為空、為 ERROR 或不為 safe/unsafe 的項目，這些需要重跑
            valid_existing = existing_df[
                existing_df['model_reply'].notna() & 
                (existing_df['model_reply'] != '') & 
                (existing_df['model_reply'].str.strip().str.lower().isin(['safe', 'unsafe']))
            ]
            processed_ids = set(valid_existing['id'].tolist())
            existing_results = valid_existing.to_dict(orient='records')
            
            ignored_count = len(existing_df) - len(valid_existing)
            print(f"[讀取] 偵測到已存在的存檔，共 {len(existing_df)} 筆。")
            print(f"  └─ 有效筆數 (safe/unsafe): {len(valid_existing)} 筆 (保留)")
            if ignored_count > 0:
                print(f"  └─ 無效/錯誤筆數: {ignored_count} 筆 (已排除，本次將會重跑)")
        except Exception as e:
            print(f"[警告] 讀取既有 pickle 檔案失敗，將當作全新實驗開始。錯誤資訊: {e}")

    print("正在登入 Hugging Face 並載入 train 資料集...")
    login(token=hf_token)
    dataset = load_dataset("allenai/wildjailbreak", "train", delimiter="\t", keep_default_na=False)
    df = dataset['train'].to_pandas()

    print(f"[成功] 載入 'train' 資料集，共 {len(df)} 筆資料")
    df['original_index'] = df.index  # 紀錄 Hugging Face 的原始 index

    # 排除已成功處理的資料
    if processed_ids:
        filtered_df = df[~df['original_index'].isin(processed_ids)].reset_index(drop=True)
        print(f"[過濾] 排除 {len(processed_ids)} 筆已完成項目，剩餘 {len(filtered_df)} 筆未處理資料")
    else:
        filtered_df = df.reset_index(drop=True)

    if len(filtered_df) == 0:
        print("[提示] 所有資料皆已處理完畢，無需進行推論！")
        return

    # 決定使用的資料範圍
    if args.all:
        experiment_df = filtered_df
        print(f"[選取] 使用所有剩餘未處理資料，共 {len(experiment_df)} 筆資料")
    else:
        n_samples = min(args.n_samples, len(filtered_df))
        # 預設為隨機選取且排除已處理
        experiment_df = filtered_df.sample(n=n_samples, random_state=42).reset_index(drop=True)
        print(f"[選取] 從剩餘未處理資料中隨機抽取 {n_samples} 筆資料 (原總數 {len(df)} 筆)")

    print(f"\n[開始] 對 {len(experiment_df)} 筆 train 資料進行非同步推論與特徵萃取！")
    print("[預計時間] 取決於網路狀況與資料量")

    rows = experiment_df.to_dict(orient='records')
    tasks = [process_row(row['original_index'], row) for row in rows]

    new_results = []
    # 使用 asyncio.as_completed，並搭配 tqdm 顯示進度
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="推論進度"):
        result = await f
        new_results.append(result)
        
        # 每 100 筆資料存檔一次 (與舊資料合併儲存)
        if len(new_results) % 100 == 0:
            save_results(new_results, existing_results, checkpoint=True)

    # 最終存檔
    save_results(new_results, existing_results, checkpoint=False)

    success_count = sum(1 for r in new_results if r.get('is_valid', False))
    invalid_count = sum(1 for r in new_results if not r.get('is_valid', False) and not str(r.get('model_reply', '')).startswith('ERROR:'))
    error_count = sum(1 for r in new_results if str(r.get('model_reply', '')).startswith('ERROR:'))
    
    print(f"[完成] Train 資料已增量儲存!")
    print(f"  - experiment_results_train.pkl")
    print(f"  - experiment_results_train.csv")
    print(f"[統計] 本次處理總數: {len(new_results)}")
    print(f"  - 成功 (safe/unsafe): {success_count} 筆")
    if invalid_count > 0:
         print(f"  - 回覆無效 (非 safe/unsafe): {invalid_count} 筆 (已排除，不予存檔)")
    if error_count > 0:
         print(f"  - API 錯誤: {error_count} 筆 (已排除，不予存檔)")

# 啟動非同步事件迴圈
if __name__ == "__main__":
    asyncio.run(main())
