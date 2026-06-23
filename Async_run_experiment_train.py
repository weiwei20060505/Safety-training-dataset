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
sem = asyncio.Semaphore(10)

def save_results(results_list, checkpoint=False):
    if not results_list:
        return
    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values(by="id").reset_index(drop=True)
    
    pkl_name = "experiment_results_train.pkl"
    csv_name = "experiment_results_train.csv"
    
    results_df.to_pickle(pkl_name)
    results_df.to_csv(csv_name, index=False, encoding='utf-8-sig')
    
    status = "暫存" if checkpoint else "完成"
    tqdm.write(f"[{status}] 已儲存 {len(results_df)} 筆資料至 {pkl_name} 與 {csv_name}")

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

                return {
                    "id": index,
                    "data_type": row['data_type'],
                    "vanilla": row.get('vanilla', ''),
                    "adversarial": row.get('adversarial', ''),
                    "prompt_source": prompt_source,
                    "prompt": prompt_text,
                    "model_reply": model_reply,
                    "hidden_state": hidden_state
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
                        "hidden_state": []
                    }

                sleep_time = (base_delay ** attempt) + random.uniform(0, 1)
                await asyncio.sleep(sleep_time)

# ================= 3. 主程式進入點 =================
async def main():
    # 解析參數
    parser = argparse.ArgumentParser(description="Async run experiment train")
    parser.add_argument("--all", action="store_true", help="使用全部資料集")
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

    print("正在登入 Hugging Face 並載入 train 資料集...")
    login(token=hf_token)
    dataset = load_dataset("allenai/wildjailbreak", "train", delimiter="\t", keep_default_na=False)
    df = dataset['train'].to_pandas()

    print(f"[成功] 載入 'train' 資料集，共 {len(df)} 筆資料")
    print(f"[欄位] {list(df.columns)}")
    print(f"[資料類型分布]")
    print(df['data_type'].value_counts())

    # 決定使用的資料範圍
    if args.all:
        experiment_df = df.reset_index(drop=True)
        print(f"[選取] 使用全部資料集，共 {len(experiment_df)} 筆資料")
    else:
        n_samples = min(args.n_samples, len(df))
        experiment_df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)
        print(f"[選取] 隨機抽取 {n_samples} 筆資料 (從 {len(df)} 筆中)")

    print(f"\n[開始] 對 {len(experiment_df)} 筆 train 資料進行非同步推論與特徵萃取！")
    print("[預計時間] 取決於網路狀況與資料量")

    rows = experiment_df.to_dict(orient='records')
    tasks = [process_row(idx, row) for idx, row in enumerate(rows)]

    results = []
    # 使用 asyncio.as_completed，並搭配 tqdm 顯示進度
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="推論進度"):
        result = await f
        results.append(result)
        
        # 每 100 筆資料存檔一次
        if len(results) % 100 == 0:
            save_results(results, checkpoint=True)

    # 最終存檔
    save_results(results, checkpoint=False)

    success_count = sum(1 for r in results if r['model_reply'] != 'ERROR' and not r['model_reply'].startswith('ERROR:'))
    print(f"[完成] Train 資料已儲存!")
    print(f"  - experiment_results_train.pkl")
    print(f"  - experiment_results_train.csv")
    print(f"[統計] 成功處理: {success_count} / {len(results)}")

# 啟動非同步事件迴圈
if __name__ == "__main__":
    asyncio.run(main())
