import os
import asyncio
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset
from openai import AsyncOpenAI  # 改用非同步客戶端
from tqdm.asyncio import tqdm  # 改用支援非同步的 tqdm

# ================= 1. 初始化與資料準備 =================
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
base_url = os.getenv("BASE_URL")
api_key = os.getenv("API_KEY")
model_name = os.getenv("MODEL_NAME")

# 設定非同步 API 客戶端
client = AsyncOpenAI(api_key=api_key, base_url=base_url)

print("正在登入 Hugging Face 並載入資料集...")
login(token=hf_token)
dataset = load_dataset("allenai/wildjailbreak", "eval", delimiter="\t", keep_default_na=False)
df = dataset['train'].to_pandas()

print(f"[成功] 載入 'eval' 資料集，共 {len(df)} 筆資料")
print(f"[欄位] {list(df.columns)}")

# 使用 eval 資料集（約2000筆），無需抽樣
experiment_df = df.reset_index(drop=True)

import random 

# ================= 2. 定義單一請求的非同步工作 =================
sem = asyncio.Semaphore(30)

async def process_row(index, row):
    # 決定要用的 Prompt（eval 資料集只有 adversarial 字段，無 vanilla）
    prompt_text = row['adversarial'] if pd.notna(row['adversarial']) and row['adversarial'] != "" else ""
    prompt_source = "adversarial" if prompt_text else "empty"
    
    # 💡 修正 2：設定重試機制參數
    max_retries = 3      # 最多重試 3 次
    base_delay = 2.0     # 基礎等待秒數
    
    async with sem:
        for attempt in range(max_retries):
            try:
                response = await client.completions.create(
                    model=model_name,
                    prompt=prompt_text,
                    max_tokens=1000,
                    temperature=0.0,
                    timeout=30.0  # 💡 修正 3：加入超時設定，避免 API 卡死
                )
                
                dumped_data = response.model_dump()
                model_reply = dumped_data['choices'][0]['message']['content']
                hidden_state = dumped_data.get('kv_transfer_params', {}).get('last_input_hidden_state', [])
                
                return {
                    "id": index,
                    "data_type": row['data_type'],
                    "prompt_source": prompt_source,
                    "prompt": prompt_text,
                    "model_reply": model_reply,
                    "hidden_state": hidden_state
                }
                
            except Exception as e:
                # 💡 修正 4：如果已經是最後一次重試，才真正宣告失敗
                if attempt == max_retries - 1:
                    return {
                        "id": index,
                        "data_type": row['data_type'],
                        "prompt_source": prompt_source,
                        "prompt": prompt_text,
                        "model_reply": f"ERROR: {e}",
                        "hidden_state": []
                    }
                
                # 💡 修正 5：指數退避 (Exponential Backoff) 與隨機抖動 (Jitter)
                sleep_time = (base_delay ** attempt) + random.uniform(0, 1)
                await asyncio.sleep(sleep_time)

# ================= 3. 主程式進入點 =================
async def main():
    print(f"\n[開始] 對 {len(experiment_df)} 筆資料進行非同步推論與特徵萃取！")

    # 優化：使用 to_dict(orient='records') 替代 iterrows()，效率提升 ~8-10 倍
    rows = experiment_df.to_dict(orient='records')
    tasks = [process_row(idx, row) for idx, row in enumerate(rows)]
    
    # 使用 tqdm.gather 同時執行所有任務，並顯示漂亮的非同步進度條
    results = await tqdm.gather(*tasks)
    
    # ================= 4. 資料存檔 =================
    print("\n正在將收集到的特徵與結果存檔...")
    results_df = pd.DataFrame(results)

    # 排序確保 ID 順序正確（非同步完成順序可能交錯，但 gather 會幫我們對齊，保險起見可按 id 排序）
    results_df = results_df.sort_values(by="id").reset_index(drop=True)

    # 存成 Pickle 格式
    results_df.to_pickle("experiment_results.pkl")
    # 同時存一份 CSV
    results_df.to_csv("experiment_results.csv", index=False, encoding='utf-8-sig')

    print("實驗完成！特徵資料已儲存為 experiment_results.pkl")

# 啟動非同步事件迴圈
if __name__ == "__main__":
    asyncio.run(main())