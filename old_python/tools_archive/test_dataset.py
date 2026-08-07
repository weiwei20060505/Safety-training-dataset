from huggingface_hub import login
from datasets import load_dataset
from dotenv import load_dotenv  # 引入環境變數載入工具
import os  # 引入操作系統模組以讀取環境變數
import pandas as pd

# 1. 自動載入 .env 檔案中的所有設定
load_dotenv()

# 2. 從環境變數中讀取設定值
hf_token = os.getenv("HF_TOKEN")
base_url = os.getenv("BASE_URL")
api_key = os.getenv("API_KEY")
model_name = os.getenv("MODEL_NAME")
# 1. 使用你的 Hugging Face Token 進行身分驗證
login(token=hf_token)  # hf_token 是你從 .env 檔案中讀取的 Hugging Face Token

# 2. 載入 WildJailbreak 訓練資料集與評估集
# 分別指定載入 "train" 與 "eval" 設定檔
dataset_train = load_dataset("allenai/wildjailbreak", "train", delimiter="\t", keep_default_na=False)
dataset_eval = load_dataset("allenai/wildjailbreak", "eval", delimiter="\t", keep_default_na=False)

print("【訓練集】\n", dataset_train)
print("="*50)
print("【評估集】\n", dataset_eval)
print("\n✅ 成功載入資料集！")

# 3. 將資料集轉換為 Pandas DataFrame 以便後續處理
# 提示：在 Hugging Face 中，即使你載入了 "eval" 設定檔，它的預設切分鍵值通常還是會叫做 'train'
df_train = dataset_train['train'].to_pandas()
print(df_train.head())   
print("\n✅ 成功將訓練集轉換為 DataFrame！")
print(f"訓練集總共有 {len(df_train)} 筆資料。")
print("\n✅ 訓練集的基本資訊：")
print(df_train.info())

df_eval = dataset_eval['train'].to_pandas()
print(df_eval.head())
print("\n✅ 成功將評估集轉換為 DataFrame！")
print(f"評估集總共有 {len(df_eval)} 筆資料。")
print("\n✅ 評估集的基本資訊：")
print(df_eval.info())
# 1. 計算 'data_type' 欄位中各個類別的總數量
category_counts = df_train['data_type'].value_counts()

# 2. 將統計結果印出來
print("【訓練集 - 各類別資料筆數】")
print(category_counts)

# 3. 順便計算各類別所佔的百分比 (機率分佈)
category_percentages = df_train['data_type'].value_counts(normalize=True) * 100
print("\n【訓練集 - 各類別佔比 (%)】")
print(category_percentages)

