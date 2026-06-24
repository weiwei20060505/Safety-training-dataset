import os
from dotenv import load_dotenv
from openai import OpenAI

# 1. 載入 .env 設定
load_dotenv()
base_url = os.getenv("BASE_URL")
api_key = os.getenv("API_KEY")
model_name = os.getenv("MODEL_NAME")

print("=== 測試連線資訊 ===")
print(f"BASE_URL: {base_url}")
print(f"API_KEY: {api_key[:10] if api_key else 'None'}...")
print(f"MODEL_NAME: {model_name}")
print("====================")

# 2. 初始化 OpenAI 用戶端
client = OpenAI(api_key=api_key, base_url=base_url)

# 3. 發送測試請求
try:
    print("正在發送請求至 completions.create...")
    response = client.completions.create(
        model=model_name,
        prompt="Hi, who are you? Please answer in one short sentence.",
        max_tokens=50,
        temperature=0.0,
        timeout=15.0
    )
    
    print("\n[SUCCESS] 成功連線！")
    print("原始 Response 物件:", response)
    
    dumped_data = response.model_dump()
    print("Dumped Data:", dumped_data)
    
    # 嘗試解析回覆
    choices = dumped_data.get('choices', [])
    if choices:
        choice = choices[0]
        # 測試是否是 Chat completion 格式
        if 'message' in choice and choice['message'] and 'content' in choice['message']:
            print("取得回覆 (message/content 格式):", choice['message']['content'])
        # 測試是否是傳統 completion 格式
        elif 'text' in choice:
            print("取得回覆 (text 格式):", choice['text'])
        else:
            print("未知的 choices 格式:", choice)
    else:
        print("未取得 choices 欄位。")
        
except Exception as e:
    print(f"\n[FAIL] 連線或請求失敗: {e}")
