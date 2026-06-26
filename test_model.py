import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
base_url = os.getenv("BASE_URL")
api_key = os.getenv("API_KEY")
model_name = os.getenv("MODEL_NAME")

client = OpenAI(api_key=api_key, base_url=base_url)

try:
    print(f"正在嘗試連線至伺服器: {base_url} ...")
    
    response = client.completions.create(
        model=model_name,
        prompt="Hello",
        max_tokens=50,
        temperature=0.0
    )
    
    print("✨ 連線成功！伺服器正常運作中。")
    
except Exception as e:
    print(f"❌ 連線失敗，請檢查網路或伺服器狀態。")
    print(f"錯誤詳細資訊：\n{e}")
