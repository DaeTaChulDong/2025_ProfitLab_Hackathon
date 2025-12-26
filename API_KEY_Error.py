# @title
# (옵션) 만약 위 코드로 키 인식이 안 될 경우 이 셀을 한 번 실행하세요.
import os
from google.colab import userdata

os.makedirs("/content/.streamlit", exist_ok=True)
with open("/content/.streamlit/secrets.toml", "w") as f:
    f.write(f'OPENAI_API_KEY = "{userdata.get("OPENAI_API_KEY")}"')
print("키 설정 완료!")
