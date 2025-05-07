import json
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

load_dotenv()

import sys
import os
import tempfile  # ✅ tempfile 추가

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.graph import Graph
from langchain_core.runnables import RunnableConfig
from langchain_teddynote.messages import random_uuid
from langchain_teddynote import logging

logging.langsmith("GSS_TEST")


app = FastAPI()

class UploadRequest(BaseModel):
    upload_file_path: str
    upload_file_type: str
    recursive_limit: int = 30

@app.post("/upload")
async def run_graph(request: UploadRequest):
    config = RunnableConfig(
        recursion_limit=request.recursive_limit, configurable={"thread_id": random_uuid()}
    )
    # ✅ Graph 실행 및 결과 반환
    result = await Graph().ainvoke({"file_path": request.upload_file_path, "file_type": request.upload_file_type}, config=config)

    # ✅ 변환된 텍스트 가져오기
    converted_data = result.get("gss_converted_vocabulary", [])

    # ✅ JSON으로 직렬화하여 임시 파일로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w", encoding="utf-8") as temp_file:
        json.dump(converted_data, temp_file, ensure_ascii=False, indent=2)
        temp_file_path = temp_file.name

    print(f"✅ 변환된 문서가 임시 JSON 파일로 저장됨: {temp_file_path}")

    return {"file_path": temp_file_path}  # 경로를 클라이언트에게 반환

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=False)