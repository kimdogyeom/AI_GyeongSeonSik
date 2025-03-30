from dotenv import load_dotenv
load_dotenv()

import sys
import os
import tempfile  # ✅ tempfile 추가

# src 폴더를 Python이 찾을 수 있도록 추가
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from utils.graph import Graph
from langchain_core.runnables import RunnableConfig
from langchain_teddynote.messages import random_uuid
from langchain_teddynote import logging

# 프로젝트 이름을 입력합니다.
logging.langsmith("GSS_TEST")

async def run_graph(upload_file_path, upload_file_type, recursive_limit: int = 30):

    config = RunnableConfig(
        recursion_limit=recursive_limit, configurable={"thread_id": random_uuid()}
    )

    # ✅ Graph 실행 및 결과 반환
    result = await Graph().invoke({"file_path": upload_file_path, "file_type": upload_file_type}, config=config)

    # ✅ 변환된 텍스트 가져오기
    converted_doc = result.get("gss_converted_vocabulary", "")

    # ✅ 임시 파일 생성 및 저장 (자동 .txt 확장자)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as temp_file:
        temp_file.write(converted_doc)
        temp_file_path = temp_file.name  # 임시 파일 경로 저장

    print(f"✅ 변환된 문서가 임시 파일로 저장됨: {temp_file_path}")

    return temp_file_path  # 저장된 파일 경로 반환

if __name__ == "__main__":
    temp_file = run_graph("C:\\Users\\rlaeh\\AppData\\Local\\Temp\\tmp5juax5iz.pdf", "pdf")
