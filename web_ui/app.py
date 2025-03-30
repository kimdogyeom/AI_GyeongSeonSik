import sys
import os
import tempfile

import asyncio
import streamlit as st

# 현재 파일(app.py)의 위치를 기준으로 src 폴더가 있는 루트를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.main import run_graph  # src.main 가져오기
# 📌 Streamlit UI 구성
st.title("📄 파일 업로드 및 다운로드")

# 📂 파일 업로드
upload_file = st.file_uploader("📂 파일을 업로드하세요", type=["pdf", "txt", "jpg", "jpeg", "png", "xls", "xlsx"])

# ✅ 파일 업로드 여부 확인
if upload_file is not None:
    st.success(f"✅ 업로드된 파일: {upload_file.name}")

    # 임시 파일 생성 (업로드된 파일 저장)
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(upload_file.name)[1]) as temp_file:
        temp_file.write(upload_file.getvalue())
        upload_file_path = temp_file.name

    # 🌀 스피너 추가 (LangGraph에서 처리하는 동안)
    with st.spinner("⏳ 파일을 처리하는 중... 잠시만 기다려 주세요."):
        print(f"🔹 업로드된 파일 경로: {upload_file_path}")
        file_type = upload_file.type.split("/")[-1]
        print(f"🔹 업로드된 파일 타입: {file_type}")

        result_file_path = asyncio.run(run_graph(upload_file_path, file_type))  # ✅ 수정됨


    # 📥 특정 경로의 파일 다운로드 버튼
    with open(result_file_path, "rb") as file:
        st.download_button(
            label="📥 처리된 파일 다운로드",
            data=file,
            file_name="processed_result.txt",
            mime="text/plain",
        )
else:
    st.warning("⚠️ 파일을 업로드해주세요!")  # 업로드하지 않으면 경고 메시지 표시
