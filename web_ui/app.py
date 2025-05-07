import os
import tempfile
import requests
import streamlit as st

backend_address = "http://localhost:5000"

st.title("📄 파일 업로드 및 다운로드")

# 세션 상태 초기화
if "uploaded" not in st.session_state:
    st.session_state.uploaded = False
if "file_path" not in st.session_state:
    st.session_state.file_path = None

# 📂 파일 업로드
upload_file = st.file_uploader("📂 파일을 업로드하세요", type=["pdf", "txt", "jpg", "jpeg", "png", "xls", "xlsx"])

if upload_file is not None and not st.session_state.uploaded:
    st.success(f"✅ 업로드된 파일: {upload_file.name}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(upload_file.name)[1]) as temp_file:
        temp_file.write(upload_file.getvalue())
        upload_file_path = temp_file.name

    with st.spinner("⏳ 처리 중..."):
        file_type = upload_file.type.split("/")[-1]
        response = requests.post(
            backend_address + "/upload",
            json={"upload_file_path": upload_file_path, "upload_file_type": file_type}
        )

    if response.status_code == 200:
        result = response.json()
        st.session_state.uploaded = True
        st.session_state.file_path = result["file_path"]
        st.success("✅ 변환 완료! 아래에서 다운로드하세요.")
    else:
        st.error("❌ 파일 업로드 및 처리 중 오류가 발생했습니다.")
        st.error(response.text)

# ✅ 변환된 파일 다운로드 버튼
if st.session_state.file_path:
    with open(st.session_state.file_path, "rb") as file:
        st.download_button(
            label="📥 변환된 JSON 다운로드",
            data=file,
            file_name="converted_words.json",
            mime="application/json"
        )

    # 🔁 다시 업로드할 수 있도록 리셋 버튼 제공
    if st.button("🔄 다른 파일 업로드하기"):
        st.session_state.uploaded = False
        st.session_state.file_path = None
        st.rerun()

else:
    st.warning("⚠️ 파일을 업로드해주세요!")
