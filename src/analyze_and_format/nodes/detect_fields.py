from langchain_openai import ChatOpenAI
from langchain_teddynote.models import get_model_name, LLMs

from analyze_and_format.state import TransformState


GPT4o = get_model_name(LLMs.GPT4o)
llm = ChatOpenAI(temperature=0.4, model=GPT4o)

def detect_fields(state : TransformState) -> TransformState:
    sample = state["sample_text"]
    prompt = f"""
다음 텍스트는 단어장을 구성하는 일부입니다.
각 단어 묶음에는 어떤 정보가 포함되어 있는지 분석하고,  
다음과 같은 JSON 키 이름 형식으로 알려주세요.

예: ["word", "korean_meaning", "part_of_speech", "pronunciation"]

텍스트:
{sample}
"""
    response = llm.invoke(prompt)

    # JSON 리스트 형태로 응답 받는다고 가정
    try:
        import json
        state["detected_fields"] = json.loads(response.content)
    except:
        state["detected_fields"] = []  # 오류 방지 fallback

    return state
