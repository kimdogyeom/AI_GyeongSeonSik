from langchain_openai import ChatOpenAI
from langchain_teddynote.models import get_model_name, LLMs

from analyze_and_format.state import TransformState

GPT4o = get_model_name(LLMs.GPT4o)
llm = ChatOpenAI(temperature=0.4, model=GPT4o)

def analyze_structure(state : TransformState) -> TransformState:
    sample = state["sample_text"]

    prompt = f"""
    다음은 영어 단어장 일부 예시입니다.

    이 텍스트의 구조를 파악하고, 다음 항목들을 명확히 분석해주세요:
    1. 각 단어 묶음(청크)는 어떻게 구분되는가? (예: 빈 줄, 개행, 구분자 등)
    2. 한 줄에 포함된 정보는 어떤 필드들로 구성되어 있는가? (예: 단어, 품사, 의미, 레벨 등)
    3. 각 필드는 어떤 순서와 패턴으로 구성되어 있는가? 예: "단어 품사 의미", "단어 의미 품사" 등
    4. 특수한 패턴(예: 쉼표 구분, 마침표 포함 등)이 있다면 설명해주세요.

    텍스트 예시:
    {sample}
    """

    response = llm.invoke(prompt)
    state["structure_description"] = response.content
    return state
