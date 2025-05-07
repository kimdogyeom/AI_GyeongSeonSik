import re

from langchain_openai import ChatOpenAI
from langchain_teddynote.models import get_model_name, LLMs

from analyze_and_format.state import TransformState


GPT4o = get_model_name(LLMs.GPT4o)
llm = ChatOpenAI(temperature=0.4, model=GPT4o)

def generate_parser_code(state : TransformState) -> TransformState:
    structure_description = state["structure_description"]
    sample_text=state["sample_text"]

    prompt = f"""
    다음 설명을 기반으로 파이썬 코드만 작성해주세요.

    당신은 'text'라는 문자열 변수가 이미 정의되어 있다고 가정해야 하며,
    이 변수의 내용을 바꾸거나 재정의하지 마세요. 

    해당 코드는 'text'에 있는 전체 텍스트를 단어 묶음 단위로 나누고,
    각 단어 묶음에서 필요한 정보를 추출하여 Python dict로 변환한 뒤,
    모든 dict를 하나의 리스트로 모아 'result' 변수에 저장합니다.

    요구사항:
    - 'text'는 이미 정의되어 있으며 긴 문자열입니다. 재정의하지 마세요.
    - 각 단어 묶음은 빈 줄 또는 특수 구분자로 나뉘며, 다음과 같은 필드를 포함할 수 있습니다:
      단어, 영어 뜻, 한글 뜻, 품사, 발음기호 등
    - 각 묶음은 dict 형태로 구성하세요.
    - 모든 dict를 하나의 리스트에 담아 result 변수에 저장하세요.
    - result 변수는 반드시 Python 리스트 (List[dict]) 형식이어야 합니다.
    - json.dumps()를 절대 사용하지 마세요.
    - 출력은 파이썬 코드만, 설명 없이 작성하세요.
    

    #### 설명:
    {structure_description}

    #### 구조:
    {sample_text}
    """

    response = llm.invoke(prompt)
    code = response.content.strip()
    if code.startswith("```python"):
        code = re.sub(r"^```python\s*", "", code)
    if code.endswith("```"):
        code = re.sub(r"\s*```$", "", code)

    state["parser_code"] = code
    return state


# 요구사항:
# - 'text' 변수는 이미 정의된 상태이며, 긴 문자열입니다. 재정의하지 마세요.
# - 각 단어 묶음은 빈 줄 또는 특수 구분자로 나뉘며, 다음과 같은 필드를 포함할 수 있습니다:
#   단어, 영어 뜻, 한글 뜻, 품사, 발음기호 등
# - 각 단어 묶음은 하나의 dict로 구성하고, 모든 dict는 result라는 리스트에 저장하세요.
# - result 변수는 반드시 Python 리스트(List[dict]) 형식이어야 합니다.
# - json.dumps()는 절대 사용하지 마세요.
# - 출력은 파이썬 코드만, 설명 없이 작성하세요.
# - 생성된 코드가 JSON 형식으로 유효한 구조를 따르도록 하세요:
#   - 전체는 리스트([])로 감싸고,
#   - 각 객체는 완전한 key-value 쌍으로 구성되어야 하며,
#   - 항목 간 쉼표(,)를 반드시 포함해야 합니다.

