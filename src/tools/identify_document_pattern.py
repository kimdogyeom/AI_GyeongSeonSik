import re

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI

from ..utils.state import State
from langchain_teddynote.models import get_model_name, LLMs

GPT4oMINI = get_model_name(LLMs.GPT4o_MINI)
llm = ChatOpenAI(temperature=0, model=GPT4oMINI)

def contains_korean(text: str) -> bool:
    """한글이 포함되었는지 확인하는 함수"""
    return bool(re.search("[\uac00-\ud7a3]", text))

def identify_document_pattern(state: State):
    """LLM을 호출하여 문서의 패턴을 분석 후 템플릿 재정의"""

    english_vocabulary = "\n".join([doc.page_content for doc in state["before_vocabulary"]])

    prompt = PromptTemplate.from_template("""
    You are an expert who has the ability to understand how it is written in any English word field.

    When you enter english_vocabulary, you can reconfigure it according to the @example_Template below,
    If there is no information to be written in the input english_vocabulary, the information can be written in a blank space, and any information that does not correspond to the writing template should be judged on its own and excluded from the reasoning process so that there is no confusion in the work.

    The final result is

    @example_Template (|word|Korean meaning|Korean pronunciation
|phonetic code|part of speech|)

    Tasks to be performed
    {{
    Identify the structure and pattern of the English word field.
    (ex: able adj. A2 -> |able| | |adj.|)
    }}

    input english_vocabulary :
    {english_vocabulary}


    Generate a final result in markdown format.""")

    prompt_llm = prompt | llm

    # LLM 실행
    response = prompt_llm.invoke({"english_vocabulary":english_vocabulary})
    print("확인")
    print(english_vocabulary)
    print("확인")
    # ✅ 변환된 결과에서 한글 포함 여부 판단
    has_translation = contains_korean(response.content)

    return {
        "gss_converted_vocabulary": response.content,
        "has_kor_translated": has_translation
    }
