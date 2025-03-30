from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from ..utils.state import State
from langchain_teddynote.models import get_model_name, LLMs

GPT4oMINI = get_model_name(LLMs.GPT4o_MINI)
llm = ChatOpenAI(temperature=0, model=GPT4oMINI)


def create_korean_meaning(state: State):
    """LLM을 호출하여 문서의 패턴을 분석 후 json 템플릿으로 재정의"""

    prompt = PromptTemplate.from_template("""
   You are a translation expert who can translate Korean meanings for English words
    
When you receive the english_vocabulary, please refer to @example_Template and add an empty Korean meaning and Korean pronunciations 

        The final result is

        @example_Template (
  {{
    "word": "",
    "meaning": "",
    "pronunciation": "",
    "phonetic": "",
    "part_of_speech": "",
    "association": "",
    "description": ""
  }}

        Tasks to be performed
        {{
Write empty Korean meanings and Korean pronunciations in the structure of the English word field.        
    (ex:     
    [
    {{
      "word": "perish",
      "meaning": "죽다",
      "pronunciation": "페리쉬",
      "phonetic": "ˈper.ɪʃ",
      "part_of_speech": "v"
    }},
    {{
      "word": "chill",
      "meaning": "냉기, 한기",
      "pronunciation": "칠",
      "phonetic": "tʃɪl",
      "Part of speech": "n",
    }}
  ]
    
    )
        }}

        input english_vocabulary :
        {english_vocabulary}


        Please print only the data in json format for the final response.
        """)

    prompt_llm = prompt | llm

    # LLM 실행
    english_vocabulary = "\n".join([doc.page_content for doc in state["before_vocabulary"]])
    response = prompt_llm.invoke({"english_vocabulary": english_vocabulary})

    return {"has_kor_translated": True, "converted_vocabulary": response.content}
