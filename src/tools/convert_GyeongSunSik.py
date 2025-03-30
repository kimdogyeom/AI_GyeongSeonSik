import asyncio
import re
import json
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from typing import List, Dict

from ..utils.state import State
from langchain_teddynote.models import get_model_name, LLMs

GPT4o = get_model_name(LLMs.GPT4o)
llm = ChatOpenAI(temperature=1, model=GPT4o)

async def convert_GyeongSunSik_chunk(chunk: str):
    prompt = PromptTemplate.from_template("""
    당신은 "경선식 영단어"를 이해하고 학습자에게 단어를 설명하거나 연상법을 만들어주는 AI입니다. 이 방법은 한국의 영어 강사 경선식이 개발한 것으로, 연상법과 해마학습법을 활용해 단어를 빠르고 오래 기억하게 합니다. 아래 지침을 따라 작동하세요.

#### 경선식 영단어의 핵심 원리:
1. ##중요## **연상법**: 단어의 뜻을 비슷한 발음이나 유머러스한 이야기로 연결해 기억을 돕습니다.
예:
- "senior" (선배) → "선배는 신이여!"
- "hope" (희망) → "호프집에서 희망을 찾다"
#중요## - 연상은 한국어 발음이나 문화적 맥락을 활용할 수 있습니다.

2. **특징**:
- 재미있고 과감한 표현 가능 (저속한 표현은 피함).
- 학습자가 단어와 연상을 쉽게 떠올리도록 단순하고 강렬하게.

#### 작업 지침:
1. 단어를 입력받으면, 뜻과 발음을 확인한 뒤 경선식 스타일의 연상법을 만들어 출력하세요. 입출력 예시는 아래 ####예시 항목을 참고하세요.

2. **주의사항**:
- 연상은 한국어 기반으로, 발음이나 뜻을 자연스럽게 연결.
- 지나치게 복잡하거나 억지스럽지 않게, 학습자가 쉽게 이해할 수 있도록.
- 단어의 뜻을 왜곡하지 마세요.

[
  {{
    "word": "",
    "meaning": "",
    "pronunciation": "",
    "phonetic": "",
    "part_of_speech": "",
    "association": "",
    "description": ""
  }}
]

word : 영어 단어
meaning : 한국어 뜻
pronunciation : 한국어 발음 (연상 기반)
phonetic : 발음 기호
part_of_speech : 품사
association	: 경선식식 연상 문장
description	: 해당 문장을 시각적으로 표현한 이미지 프롬프트

#### 예시:
  "input": [
    {{
      "word": "perish",
      "meaning": "죽다",
      "pronunciation": "페리쉬",
      "phonetic": "ˈper.ɪʃ",
      "part_of_speech": "v"
    }},
    {{
      "word": "eager",
      "meaning": "간절히 바라는",
      "pronunciation": "이거",
      "phonetic": "",
      "part_of_speech": ""
    }},
    {{
      "word": "cohesion",
      "meaning": "결합, 화합",
      "pronunciation": "코헤션",
      "phonetic": "n",
      "part_of_speech": ""
    }},
    {{
      "word": "chill",
      "meaning": "냉기, 한기",
      "pronunciation": "칠",
      "phonetic": "n",
      "part_of_speech": ""
    }}
  ],
  "output": [
    {{
      "word": "perish",
      "meaning": "죽다",
      "pronunciation": "페리쉬",
      "phonetic": "ˈper.ɪʃ",
      "Part of speech": "",
      "association": "[파리 쉬~]",
      "description": "A lifeless corpse lying on the ground with flies buzzing around it, symbolizing death."
    }},
    {{
      "word": "eager",
      "meaning": "간절히 바라는",
      "pronunciation": "이거",
      "phonetic": "",
      "Part of speech": "",
      "association": "[이거] 사주세요.",
      "description": "A small child pointing at a teddy bear in a toy store, clearly begging a parent to buy it."
    }},
    {{
      "word": "cohesion",
      "meaning": "결합, 화합",
      "pronunciation": "코헤션",
      "phonetic": "",
      "Part of speech": "",
      "association": "[코]를 맞대고 입을 결합 [헤션]",
      "description": "A man and woman gently touching noses and kissing, expressing bonding."
    }},
    {{
      "word": "chill",
      "meaning": "냉기, 한기",
      "pronunciation": "칠",
      "phonetic": "tʃɪl",
      "Part of speech": "",
      "association": "[칠]도의 기온으로 떨어져 쌀쌀한 날씨.",
      "description": "A person shivering from the cold beside a thermometer showing 7°C."
    }}
  ]
#### 이제 예시를 참고해서 출력물을 만들어주면 됩니다.
{input_document}
    """)
    
    prompt_llm = prompt | llm
    try:
        response = await prompt_llm.ainvoke({"input_document": chunk})
        # LLM이 JSON 형식으로 응답하도록 수정해야 함.
        return json.loads(str(response.content))  # JSON으로 파싱
    except (json.JSONDecodeError, Exception) as e:
        print(f"Error in LLM response or JSON decoding: {e}, Response: {response.content}")
        return [] # 에러 발생 시 빈 리스트 반환


async def convert_GyeongSunSik_async(state: State):
    vocabulary_str = state["converted_vocabulary"]

    # 정규 표현식을 사용하여 데이터 추출
    pattern = r'\{\{\s*("word":\s*"(.*?)",\s*"meaning":\s*"(.*?)",\s*"pronunciation":\s*"(.*?)",\s*"phonetic":\s*"(.*?)",\s*"part_of_speech":\s*"(.*?)")\s*\}\}'
    matches = re.findall(pattern, vocabulary_str)

    vocabulary = []
    for match in matches:
        vocabulary.append({
            "word": match[1],
            "meaning": match[2],
            "pronunciation": match[3],
            "phonetic": match[4],
            "part_of_speech": match[5]
        })


    chunk_size = 10 # 덩어리 크기 조정 (4개의 단어씩 처리)
    chunks = [vocabulary[i:i + chunk_size] for i in range(0, len(vocabulary), chunk_size)]

    semaphore = asyncio.Semaphore(7)

    async def limited_chunk(chunk):
        async with semaphore:
            return await convert_GyeongSunSik_chunk(chunk)

    results = await asyncio.gather(*[limited_chunk(chunk) for chunk in chunks])

    merged = []
    for result in results:
        merged.extend(result)

    return {"gss_converted_vocabulary": merged}

async def convert_GyeongSunSik(state: State):
    result = await convert_GyeongSunSik_async(state)
    return result  
  
# ----------------- Original code -----------------

# from langchain_core.prompts import PromptTemplate
# from langchain_openai import ChatOpenAI

# from ..utils.state import State
# from langchain_teddynote.models import get_model_name, LLMs

# GPT4o = get_model_name(LLMs.GPT4o)
# llm = ChatOpenAI(temperature=1, model=GPT4o)

# def convert_GyeongSunSik(state: State):
#     prompt = PromptTemplate.from_template("""
# 당신은 "경선식 영단어"를 이해하고 학습자에게 단어를 설명하거나 연상법을 만들어주는 AI입니다. 이 방법은 한국의 영어 강사 경선식이 개발한 것으로, 연상법과 해마학습법을 활용해 단어를 빠르고 오래 기억하게 합니다. 아래 지침을 따라 작동하세요.

# #### 경선식 영단어의 핵심 원리:
# 1. ##중요## **연상법**: 단어의 뜻을 비슷한 발음이나 유머러스한 이야기로 연결해 기억을 돕습니다.
# 예:
# - "senior" (선배) → "선배는 신이여!"
# - "hope" (희망) → "호프집에서 희망을 찾다"
# #중요## - 연상은 한국어 발음이나 문화적 맥락을 활용할 수 있습니다.

# 2. **특징**:
# - 재미있고 과감한 표현 가능 (저속한 표현은 피함).
# - 학습자가 단어와 연상을 쉽게 떠올리도록 단순하고 강렬하게.

# #### 작업 지침:
# 1. 단어를 입력받으면, 뜻과 발음을 확인한 뒤 경선식 스타일의 연상법을 만들어 출력하세요. 입출력 예시는 아래 ####예시 항목을 참고하세요.

# 2. **주의사항**:
# - 연상은 한국어 기반으로, 발음이나 뜻을 자연스럽게 연결.
# - 지나치게 복잡하거나 억지스럽지 않게, 학습자가 쉽게 이해할 수 있도록.
# - 단어의 뜻을 왜곡하지 마세요.

# [
#   {{
#     "word": "",
#     "meaning": "",
#     "pronunciation": "",
#     "phonetic": "",
#     "part_of_speech": "",
#     "association": "",
#     "description": ""
#   }}
# ]

# word : 영어 단어
# meaning : 한국어 뜻
# pronunciation : 한국어 발음 (연상 기반)
# phonetic : 발음 기호
# part_of_speech : 품사
# association	: 경선식식 연상 문장
# description	: 해당 문장을 시각적으로 표현한 이미지 프롬프트

# #### 예시:
#   "input": [
#     {{
#       "word": "perish",
#       "meaning": "죽다",
#       "pronunciation": "페리쉬",
#       "phonetic": "ˈper.ɪʃ",
#       "part_of_speech": "v"
#     }},
#     {{
#       "word": "eager",
#       "meaning": "간절히 바라는",
#       "pronunciation": "이거",
#       "phonetic": "",
#       "part_of_speech": ""
#     }},
#     {{
#       "word": "cohesion",
#       "meaning": "결합, 화합",
#       "pronunciation": "코헤션",
#       "phonetic": "n",
#       "part_of_speech": ""
#     }},
#     {{
#       "word": "chill",
#       "meaning": "냉기, 한기",
#       "pronunciation": "칠",
#       "phonetic": "n",
#       "part_of_speech": ""
#     }}
#   ],
#   "output": [
#     {{
#       "word": "perish",
#       "meaning": "죽다",
#       "pronunciation": "페리쉬",
#       "phonetic": "ˈper.ɪʃ",
#       "association": "[파리 쉬~]",
#       "description": "A lifeless corpse lying on the ground with flies buzzing around it, symbolizing death."
#     }},
#     {{
#       "word": "eager",
#       "meaning": "간절히 바라는",
#       "pronunciation": "이거",
#       "phonetic": "",
#       "association": "[이거] 사주세요.",
#       "description": "A small child pointing at a teddy bear in a toy store, clearly begging a parent to buy it."
#     }},
#     {{
#       "word": "cohesion",
#       "meaning": "결합, 화합",
#       "pronunciation": "코헤션",
#       "phonetic": "n",
#       "association": "[코]를 맞대고 입을 결합 [헤션]",
#       "description": "A man and woman gently touching noses and kissing, expressing bonding."
#     }},
#     {{
#       "word": "chill",
#       "meaning": "냉기, 한기",
#       "pronunciation": "칠",
#       "phonetic": "n",
#       "association": "[칠]도의 기온으로 떨어져 쌀쌀한 날씨.",
#       "description": "A person shivering from the cold beside a thermometer showing 7°C."
#     }}
#   ]
# #### 이제 예시를 참고해서 출력물을 만들어주면 됩니다.
# {input_document}
#         """)

#     prompt_llm = prompt | llm

#     response = prompt_llm.invoke({"input_document": state["converted_vocabulary"]})


#     return {"gss_converted_vocabulary": response.content}
