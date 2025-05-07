import asyncio
import json
import re
from typing import List, Dict

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_teddynote.models import get_model_name, LLMs

from ..utils.state import State

GPT4oMINI = get_model_name(LLMs.GPT4o_MINI)
llm = ChatOpenAI(temperature=1, model=GPT4oMINI)

async def convert_GyeongSunSik_chunk(chunk: List[Dict]) -> List[Dict]:
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

당신에게 영단어 형식의 데이터가 입력될 수 있고, 입력된 영단어의 혁식을 파악한 뒤 다음 출력예시와 같이 출력해야 합니다.

#### 출력예시:
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
#### 이제 예시를 참고해서 json형식으로만 출력물을 만들면 됩니다.
{input_document}
    """)

    prompt_llm = prompt | llm
    try:
        response = await prompt_llm.ainvoke({
            "input_document": json.dumps(chunk, ensure_ascii=False)
        })

        raw = response.content.strip()
        if raw.startswith("```json"):
            raw = re.sub(r"^```json\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
        elif isinstance(parsed, dict):
            return parsed.get("output", [])
        else:
            return []

    except Exception as e:
        print(f"[LLM 처리 오류] {type(e).__name__}: {e}")
        # 여기서 response.content 접근 금지!
        return []

async def convert_GyeongSunSik_async(state: State) -> State:
    formatted_chunks = state["formatted_chunks"]

    chunk_size = 50
    semaphore = asyncio.Semaphore(7)

    def chunkify(data, size):
        return [data[i:i + size] for i in range(0, len(data), size)]

    chunks = chunkify(formatted_chunks, chunk_size)
    print(f"📦 총 {len(chunks)}개의 청크를 처리합니다.")

    async def limited_chunk(chunk, idx):
        async with semaphore:
            print(f"🚀 chunk[{idx}] 처리 중...")
            result = await convert_GyeongSunSik_chunk(chunk)
            print(f"✅ chunk[{idx}] 결과 {len(result)}개")
            return result

    results = await asyncio.gather(*[
        limited_chunk(chunk, i) for i, chunk in enumerate(chunks)
    ])

    merged = []
    for i, result in enumerate(results):
        print(f"🔗 chunk[{i}] 병합 중... ({len(result)}개)")
        merged.extend(result)

    print(f"📦 총 결과 수: {len(merged)}개")
    state["gss_converted_vocabulary"] = merged
    return state



async def convert_GyeongSunSik(state: State) -> State:
    result = await convert_GyeongSunSik_async(state)
    return result
