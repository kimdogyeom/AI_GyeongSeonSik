from typing import TypedDict, List, Optional


class TransformState(TypedDict):
    sample_text: str
    full_text: str
    structure_description: str
    parser_code: str
    chunks: List[dict] # run_parser_code를 실행해서 리스트로 들어감
    error: Optional[str]
    error_type: Optional[str] 