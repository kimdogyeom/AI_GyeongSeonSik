from analyze_and_format.state import TransformState


def run_parser_code(state : TransformState) -> TransformState:
    full_text = state["full_text"]
    code = state["parser_code"]

    local_vars = {"text": full_text, "result": None}
    try:
        exec(code, {}, local_vars)

        # (1) result가 생성됐는지 확인
        if "result" not in local_vars:
            raise ValueError("Parser 코드가 result를 생성하지 않았습니다.")

        # (2) 리스트인지 확인
        if not isinstance(local_vars["result"], list):
            raise TypeError("Parser 코드의 결과는 리스트여야 합니다.")

        # (3) 이상 없을 경우에만 chunks로 저장
        state["chunks"] = local_vars["result"]

        # ✅ 성공했으므로 에러 상태 초기화
        state["error"] = None

    except Exception as e:
        state["error"] = str(e)
        state["error_type"] = type(e).__name__

    return state
