from analyze_and_format.state import TransformState


def transform_to_format(state : TransformState) -> TransformState:
    chunks = state.get("chunks", [])
    fields = state.get("detected_fields", [])
    formatted = []

    for chunk in chunks:
        lines = chunk.strip().split("\n")
        if len(lines) >= len(fields):
            item = {field: lines[i].strip() for i, field in enumerate(fields)}
            formatted.append(item)

    state["formatted_chunks"] = formatted
    return state