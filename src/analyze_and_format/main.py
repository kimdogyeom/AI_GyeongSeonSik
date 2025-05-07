from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda
from .nodes import analyze_structure, generate_parser_code, run_parser_code, transform_to_format

# 조건 판별 함수
def check_parser_success(state: dict) -> str:
    if state.get("error"):  # error 키가 존재하면 실패로 간주
        return "retry"
    return "success"

# LangGraph 서브그래프 정의
def build_transform_subgraph():
    sub_builder = StateGraph(dict)

    sub_builder.add_node("analyze_structure", RunnableLambda(analyze_structure))
    sub_builder.add_node("generate_parser_code", RunnableLambda(generate_parser_code))
    sub_builder.add_node("run_parser_code", RunnableLambda(run_parser_code))

    sub_builder.add_edge(START, "analyze_structure")
    sub_builder.add_edge("analyze_structure", "generate_parser_code")
    sub_builder.add_edge("generate_parser_code", "run_parser_code")

    # 조건 분기 등록
    sub_builder.add_conditional_edges("run_parser_code", check_parser_success, {
        "success": END,
        "retry": "generate_parser_code",  # 실패 시 parser_code 다시 생성
    })


    sub_builder.set_entry_point("analyze_structure")
    sub_builder.set_finish_point("run_parser_code")

    app = sub_builder.compile()
    app.get_graph().draw_png("subGraph.png")

    return app
