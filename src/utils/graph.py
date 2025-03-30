from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END

from src.utils.state import State
from src.tools.convert_GyeongSunSik import convert_GyeongSunSik
from src.tools.create_korean_meaning import create_korean_meaning
from src.tools.identify_document_pattern import identify_document_pattern
from src.tools.pdf_to_document import pdf_parser


def Graph():
    # 메모리에 대화 기록을 저장하기 위한 MemorySaver 초기화
    memory = MemorySaver()

    # 상태 그래프 초기화
    workflow = StateGraph(State)

    def file_type_check(state: State):
        file_type = state["file_type"]
        print(f"[DEBUG] file_type_check() - file_type: {file_type}")  # 디버깅용 로그
        if file_type == "pdf":
            return "pdf_to_document"
        else:
            return END

    # 노드 추가
    workflow.add_node("pdf_to_document", pdf_parser)
    # workflow.add_node("identify_document_pattern", identify_document_pattern)
    workflow.add_node("create_korean_meaning", create_korean_meaning)
    workflow.add_node("convert_GyeongSunSik", convert_GyeongSunSik)

    # 엣지 정의
    workflow.add_conditional_edges(
        START,
        file_type_check,
        {
            "pdf_to_document": "pdf_to_document",
            END: END,
        },
    )
    # workflow.add_edge("pdf_to_document", "identify_document_pattern")

    # workflow.add_conditional_edges(
    #     "identify_document_pattern",
    #     lambda state: state["has_kor_translated"],
    #     {
    #         True: "convert_GyeongSunSik",
    #         False: "create_korean_meaning"
    #     })

    workflow.add_edge("pdf_to_document", "create_korean_meaning")
    workflow.add_edge("create_korean_meaning", "convert_GyeongSunSik")
    workflow.add_edge("convert_GyeongSunSik", END)

    # 그래프 컴파일
    app = workflow.compile(checkpointer=memory)
    app.get_graph().draw_png("graph.png")

    return app
