from langchain_core.runnables import RunnableLambda, RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END

from src.utils.state import State
from src.tools.convert_GyeongSunSik import convert_GyeongSunSik
from src.tools.pdf_to_document import pdf_parser

from src.analyze_and_format.main import build_transform_subgraph


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

    # 부모 상태와 자식 상태 간의 데이터 변환 및 자식 그래프 호출 처리
    def analyze_and_format(state: State) -> State:
        vocabulary = "\n".join([doc.page_content for doc in state["before_vocabulary"]])

        sample_text = vocabulary[:1500] if len(vocabulary) > 1500 else vocabulary

        child_graph_input = {"sample_text": sample_text, "full_text": vocabulary}

        config = RunnableConfig(recursion_limit=15)
        child_graph_output = build_transform_subgraph().invoke(config=config, input=child_graph_input)

        state["formatted_chunks"] = child_graph_output["chunks"]
        return state

    # 노드 추가
    workflow.add_node("pdf_to_document", pdf_parser)
    workflow.add_node("analyze_and_format", analyze_and_format)
    workflow.add_node(
        "convert_GyeongSunSik",
        RunnableLambda(convert_GyeongSunSik)
    )

    # 엣지 정의
    workflow.add_conditional_edges(
        START,
        file_type_check,
        {
            "pdf_to_document": "pdf_to_document",
            END: END,
        },
    )

    workflow.add_edge("pdf_to_document", "analyze_and_format")
    workflow.add_edge("analyze_and_format", "convert_GyeongSunSik")
    workflow.add_edge("convert_GyeongSunSik", END)

    # 그래프 컴파일
    app = workflow.compile(checkpointer=memory)
    app.get_graph().draw_png("graph.png")

    return app
