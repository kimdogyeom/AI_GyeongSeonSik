from langchain_community.document_loaders import PyPDFLoader
from ..utils.state import State


def pdf_parser(state: State):
    """PDF문서를 Document형태로 반환"""

    file_path = state.get("file_path")

    loader = PyPDFLoader(file_path)

    documents = loader.load()
    return {"before_vocabulary": documents}
