from typing import TypedDict, Annotated, List
from langchain_core.documents import Document

class State(TypedDict):
    file_path : str
    file_type : str
    before_vocabulary : List[Document]
    formatted_chunks: List[dict]
    gss_converted_vocabulary : List[dict]
