from typing import TypedDict, Annotated, List
from langchain_core.documents import Document
import operator

class State(TypedDict):
    file_path : str
    file_type : str
    has_kor_translated : bool
    before_vocabulary : List[Document]
    converted_vocabulary : str
    gss_converted_vocabulary : str
