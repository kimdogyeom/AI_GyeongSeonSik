from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from src.utils.tools import tools


def input_type_router(file_type:str):
    if file_type == "pdf":
        return "pdf_to_document"
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


def _get_model(model_name: str):
    if model_name == "openai":
        model = ChatOpenAI(temperature=0.5, model="gpt-4o")
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

    model = model.bind_tools(tools)
    return model


system_prompt = """Be a helpful assistant"""

# Define the function that calls the model
def call_model(state, config):
    messages = state["messages"]
    messages = [{"role": "system", "content": system_prompt}] + messages
    model_name = config.get('configurable', {}).get("model_name", "anthropic")
    model = _get_model(model_name)
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}
