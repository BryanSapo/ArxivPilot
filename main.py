from langgraph.graph import StateGraph, MessagesState, START, END
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.messages import AnyMessage, SystemMessage, ToolMessage
from typing_extensions import TypedDict, Annotated
import operator
from typing import Literal
import requests
import os
from dotenv import load_dotenv
from datetime import date
import re
load_dotenv()

model = ChatOpenAI(
    model=os.getenv("MODEL"),  # OpenRouter model slug
    temperature=0,
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=os.getenv("OPENROUTER_ENDPOINT"),
    timeout=60,       # fail instead of hanging forever
    max_retries=2,
)

@tool
def download_arxiv_pdf(url: str) -> str:
    """Download an arxiv paper's PDF and save it in ./tmp, named after the paper title.

    Pass ONLY the arxiv abstract URL. The title and filename are determined
    automatically inside this tool.

    Args:
        url (str): the url of an arxiv abstract webpage
            (e.g. https://arxiv.org/abs/2606.27350).

    Returns:
        A status message with the saved file path.
    """
    # Resolve the paper title from the abstract page.
    abs_response = requests.get(url)
    match = re.search(r"<title>(.*?)</title>", abs_response.text, re.IGNORECASE | re.DOTALL)
    if match:
        title = match.group(1).strip()
        # Strip the leading "[arxiv-id] " prefix arxiv adds to the title.
        title = re.sub(r"^\[[^\]]*\]\s*", "", title)
    else:
        title = "Unknown_Name"

    # Build the pdf url from the arxiv id.
    arxiv_id = url.rstrip("/").split("/")[-1]
    pdf_url = "https://arxiv.org/pdf/" + arxiv_id

    # Make the title safe to use as a filename (remove / : * ? " < > | etc.).
    safe_name = re.sub(r'[\\/:*?"<>|]', "_", title).strip()
    safe_name = safe_name[:150] or arxiv_id  # avoid empty/overly long filenames

    save_path = "./tmp/"
    os.makedirs(save_path, exist_ok=True)
    full_path = os.path.join(save_path, safe_name + ".pdf")

    response = requests.get(pdf_url, stream=True)
    if response.status_code == 200:
        with open(full_path, "wb") as pdf_file:
            for chunk in response.iter_content(chunk_size=2048):
                if chunk:
                    pdf_file.write(chunk)
        msg = f"Success! PDF downloaded and saved to: {full_path}"
        print(msg)
        return msg
    else:
        msg = f"Failed to fetch PDF. Status code: {response.status_code}"
        print(msg)
        return msg


tools = [download_arxiv_pdf]
tools_by_name = {tool.name: tool for tool in tools}

model_with_tools = model.bind_tools(tools)
    

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

def tool_node(state: dict):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}

def llm_call(state: dict):
    """LLM decides whether to call a tool or not"""

    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="""You are an automated research assistant designed to retrieve and download academic papers from arXiv.

                                    When a user provides an arXiv abstract URL, call the `download_arxiv_pdf` tool with that URL. The tool resolves the paper title, downloads the PDF, and saves it to ./tmp named after the paper title.

                                    **Operating Rules:**

                                    * Pass the user-provided arXiv abstract URL directly to `download_arxiv_pdf`.
                                    * Rely strictly on the tool output. Do not hallucinate file paths, titles, or URLs.
                                    * After the tool returns, report the saved file path to the user.
                        """
                        
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }

def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "tool_node"

    # Otherwise, we stop (reply to the user)
    return END

def tool_node(state: dict):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}

    

# def p_check(obj, name):
#     print(f"{name}", end = ' ')
#     print("Done" if obj != None else "Error")

if __name__ == "__main__":
    # Build workflow
    agent_builder = StateGraph(MessagesState)
    # p_check(agent_builder, "agent_builder")
    # Add nodes
    agent_builder.add_node("llm_call", llm_call)
    agent_builder.add_node("tool_node", tool_node)

    # Add edges to connect nodes
    agent_builder.add_edge(START, "llm_call")
    agent_builder.add_conditional_edges(
        "llm_call",
        should_continue,
        ["tool_node", END]
    )
    agent_builder.add_edge("tool_node", "llm_call")

    # Compile the agent
    agent = agent_builder.compile()
    # p_check(agent, "agent")

    # Show the agent
    png = agent.get_graph(xray=True).draw_mermaid_png()
    with open("agent_graph.png", "wb") as f:
        f.write(png)

    # Invoke
    from langchain.messages import HumanMessage
    messages = [HumanMessage(content=input("Please ask a question:\n"))]
    messages = agent.invoke({"messages": messages})
    # p_check(messages, "messages")
    for m in messages["messages"]:
        m.pretty_print()