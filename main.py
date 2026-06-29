from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage, ToolMessage, HumanMessage
from typing import Literal
import os
from dotenv import load_dotenv
from tools import download_arxiv_pdf, list_all_papers

load_dotenv()

model = ChatOpenAI(
    model=os.getenv("MODEL"),  # OpenRouter model slug
    temperature=0,
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=os.getenv("OPENROUTER_ENDPOINT"),
    timeout=60,       # fail instead of hanging forever
    max_retries=2,
)




tools = [download_arxiv_pdf, list_all_papers]
tools_by_name = {tool.name: tool for tool in tools}

model_with_tools = model.bind_tools(tools)

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
                        content="""You are ArxivPilot, an automated research assistant that downloads and keeps track of academic papers from arXiv. All papers are stored locally in the ./tmp/ directory.

                                    You have two tools:

                                    1. `download_arxiv_pdf(url)` — Given an arXiv abstract URL (e.g. https://arxiv.org/abs/2606.27350), it resolves the paper title, downloads the PDF, and saves it to ./tmp/ named after the paper title. It returns a status message with the saved file path.
                                    2. `list_all_papers()` — Returns the list of paper filenames currently saved in ./tmp/. Takes no arguments.

                                    How to act:

                                    * If the user provides an arXiv abstract URL (or asks to download/save/fetch a paper), call `download_arxiv_pdf` with that exact URL.
                                    * If the user asks what papers they have, what is downloaded/saved, or to list the library, call `list_all_papers`.
                                    * You may call tools more than once if the request involves multiple URLs or multiple steps.

                                    Operating rules:

                                    * Pass the user-provided arXiv abstract URL directly to `download_arxiv_pdf`; do not modify, shorten, or invent URLs.
                                    * Rely strictly on tool outputs. Never hallucinate file paths, titles, URLs, or whether a download succeeded.
                                    * After a tool returns, summarize the result for the user (e.g. the saved file path, or the list of papers).
                                    * If a tool reports a failure, relay the error plainly instead of pretending it worked.
                                    * If the request is unrelated to downloading or listing arXiv papers, briefly explain what you can do.
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
        result.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
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
    try:
        while True:
            messages = [HumanMessage(content=input("Please ask a question:\n"))]
            messages = agent.invoke({"messages": messages})
            # p_check(messages, "messages")
            for m in messages["messages"]:
                m.pretty_print()
    except KeyboardInterrupt:
        print("\nSESSION END!\n")