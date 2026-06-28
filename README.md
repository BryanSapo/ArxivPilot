# ArxivPilot

A small command-line research assistant that downloads arXiv papers for you. Give it an
arXiv abstract URL and it fetches the paper's PDF, names the file after the paper's title,
and stores it locally. It can also list the papers you've already saved.

ArxivPilot is built as a [LangGraph](https://github.com/langchain-ai/langgraph) agent: an
LLM decides which tool to call, the tool runs, and the result is fed back to the LLM until
it has an answer for you.

## How it works

The agent is a simple loop (`llm_call → tool_node → llm_call → …`) that stops once the LLM
responds without requesting a tool:

```
START → llm_call ──(tool call?)──► tool_node ──► llm_call
                  └──(no tool call)──► END
```

- **`llm_call`** — sends the conversation (plus a system prompt) to the model and lets it
  decide whether to call a tool.
- **`should_continue`** — routes to `tool_node` if the model requested a tool, otherwise ends.
- **`tool_node`** — executes the requested tool(s) and returns the result as a `ToolMessage`.

A diagram of the compiled graph is written to `agent_graph.png` on every run.

## Tools

Defined in [`tools.py`](tools.py):

| Tool                      | Description                                                                                                                                                                                                                         |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `download_arxiv_pdf(url)` | Takes an arXiv **abstract** URL (e.g. `https://arxiv.org/abs/2606.27350`), resolves the paper title from the page, downloads the PDF, and saves it to `./tmp/` named after the title. Returns a status message with the saved path. |
| `list_all_papers()`       | Returns a newline-separated list of the PDFs currently saved in `./tmp/`.                                                                                                                                                           |

Downloaded papers are stored in the `./tmp/` directory (created automatically).

## Requirements

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- An [OpenRouter](https://openrouter.ai/) API key (or any OpenAI-compatible endpoint)

## Setup

1. Install dependencies:

    ```bash
    uv sync
    ```

2. Create a `.env` file in the project root:

    ```env
    OPENROUTER_API_KEY=sk-or-v1-...
    MODEL=openai/gpt-4o-mini
    OPENROUTER_ENDPOINT=https://openrouter.ai/api/v1
    ```

    | Variable              | Description                                                                         |
    | --------------------- | ----------------------------------------------------------------------------------- |
    | `OPENROUTER_API_KEY`  | Your API key for the LLM provider.                                                  |
    | `MODEL`               | The model slug to use. A model that supports **tool/function calling** is required. |
    | `OPENROUTER_ENDPOINT` | The OpenAI-compatible base URL.                                                     |

    > **Note:** Free (`:free`) models on OpenRouter are often heavily rate-limited and may
    > respond slowly or lack tool-calling support. A small paid model such as
    > `openai/gpt-4o-mini` is recommended for reliable, fast results.

## Usage

Run the agent and enter a request when prompted:

```bash
uv run python main.py
```

Example session:

```
Please ask a question:
https://arxiv.org/abs/2606.27350

Success! PDF downloaded and saved to: ./tmp/CHIA_ An open-source framework ... .pdf
```

You can also ask it to list what you've saved:

```
Please ask a question:
What papers do I have?
```

## Project structure

```
ArxivPilot/
├── main.py          # LangGraph agent: state graph, nodes, system prompt, CLI entry point
├── tools.py         # Tool definitions (download_arxiv_pdf, list_all_papers)
├── pyproject.toml   # Project metadata and dependencies
├── tmp/             # Downloaded PDFs (created at runtime)
└── agent_graph.png  # Graph visualization (generated at runtime)
```

## Notes & limitations

- Only arXiv **abstract** URLs are supported as input (e.g. `https://arxiv.org/abs/<id>`).
- Filenames are derived from the paper title with filesystem-unsafe characters replaced by
  `_` and truncated to 150 characters.
- The quality and speed of results depend on the configured model; weaker models may pick
  the wrong tool or be slow to respond.

## Security

Keep your `.env` file out of version control — it contains your API key. Add it to
`.gitignore` and rotate the key if it is ever exposed.
