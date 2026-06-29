from langchain.tools import tool
import re
from datetime import date
import requests
import os

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
        # print(msg)
        return msg
    else:
        msg = f"Failed to fetch PDF. Status code: {response.status_code}"
        # print(msg)
        return msg
    
@tool
def list_all_papers() -> str:
    """List all of the papers that saved under ./tmp/

    Returns:
        str: a newline-separated list of saved paper filenames.
    """
    if not os.path.isdir('./tmp/'):
        return "No papers saved yet."

    items = [f for f in os.listdir('./tmp/')
             if f.endswith('.pdf') and not f.startswith('.')]

    if not items:
        return "No papers saved yet."

    return "\n".join(sorted(items))
