import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools import download_arxiv_pdf, list_all_papers


def test_download_arxiv_pdf():
    pass

def test_list_all_papers():
    assert list_all_papers.invoke({}) == os.listdir('./tmp/')
    