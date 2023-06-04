# flake8: noqa
"""Load tools."""
from typing import Any, List, Optional

from langchain.agents.tools import Tool
from langchain.chains.api import news_docs, open_meteo_docs, tmdb_docs
from langchain.chains.api.base import APIChain
from langchain.chains.llm_math.base import LLMMathChain
from langchain.chains.pal.base import PALChain
from langchain.llms.base import BaseLLM
from langchain.python import PythonREPL
from langchain.requests import RequestsWrapper
from langchain.serpapi import SerpAPIWrapper
from langchain.utilities.bash import BashProcess
from langchain.utilities.google_search import GoogleSearchAPIWrapper
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper


def _get_python_repl() -> Tool:
    return Tool(
        "Python REPL",
        PythonREPL().run,
        "A Python shell. Use this to execute python commands. Input should be a valid python command. If you expect output it should be printed out.",
    )


def _get_serpapi() -> Tool:
    return Tool(
        "Search",
        SerpAPIWrapper().run,
        "A search engine. Useful for when you need to answer questions about current events. Input should be a search query.",
    )


def _get_google_search() -> Tool:
    return Tool(
        "Google Search",
        GoogleSearchAPIWrapper().run,
        "A wrapper around Google Search. Useful for when you need to answer questions about current events. Input should be a search query.",
    )


def _get_wolfram_alpha() -> Tool:
    return Tool(
        "Wolfram Alpha",
        WolframAlphaAPIWrapper().run,
        "A wrapper around Wolfram Alpha. Useful for when you need to answer questions about Math, Science, Technology, Culture, Society and Everyday Life. Input should be a search query.",
    )


def _get_requests() -> Tool:
    return Tool(
        "Requests",
        RequestsWrapper().run,
        "A portal to the internet. Use this when you need to get specific content from a site. Input should be a specific url, and the output will be all the text on that page.",
    )


def _get_terminal() -> Tool:
    return Tool(
        "Terminal",
        BashProcess().run,
        "Executes commands in a terminal. Input should be valid commands, and the output will be any output from running that command.",
    )


_BASE_TOOLS = {
    "python_repl": _get_python_repl,
    "serpapi": _get_serpapi,
    "requests": _get_requests,
    "terminal": _get_terminal,
    "google-search": _get_google_search,
    "wolfram-alpha": _get_wolfram_alpha,
}
