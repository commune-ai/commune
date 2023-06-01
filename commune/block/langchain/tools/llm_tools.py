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


def _get_pal_math(llm: BaseLLM) -> Tool:
    return Tool(
        "PAL-MATH",
        PALChain.from_math_prompt(llm).run,
        "A language model that is really good at solving complex word math problems. Input should be a fully worded hard word math problem.",
    )


def _get_pal_colored_objects(llm: BaseLLM) -> Tool:
    return Tool(
        "PAL-COLOR-OBJ",
        PALChain.from_colored_object_prompt(llm).run,
        "A language model that is really good at reasoning about position and the color attributes of objects. Input should be a fully worded hard reasoning problem. Make sure to include all information about the objects AND the final question you want to answer.",
    )


def _get_llm_math(llm: BaseLLM) -> Tool:
    return Tool(
        "Calculator",
        LLMMathChain(llm=llm).run,
        "Useful for when you need to answer questions about math.",
    )


def _get_open_meteo_api(llm: BaseLLM) -> Tool:
    chain = APIChain.from_llm_and_api_docs(llm, open_meteo_docs.OPEN_METEO_DOCS)
    return Tool(
        "Open Meteo API",
        chain.run,
        "Useful for when you want to get weather information from the OpenMeteo API. The input should be a question in natural language that this API can answer.",
    )


_LLM_TOOLS = {
    "pal-math": _get_pal_math,
    "pal-colored-objects": _get_pal_colored_objects,
    "llm-math": _get_llm_math,
    "open-meteo-api": _get_open_meteo_api,
}
