from __future__ import annotations
from typing import Callable as _Callable, Any as _Any, List, Tuple, Dict, Union, Annotated
import pandas as pd
from PIL import Image as PILImage
import numpy as np
import gradio as gr

# NOTE ==============================================
# - Not Finished 
# - I have not fully test all types and these 
#   might change to corresponded to there correct
#   str counter part given they have one.

# ===================================================

# REASON For Commit =================================
# Building a project that would involve Gradio UI, 
# and I thought implementing annotation interpreter
# do the inputs and outputs would clean and quick 
# with interfaces as well with tabular interfaces.
# ===================================================

Dataframe = Annotated[Union[list[list[_Any]], _Callable, None], "dataframe"]
ColorPicker = Annotated[ Union[float, _Callable, None], "colorpicker"]
Code = Annotated[Union[str , tuple[str] , None], "code"]


# Input/Output =========================
Textbox = Text = Annotated[Union[str, None, _Callable], "text"]
Image = Annotated[Union[str, PILImage.Image, np.ndarray, None], "image"]
Model3D = Annotated[Union[str, _Callable, None], "model3d"]
Number = Annotated[Union[float, int , None], "number"]
Slider = Annotated[Union[float, int, _Callable, None], "slider"]
Timeseries = Annotated[Union[str, pd.DataFrame, _Callable, None], "timeseries"]
UploadButton = Annotated[Union[str, List[str], _Callable, None], "uploadbutton"]
Video = Annotated[Union[str, Tuple[str, Union[str, None]], _Callable, None], "video"]
Audio = Annotated[Union[str, Tuple[int, np.ndarray], _Callable, None], "audio"]
Button = Annotated[Union[str, _Callable], "button"]
Checkbox = Annotated[Union[bool, _Callable],"checkbox"]
File = Annotated[Union[str, List[str], _Callable, None], "file"]

# Requires input
# Radio = _TypeVar["Radio", str, _Callable, None]
# Dropdown = _TypeVar["Dropdown",  list[str], str, _Callable, None]
# CheckboxGroup = _TypeVar["CheckboxGroup",  str , tuple[str] , None] # give default examples to run these

# Output only ==========================
AnnotatedImage = Annotated[Union[Tuple[Union[np.ndarray, PILImage.Image, str], List[Tuple[Union[np.ndarray, Tuple[int, int, int, int]], str]]], None],"annotatedimage"]
BarPlot = Annotated[Union[pd.DataFrame, _Callable, None], "barplot"]
Markdown = Annotated[Union[str, _Callable], "markdown"]
Json = Annotated[Union[str, Dict, List, _Callable, None], "json"]
Label = Annotated[Union[Dict[str, float], str, float, _Callable, None], "label"]
LinePlot = Annotated[Union[pd.DataFrame, _Callable, None], "lineplot"]
Plot = Annotated[Union[_Callable, None, pd.DataFrame], "plot"]
ScatterPlot = Annotated[Union[pd.DataFrame, _Callable, None], "scatterplot"]
Gallery = Annotated[Union[List[Union[np.ndarray, PILImage.Image, str, Tuple]], _Callable, None], "gallery"]
Chatbot = Annotated[Union[List[List[Union[str, Tuple[str], Tuple[str, str], None]]], _Callable, None], "chatbot"]
Html = Annotated[Union[str, _Callable], "html"]
HighlightedText = Annotated[Union[List[Tuple[str, Union[str, float, None]]], Dict, _Callable, None], "highlightedText"]


Block = Annotated[Union[gr.Blocks, gr.blocks.Blocks], "Block"]
Interface = Annotated[gr.Interface, "Interface"]

