import pandas as pd
import math
import numpy as np
import gc
import time
import gradio as gr
import os
from transformers.training_args import OptimizerNames
from huggingface_hub import hf_hub_download
from pathlib import Path
import traceback
import numpy as np
import glob
import shutil
import torch
import socket
from transformers import TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig
from commune..model.lora.lora import LoraModel
from datasets import Dataset

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LOCAL_HOST_IP = "0.0.0.0"
TENSORBOARD_URL = "http://" + LOCAL_HOST_IP + ":6006/"
INIT_DATASET_NAME = "test_python_code_instructions_5000_rows"

training_ret_val = -1
error_msg = ""
current_running_model_name = ""
infer_model = None
stop_generation_status = False
chatbot_history = []
chatbot_height = 500
rag_chatbot_history = []
rag_stop_generation_status = False
# qa_with_rag = QAWithRAG()
train_param_config = {}
train_param_config["dataset"] = {}
train_param_config["model"] = {}
train_param_config["training"] = {}
# transformer_optimizer_list = ['Adam', 'Ada']
transformer_optimizer_list = list(vars(OptimizerNames)["_value2member_map_"].keys())

lora_params = LoraConfig()
training_args = TrainingArguments('')

col_names = []
DATASET_FIRST_ROW = None
local_model_list = ""
local_model_root_dir = ""
base_model_names = []
training_base_model_names = [
    'mistralai/Mistral-7B-v0.1',
    'mistralai/Mistral-7B-Instruct-v0.1',
    'meta-llama/Llama-2-7b-hf',
    'meta-llama/Llama-2-7b-chat-hf',
    'meta-llama/Llama-2-70b-chat-hf',
    'openlm-research/open_llama_7b',
    'openlm-research/open_llama_13b',
    'togethercomputer/LLaMA-2-7B-32K',
    'lmsys/vicuna-7b-v1.3',
    'lmsys/vicuna-13b-v1.3',
    'lmsys/vicuna-33b-v1.3',
    'tiiuae/falcon-40b-instruct',
    'CalderaAI/30B-Lazarus'
]
embedding_model_names = []
base_model_context_window = []
local_dataset_list = []
local_dataset_root_dir = ""

model_context_window = [2048, 1024, 512]

lr_scheduler_list = ["constant",
                     "linear",
                     "cosine",
                     "cosine_with_hard_restarts",
                     "polynomial_decay",
                     "constant_with_warmup",
                     "inverse_sqrt",
                     "reduce_on_plateau"]

INIT_PREFIX1 = "<s>[INST] "
INIT_PREFIX2 = "here are the inputs "
INIT_PREFIX3 = " [/INST]"
INIT_PREFIX4 = "</s>"

INIT_COL1_TEXT = ''
INIT_COL2_TEXT = ''
INIT_COL3_TEXT = ''
INIT_COL4_TEXT = ''


def list_adaptors(path):
    adaptor_list = []
    for dir in os.listdir(path):
        if os.path.isdir(os.path.join(path, dir)):
            if 'adapter_config.json' in os.listdir(os.path.join(path, dir)):
                adaptor_list.append(os.path.join(path, dir))
            else:
                adaptor_list += list_adaptors(os.path.join(path, dir))

    return adaptor_list

with gr.Blocks(
        title="FINETUNE",
        css="#vertical_center_align_markdown { position:absolute; top:30%;background-color:white;} .white_background {background-color: #ffffff} .none_border {border: none;border-collapse:collapse;}"
) as demo:

    # Local variables
    adaptor = LoraModel()
    adaptor.base_model_name = ''#training_base_model_names[0]
    text_file_list = []
    base_lora_path = '/home/v/adaptors'
    lora_output_path = ''

    def list_text_files(data_files):
        global text_file_list
        for file in data_files:
            if os.path.isfile(file):
                text_file_list.append(file)
        if len(text_file_list) > 0:
            if os.path.isfile(text_file_list[0]):
                with open(text_file_list[0], 'rt', encoding='utf8') as f:
                    raw_txt = f.readline()
        file_list_text = '\n'.join(text_file_list)
        return gr.update(value=file_list_text), gr.update(value=raw_txt)

    # local_model_root_dir_textbox = gr.Textbox(label="", value=local_model_root_dir, visible=False)
    # local_dataset_root_dir_textbox = gr.Textbox(label="", value=local_dataset_root_dir, visible=False)
    # local_embedding_model_root_dir_textbox = gr.Textbox(
    #     label="",
    #     value=os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag", "embedding_models"),
    #     visible=False)
    # local_chat_model_root_dir_textbox = gr.Textbox(label="", value=local_model_root_dir, visible=False)
    # local_home_chat_model_root_dir_textbox = gr.Textbox(label="", value=local_model_root_dir, visible=False)
    session_state = gr.State(value={})
    # html = gr.HTML("<p  align='center';>llm-web-ui</p>",elem_id="header")

    with gr.Tab("Fine-Tuning"):
        with gr.Tabs() as tensorboard_tab:
            with gr.TabItem("Training", id=0):
                with gr.Row():
                    with gr.Column(scale=1, min_width=1):
                        with gr.Group():
                            gr.Markdown("## &nbsp;1.Training")  # , elem_classes="white_background"
                            with gr.Group():
                                gr.Markdown("### &nbsp;1).Model")  # , elem_classes="white_background"
                                with gr.Group():
                                    # gr.Markdown("<br> &nbsp;&nbsp;&nbsp; Base Model")

                                    # Base model selection part
                                    # base_model_source_radio_choices = ["Download From Huggingface Hub",
                                    #                                    f"From Local Dir(hg format:{local_model_root_dir})"]
                                    # base_model_source_radio = gr.Radio(base_model_source_radio_choices,
                                    #                                    label="Base Model",
                                    #                                    value=base_model_source_radio_choices[0],
                                    #                                    interactive=True)
                                    with gr.Row():  # elem_classes="white_background"
                                        base_model_name_dropdown = gr.Dropdown(training_base_model_names,
                                                                               label="Base Model Name",
                                                                               value=training_base_model_names[
                                                                                   0] if training_base_model_names else None,
                                                                               interactive=True, visible=True, scale=5,
                                                                               allow_custom_value=True)
                                        load_base_model_btn = gr.Button("Load Model", scale=1, visible=True)
                                        # stop_download_local_model_btn = gr.Button("Stop", scale=1, visible=False)

                                    # TODO : Should define the validate_model_path
                                    # if validate_model_path(training_base_model_names[0])[0]:
                                    #     download_model_status_markdown = gr.Markdown('<span style="color:green">&nbsp;&nbsp;&nbsp;&nbsp;This model has already been downloaded to local.</span>')
                                    # else:
                                    #     download_model_status_markdown = gr.Markdown('<span style="color:red">&nbsp;&nbsp;&nbsp;&nbsp;This model has not been downloaded.</span>')

                                    # with gr.Row():
                                    #     # TODO : implement get_local_models func
                                    #     # local_model_list = get_hg_model_names_from_dir(os.path.dirname(os.path.abspath(__file__)), "models")
                                    #     base_model_names = ['A', 'B']
                                    #     local_model_dropdown = gr.Dropdown(local_model_list, label="Local Model",
                                    #                                        info="",
                                    #                                        value=local_model_list[0] if len(local_model_list) > 0 else None,
                                    #                                        interactive=True,
                                    #                                        elem_classes="white_background", scale=5,
                                    #                                        visible=False)
                                    #     refresh_local_model_list_btn = gr.Button("Refresh", scale=1, visible=False)

                                    fine_tuning_type_dropdown = gr.Dropdown(["QLoRA", "LoRA"],
                                                                            label="Fine-Tuning Type", info="",
                                                                            value="QLoRA", interactive=True)

                                with gr.Group():
                                    gr.Markdown(
                                        "###  &nbsp;&nbsp;&nbsp; LoRA Config")  # , elem_classes="white_background"
                                    with gr.Row():  # elem_classes="white_background"
                                        lora_r_slider = gr.Slider(8, 64, value=8, step=8, label="lora_r", interactive=True)

                                        lora_alpha_slider = gr.Slider(8, 96, value=32, step=8, label="lora_alpha", interactive=True)
                                    with gr.Row():  # elem_classes="white_background"
                                        lora_dropout_slider = gr.Slider(
                                            0,
                                            1,
                                            value=0.05,
                                            step=0.01,
                                            label="lora_dropout",
                                            interactive=True
                                        )

                                        lora_bias_dropdown = gr.Dropdown(
                                            ["none", "all", "lora_only"],
                                            label="lora_bias",
                                            info="",
                                            value="none",
                                            interactive=True)
                                    adaptor_output_path_textbox = gr.Textbox(
                                        label='Adaptor Output Name',
                                        interactive=True,
                                        visible=True
                                    )

                            with gr.Group():
                                gr.Markdown("### &nbsp;2).Dataset")  # ,elem_classes="white_background"

                                # TODO : implement for huggingface dataset
                                # dataset_source_radio_choices = ["Download From Huggingface Hub",
                                #                                    f"From Local HG Dataset In {local_dataset_root_dir})"]
                                # dataset_source_radio = gr.Radio(dataset_source_radio_choices, label="Dataset Source",
                                #                                    value=dataset_source_radio_choices[1], interactive=True)

                                # with gr.Row(equal_height=True):
                                #     hg_dataset_path_textbox = gr.Textbox(label="Dataset Name:",elem_classes="none_border",visible=False, interactive=True, scale=4,
                                #                                          value="iamtarun/python_code_instructions_18k_alpaca")
                                #     download_local_dataset_btn = gr.Button("Download", scale=1, visible=False)
                                #     stop_download_local_dataset_btn = gr.Button("Stop", scale=1, visible=False)
                                # download_dataset_status_markdown = gr.Markdown('')
                                # with gr.Row():
                                #         hg_train_dataset_dropdown = gr.Dropdown(["train"], label="Train set", info="", interactive=False,visible=False,
                                #                                            scale=1,value="train")#elem_classes="white_background",
                                #         hg_val_dataset_dropdown = gr.Dropdown([], label="Val set", info="", interactive=False,visible=False,
                                #                                            scale=1)#elem_classes="white_background",

                                # with gr.Row():
                                    # local_dataset_list.pop(
                                    #     local_dataset_list.index(INIT_DATASET_NAME))
                                    # local_dataset_list.insert(0, INIT_DATASET_NAME)
                                    # local_train_path_dataset_dropdown = gr.Dropdown(local_dataset_list, label="Train Dataset", info="",
                                    #                                    value=local_dataset_list[0] if len(local_dataset_list)>0 else None, interactive=True,
                                    #                                    scale=5, visible=True)#elem_classes="white_background",
                                    # refresh_local_train_path_dataset_list_btn = gr.Button("Refresh", scale=1, visible=True)
                                    # local_train_dataset_path_textbox = gr.Textbox(
                                    #     label='Dataset path:',
                                    #     interactive=True,
                                    #     scale=4,
                                    #     visible=True)
                                local_train_data_files = gr.File(
                                    label='Text files',
                                    file_count='multiple',
                                    height=100,
                                    interactive=True,
                                    visible=True
                                )

                                local_train_data_files_textbox = gr.Textbox(
                                    label='Text files :',
                                    max_lines=8,
                                    interactive=False,
                                    visible=True
                                )

                                data_sample_textbox = gr.Textbox(label='Sample data:', interactive=False,
                                                                 value="", lines=4)

                                local_train_dataset_path_browser = gr.Interface(
                                    fn=list_text_files,
                                    inputs=local_train_data_files,
                                    outputs=[local_train_data_files_textbox, data_sample_textbox],
                                    allow_flagging='never'
                                )

                                # TODO : implement for huggingface dataset
                                # with gr.Row():
                                #         local_train_dataset_dropdown = gr.Dropdown(["train"], label="Train set", info="", interactive=True,
                                #                                            scale=1,value="train",visible=True)#elem_classes="white_background",
                                #         local_val_dataset_dropdown = gr.Dropdown([], label="Val set", info="", interactive=True,
                                #                                            scale=1,visible=True)#elem_classes="white_background",

                                # with gr.Group():#elem_classes="white_background"
                                #     # gr.Markdown("<h4><br> &nbsp;&nbsp;Prompt Template: (Prefix1 + ColumnName1 + Prefix2 + ColumnName2)</h4>",elem_classes="white_background")
                                #     gr.Markdown("<br> &nbsp;&nbsp;&nbsp;&nbsp;**Prompt Template: (Prefix1+ColumnName1+Prefix2+ColumnName2+Prefix3+ColumnName3+Prefix4+ColumnName4)**")#,elem_classes="white_background"
                                #     gr.Markdown(
                                #         "<span> &nbsp;&nbsp;&nbsp;&nbsp;**Note**:&nbsp;&nbsp;Llama2/Mistral Chat Template:<s\>[INST] instruction+input [/INST] output</s\> </span>")#,elem_classes="white_background"
                                #     # using_llama2_chat_template_checkbox = gr.Checkbox(True, label="Using Llama2/Mistral chat template",interactive=True,visible=False)
                                #     with gr.Row():#elem_classes="white_background"
                                #         # prompt_template
                                #         prefix1_textbox = gr.Textbox(label="Prefix1:",value=INIT_PREFIX1,lines=2,interactive=True)#,elem_classes="white_background"
                                #         datatset_col1_dropdown = gr.Dropdown(col_names, label="ColumnName1:", info="",value=col_names[1] if len(col_names) > 1 else None,interactive=True)#,elem_classes="white_background"
                                #         prefix2_textbox = gr.Textbox(label="Prefix2:",value=INIT_PREFIX2,lines=2,interactive=True)#,elem_classes="white_background"
                                #         datatset_col2_dropdown = gr.Dropdown(col_names, label="ColumnName2:", info="",value=col_names[2] if len(col_names) > 2 else None,interactive=True)#,elem_classes="white_background"
                                #     with gr.Row():#elem_classes="white_background"
                                #         prefix3_textbox = gr.Textbox(label="Prefix3:",value=INIT_PREFIX3,lines=2,interactive=True)#,elem_classes="white_background"
                                #         datatset_col3_dropdown = gr.Dropdown(col_names, label="ColumnName3:", info="",value=col_names[3] if len(col_names) > 3 else None,interactive=True)#,elem_classes="white_background"
                                #         prefix4_textbox = gr.Textbox(label="Prefix4:",value=INIT_PREFIX4,lines=2,interactive=True)#,elem_classes="white_background"
                                #         datatset_col4_dropdown = gr.Dropdown(col_names, label="ColumnName4:", info="",value=col_names[0] if len(col_names) > 4 else None,interactive=True)#,elem_classes="white_background"
                                #     # print("")
                                #     prompt_sample = INIT_PREFIX1 + INIT_COL1_TEXT + INIT_PREFIX2 + INIT_COL2_TEXT + INIT_PREFIX3 + INIT_COL3_TEXT + INIT_PREFIX4 + INIT_COL4_TEXT
                                #     prompt_sample_textbox = gr.Textbox(label="Prompt Sample:",interactive=False,value=prompt_sample,lines=4)

                                # max_length_dropdown = gr.Dropdown(["Model Max Length"] + model_context_window,
                                #                                   label="Max Length", value="Model Max Length",
                                #                                   interactive=True, allow_custom_value=True)

                            with gr.Group():
                                gr.Markdown("### &nbsp;3).Training Arguments")  # ,elem_classes="white_background"
                                with gr.Row():  # elem_classes="white_background"
                                    epochs_slider = gr.Slider(1, 100, value=10, step=1, label="Epochs",
                                                              interactive=True)
                                    batch_size_list = [1, 2, 3] + [bi for bi in range(4, 32 + 1, 4)]
                                    batch_size_slider = gr.Slider(1, 100, value=1, step=1, label="Batch Size",
                                                                  interactive=True)
                                with gr.Row():  # elem_classes="white_background"
                                    learning_rate_slider = gr.Slider(0, 0.01, value=2e-4, step=0.0001,
                                                                     label="Learning Rate", interactive=True)
                                    warmup_steps_slider = gr.Slider(0, 400, value=100, step=10, label="Warmup Steps",
                                                                    interactive=True)
                                with gr.Row():  # elem_classes="white_background"
                                    optimizer_dropdown = gr.Dropdown(transformer_optimizer_list, label="Optimizer",
                                                                     info="",
                                                                     value=transformer_optimizer_list[1],
                                                                     interactive=True)

                                    lr_scheduler_type_dropdown = gr.Dropdown(lr_scheduler_list,
                                                                             label="LR Scheduler Type", info="",
                                                                             value=lr_scheduler_list[0],
                                                                             interactive=True)
                                with gr.Row():  # elem_classes="white_background"
                                    early_stopping_patience_slider = gr.Slider(0, 50 + 1, value=0, step=5,
                                                                               label="Early Stopping Patience",
                                                                               interactive=True)
                                    gradient_accumulation_steps_slider = gr.Slider(1, 50, value=1, step=1,
                                                                                   label="Gradient Accumulation Steps")
                                with gr.Row():  # elem_classes="white_background"
                                    eval_steps_slider = gr.Slider(0, 1000, value=100, step=100, label="eval_steps",
                                                                  interactive=True)
                                    gradient_checkpointing_checkbox = gr.Checkbox(False, label="Gradient Checkpointing",
                                                                                  interactive=True)
                            train_btn = gr.Button("Start Training")

                    with gr.Column(scale=1, min_width=1):
                        with gr.Group():
                            gr.Markdown("## &nbsp;2.Test")  # ,elem_classes="white_background"
                            local_lora_list = list_adaptors(base_lora_path)
                            local_lora_list = [lora_path[len(base_lora_path)+1:] for lora_path in local_lora_list]
                            # training_runs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs')
                            # run_names = os.listdir(training_runs_dir)
                            # run_names.sort(key=lambda file:os.path.getmtime(os.path.join(training_runs_dir,file)))
                            # runs_output_model = []
                            # for run_name in run_names:
                            #     run_name_dir = os.path.join(training_runs_dir,run_name)
                            #     run_output_model = os.path.join(run_name_dir,"output_model")
                            #     if os.path.exists(run_output_model):
                            #         run_output_model_names = os.listdir(run_output_model)
                            #         for run_output_model_name in run_output_model_names:
                            #             if run_output_model_name.find("merged_")>=0:
                            #                 runs_output_model.append(os.path.join(run_name,"output_model",run_output_model_name, "ori"))
                            local_lora_list = local_lora_list[::-1]
                            local_lora_dropdown = gr.Dropdown(
                                local_lora_list,
                                label="Pretrained LoRAs",
                                value=local_lora_list[0] if local_lora_list else None,
                                interactive=True
                            )
                            with gr.Row():
                                refresh_lora_list_btn = gr.Button('Refresh', scale=1)
                                load_lora_btn = gr.Button('Load', scale=1)
                            # gr.Markdown("")
                            # gr.Markdown(
                            #     "<span> &nbsp;&nbsp;&nbsp;&nbsp;**Note**:&nbsp;&nbsp;Llama2/Mistral Chat Template:<s\>[INST] instruction+input [/INST] output</s\> </span>"
                            #     )#, elem_classes="white_background"
                            # with gr.Row():
                            test_input_textbox = gr.Textbox(
                                label="Input:",
                                interactive=True,
                                value="",
                                lines=6,
                                scale=4
                            )
                            with gr.Row():
                                generate_text_btn = gr.Button("Generate", scale=1, interactive=False)
                                lora_4bit_quant_checkbox = gr.Checkbox(
                                    True,
                                    label="Using 4-bit quantization",
                                    interactive=True,
                                    visible=True,
                                    info="Less memory but slower",
                                    scale=1
                                )
                                # test_prompt = gr.Textbox(label="Prompt:", interactive=False, lines=2, scale=1)
                            output_textbox = gr.Textbox(
                                label="Output:",
                                interactive=False,
                                lines=8,
                                scale=1
                            )

                        # # with gr.Group():
                        # #     gr.Markdown("## &nbsp;3.Quantization",elem_classes="white_background")
                        # #     with gr.Row():
                        # #         quantization_type_list = ["gguf"]
                        # #         quantization_type_dropdown = gr.Dropdown(
                        # #             quantization_type_list,
                        # #             label="Quantization Type",
                        # #             value=quantization_type_list[0],
                        # #             interactive=True,scale=3
                        # #             )
                        # #         local_quantization_dataset_dropdown = gr.Dropdown(local_dataset_list, label="Dataset for quantization",
                        # #                                                    value=local_dataset_list[0] if len(
                        # #                                                        local_dataset_list) > 0 else None,
                        # #                                                    interactive=True,
                        # #                                                    elem_classes="white_background", scale=7,
                        # #                                                    visible=False)
                        # #         refresh_local_quantization_dataset_btn = gr.Button("Refresh", scale=2, visible=False)
                        # #         def click_refresh_local_quantization_dataset_btn():
                        # #             local_dataset_list, _ = get_local_dataset_list()
                        # #             return gr.update(choices=local_dataset_list,
                        # #                              value=local_dataset_list[0] if len(local_dataset_list) > 0 else "")
                        # #         refresh_local_quantization_dataset_btn.click(click_refresh_local_quantization_dataset_btn,[],local_quantization_dataset_dropdown)

                        #     with gr.Row():
                        #         training_runs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs')
                        #         run_names = os.listdir(training_runs_dir)
                        #         run_names.sort(key=lambda file: os.path.getmtime(os.path.join(training_runs_dir, file)))
                        #         runs_output_model = []
                        #         for run_name in run_names:
                        #             run_name_dir = os.path.join(training_runs_dir, run_name)
                        #             run_output_model = os.path.join(run_name_dir, "output_model")
                        #             if os.path.exists(run_output_model):
                        #                 run_output_model_names = os.listdir(run_output_model)
                        #                 for run_output_model_name in run_output_model_names:
                        #                     if run_output_model_name.find("merged_") >= 0:
                        #                         runs_output_model.append(
                        #                             os.path.join(run_name, "output_model", run_output_model_name,
                        #                                          "ori"))
                        #         runs_output_model = runs_output_model[::-1]
                        #         quantization_runs_output_model_dropdown = gr.Dropdown(runs_output_model,
                        #                                                             label="runs_output_model",
                        #                                                             value=runs_output_model[
                        #                                                                 0] if runs_output_model else None,
                        #                                                             interactive=True, scale=6)

                        #         quantize_btn = gr.Button("Quantize", scale=1,visible=False)
                        #     if runs_output_model:
                        #         model_name = runs_output_model[0].split(os.sep)[-2].split('_')[-1]
                        #         quantized_model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs',
                        #                                            os.sep.join(runs_output_model[0].split(os.sep)[0:-1]),
                        #                                            "quantized_" + quantization_type_list[0] + "_" + model_name)
                        #         if not os.path.exists(quantized_model_dir):
                        #             os.makedirs(quantized_model_dir)
                        #         quantization_logging_markdown = gr.Markdown("")
                        #         gguf_quantization_markdown0 = gr.Markdown("### &nbsp;&nbsp;&nbsp;&nbsp;GGUF Quantization Instruction:", elem_classes="white_background", visible=True)
                        #         gguf_quantization_markdown1 = gr.Markdown('''&nbsp;&nbsp;&nbsp;&nbsp;1.Follow the instructions in the llama.cpp to generate a GGUF:[https://github.com/ggerganov/llama.cpp#prepare-data--run](https://github.com/ggerganov/llama.cpp#prepare-data--run),<span style="color:red">&nbsp;&nbsp;Q4_K_M is recommend</span>''',visible=True)
                        #         if runs_output_model:
                        #             gguf_quantization_markdown2 = gr.Markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;2.Convert {runs_output_model[0]} to gguf model",visible=True)
                        #         else:
                        #             gguf_quantization_markdown2 = gr.Markdown(
                        #                 f"", visible=True)
                        #         gguf_quantization_markdown3 = gr.Markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;3.Deploy gguf model", visible=False)
                        #     else:

                        #         quantization_logging_markdown = gr.Markdown("")
                        #         gguf_quantization_markdown0 = gr.Markdown("### &nbsp;&nbsp;&nbsp;&nbsp;GGUF Quantization Instruction:", elem_classes="white_background", visible=True)
                        #         gguf_quantization_markdown1 = gr.Markdown('''''',visible=True)
                        #         gguf_quantization_markdown2 = gr.Markdown(f"",visible=True)
                        #         gguf_quantization_markdown3 = gr.Markdown(f"", visible=True)


    #                         with gr.Group(visible=False):
    #                             gr.Markdown("## &nbsp;4.Deploy",elem_classes="white_background")
    #                             with gr.Row():
    #                                 deployment_framework_dropdown = gr.Dropdown(["TGI","llama-cpp-python"], label="Deployment Framework",value="TGI", interactive=True)
    #                             with gr.Row():
    #                                 training_runs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs')
    #                                 run_names = os.listdir(training_runs_dir)
    #                                 run_names.sort(key=lambda file: os.path.getmtime(os.path.join(training_runs_dir, file)))
    #                                 # ori_model_runs_output_model = []
    #                                 tgi_model_format_runs_output_model = []
    #                                 gguf_model_format_runs_output_model = []
    #                                 for run_name in run_names:
    #                                     run_name_dir = os.path.join(training_runs_dir, run_name)
    #                                     run_output_model = os.path.join(run_name_dir, "output_model")
    #                                     if os.path.exists(run_output_model):
    #                                         run_output_model_names = os.listdir(run_output_model)
    #                                         for run_output_model_name in run_output_model_names:
    #                                             model_bin_path = os.path.exists(
    #                                                 os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs',
    #                                                              run_name, "output_model", run_output_model_name, "ori",
    #                                                              "pytorch_model.bin"))
    #                                             if run_output_model_name.find("merged_") >= 0 and model_bin_path:
    #                                                 tgi_model_format_runs_output_model.append(
    #                                                     os.path.join(run_name, "output_model", run_output_model_name, "ori"))

    #                                                 gptq_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs',run_name, "output_model", run_output_model_name, "quantized_gptq_"+run_output_model_name.split('_')[-1],
    #                                                              "pytorch_model.bin")
    #                                                 if os.path.exists(gptq_model_path):
    #                                                     tgi_model_format_runs_output_model.append(os.path.join(run_name, "output_model", run_output_model_name, "quantized_gptq_"+run_output_model_name.split('_')[-1]))
    #                                                 gguf_model_dir = os.path.join(
    #                                                     os.path.dirname(os.path.abspath(__file__)), 'runs', run_name,
    #                                                     "output_model", run_output_model_name,
    #                                                     "quantized_gguf_" + run_output_model_name.split('_')[-1])
    #                                                 if os.path.exists(gguf_model_dir):
    #                                                     gguf_model_names = os.listdir(gguf_model_dir)
    #                                                     for gguf_model_name in gguf_model_names:
    #                                                         if gguf_model_name.split('.')[-1] == "gguf":
    #                                                             gguf_model_format_runs_output_model.append(
    #                                                                 os.path.join(run_name, "output_model",
    #                                                                              run_output_model_name, "quantized_gguf_" +
    #                                                                              run_output_model_name.split('_')[-1],
    #                                                                              gguf_model_name))

    #                                 tgi_model_format_runs_output_model = tgi_model_format_runs_output_model[::-1]
    #                                 gguf_model_format_runs_output_model = gguf_model_format_runs_output_model[::-1]

    #                                 deployment_runs_output_model_dropdown = gr.Dropdown(tgi_model_format_runs_output_model, label="runs_output_model",
    #                                                                          value=tgi_model_format_runs_output_model[
    #                                                                              0] if tgi_model_format_runs_output_model else None,
    #                                                                          interactive=True,scale=6)
    #                                 refresh_deployment_runs_output_model_btn = gr.Button("Refresh", scale=1, visible=True)

    #                             if tgi_model_format_runs_output_model:
    #                                 model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs',
    #                                                          os.path.dirname(tgi_model_format_runs_output_model[0]))
    #                                 model_name = os.path.basename(tgi_model_format_runs_output_model[0])
    #                                 if model_name.rfind("quantized_gptq_") >= 0:
    #                                     run_server_value = f'''docker run --gpus all --shm-size 1g -p 8080:80 -v {model_dir}:/data ghcr.io/huggingface/text-generation-inference:latest --model-id /data/{model_name} --quantize gptq'''
    #                                 else:
    #                                     run_server_value = f'''docker run --gpus all --shm-size 1g -p 8080:80 -v {model_dir}:/data ghcr.io/huggingface/text-generation-inference:latest --model-id /data/{model_name}'''

    #                                 run_server_script_textbox = gr.Textbox(label="Run Server:", interactive=False,lines=2, scale=1,value=run_server_value)
    #                                 run_client_value = '''Command-Line Interface(CLI):\ncurl 127.0.0.1:8080/generate -X POST  -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":20}}' -H 'Content-Type: application/json'\n\nPython:\nfrom huggingface_hub import InferenceClient \nclient = InferenceClient(model="http://127.0.0.1:8080")\noutput = client.text_generation(prompt="What is Deep Learning?",max_new_tokens=512)
    #                                 '''
    #                                 run_client_script_textbox = gr.Textbox(label="Run Client:", interactive=False, lines=6,scale=1,value=run_client_value)
    #                             else:
    #                                 run_server_script_textbox = gr.Textbox(label="Run Server:", interactive=False,lines=2, scale=1,value="")
    #                                 run_client_script_textbox = gr.Textbox(label="Run Client:", interactive=False, lines=6,
    #                                                                        scale=1, value="")

    #                             # deploy_llm_code = gr.Code(code_str, language="shell", lines=5, label="Install Requirements:")
    #                             install_requirements_value = '''
    #                             ### &nbsp;&nbsp; 1.install docker
    #                             ### &nbsp;&nbsp; 2.Install NVIDIA Container Toolkit
    #                             <h4> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.1 Configure the repository: </h4>
    #                             <p> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    #   && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    #     sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    #     sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
    #   && \
    #     sudo apt-get update </p>
    #                                 <h4> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.2 Install the NVIDIA Container Toolkit packages: </h4>
    #                                 <p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; sudo apt-get install -y nvidia-container-toolkit </p>
    #                                 '''
    #                             with gr.Accordion("Install Requirements",open=False) as install_requirements_accordion:
    #                                 install_requirements_markdown = gr.Markdown(install_requirements_value)

    #                             run_llama_cpp_python_code = gr.Code("", language="python", lines=10, label="run_model_using_llama_cpp_python.py",visible=False)
    # run_script_textbox = gr.Textbox(label="Install Requirements:", interactive=False, scale=1,value=install_requirements_value)
    # dependencies

    ######### ACTIONS #########
    def load_base_model(base_model_name):
        torch.cuda.empty_cache()
        text_file_list = []
        if base_model_name!=adaptor.base_model_name:
            # Init the adaptor
            adaptor.base_model_ = None
            adaptor.tokenizer = None
            adaptor.lora_model_ = None
            try:
                gc.collect()
                torch.cuda.empty_cache()
            except:
                pass

            # Load the base model
            gr.update(visible=False)
            if fine_tuning_type_dropdown.value == 'QLoRA':
                adaptor.quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type='nf4',
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=False
                )
            else:
                adaptor.quant_config = None

            try:
                adaptor.init_tokenizer(base_model_name)
                adaptor.init_base_model(base_model_name, adaptor.quant_config)
            except ValueError as e:
                raise gr.Error(e)

            return gr.update(interactive=False), gr.update(interactive=True)
        else:
            return gr.update(interactive=True), gr.update(interactive=True)


    def load_dataset():
        try:
            train_dataset = {'prediction':[]}
            for file_path in text_file_list:
                with open(file_path, 'rt', encoding='utf8') as f:
                    for text in f.readlines():
                        train_dataset['prediction'].append(text)

            return Dataset.from_dict(train_dataset)
        except:
            raise ValueError("Dataset loading failure")

    def check_training_args():
        return TrainingArguments(
            output_dir=os.path.join(base_lora_path, adaptor.base_model_name, lora_output_path) if lora_output_path!='' else os.path.join(base_lora_path, adaptor.base_model_name, 'lora'),
            num_train_epochs=epochs_slider.value,
            per_device_train_batch_size=batch_size_slider.value,
            gradient_accumulation_steps=gradient_accumulation_steps_slider.value,
            optim=optimizer_dropdown.value,
            save_steps=25,
            logging_steps=25,
            learning_rate=learning_rate_slider.value,
            weight_decay=0.001,
            fp16=True,
            bf16=False,
            max_grad_norm=0.3,
            max_steps=-1,
            warmup_steps=warmup_steps_slider.value,
            # warmup_ratio=0.03,
            lr_scheduler_type=lr_scheduler_type_dropdown.value,
            remove_unused_columns=False
        )

    # def base_model_change(base_model_name):
    #     # adaptor.base_model_name = base_model_name
    #     return gr.update(interactive=True), gr.update(interactive=True)

    def train_lora():

        # Init tokenizer
        # adaptor.init_tokenizer(base_model_name_dropdown)

        # Init base model
        try:
            load_base_model(adaptor.base_model_name)
        except ValueError as e:
            gr.Error(e)

        # Collect training arguments
        adaptor.training_params = check_training_args()

        # Collect lora params
        if adaptor.use_quant:
            adaptor.config_lora(
                alpha=lora_alpha_slider.value,
                dropout=lora_dropout_slider.value,
                r=lora_r_slider.value,
                bias=lora_bias_dropdown.value,
                task_type='CASUAL_LM'
            )
        else:
            adaptor.lora_config=None
        # Init lora adaptor
        try:
            adaptor.init_lora(adaptor.lora_config)
        except ValueError as e:
            raise gr.Error(e)

        # Load dataset from text files
        try:
            adaptor.train_data = load_dataset()
        except ValueError as e:
            raise gr.Error(e)

        # Train the lora adaptor
        try:
            adaptor.train()
        except ValueError as e:
            raise gr.Error(e)

        # Release GPU memory
        adaptor.lora_model_ = None
        try:
            gc.collect()
            torch.cuda.empty_cache()
        except:
            pass

        return gr.update(interactive=True)

    def load_adaptor(adaptor_name, use_quant=True):
        if adaptor.base_model_name != os.path.split(adaptor_name)[0] or adaptor.use_quant!=use_quant:
            adaptor.tokenizer=None
            adaptor.base_model_=None
            adaptor.lora_model_=None
            try:
                gc.collect()
                torch.cuda.empty_cache()
            except:
                pass
            if use_quant:
                adaptor.quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type='nf4',
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=False
                )
            else:
                adaptor.quant_config = None
            adaptor.use_quant=use_quant
            adaptor.init_tokenizer(os.path.split(adaptor_name)[0])
            adaptor.init_base_model(os.path.split(adaptor_name)[0], adaptor.quant_config)
        adaptor.load_adaptor(os.path.join(base_lora_path, adaptor_name))
        return gr.update(interactive=False), gr.update(interactive=True), gr.update(interactive=True)

    def reload_lora_list():
        lora_list = list_adaptors(base_lora_path)
        if lora_list:
            new_lora_list = [lora_path[len(base_lora_path) + 1:] for lora_path in lora_list]
            return gr.update(choices=new_lora_list, value=new_lora_list[0]), gr.update(interactive=True)
        else:
            return gr.update(choices=[]), gr.update(interactive=True)

    def generate_text(prompt):
        output = adaptor.generate(prompt)
        return gr.update(value=output)

    def lora_dropdown_change():
        return gr.update(interactive=True)

    def update_lora_output_path(text):
        global lora_output_path
        lora_output_path = text


    def enable_train_btn():
        return gr.update(interactive=True)

    # base_model_name_dropdown.select(base_model_change, inputs=[base_model_name_dropdown], outputs=[load_base_model_btn, train_btn])
    load_base_model_btn.click(load_base_model, inputs=[base_model_name_dropdown], outputs=[load_base_model_btn, base_model_name_dropdown])
    train_btn.click(train_lora, inputs=[], outputs=[train_btn])
    refresh_lora_list_btn.click(reload_lora_list, inputs=[], outputs=[local_lora_dropdown, load_lora_btn])
    load_lora_btn.click(load_adaptor, inputs=[local_lora_dropdown, lora_4bit_quant_checkbox], outputs=[load_lora_btn, generate_text_btn, local_lora_dropdown])
    generate_text_btn.click(generate_text, inputs=[test_input_textbox], outputs=[output_textbox])
    local_lora_dropdown.change(lora_dropdown_change, inputs=[], outputs=[load_lora_btn])
    adaptor_output_path_textbox.change(update_lora_output_path, inputs=[adaptor_output_path_textbox])
    local_train_data_files_textbox.change(enable_train_btn, inputs=[], outputs=[train_btn])

demo.queue()
demo.launch(share=True)