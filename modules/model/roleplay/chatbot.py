import gradio as gr
import os
import pickle
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import shutil
from PIL import Image
from commune..model.roleplay.miner.miner import Roleplay
from transformers import BitsAndBytesConfig, TrainingArguments
import torch

with gr.Blocks(
        title="Roleplay Chatbot",
        css="#vertical_center_align_markdown { position:absolute; top:30%;background-color:white;} .white_background {background-color: #ffffff} .none_border {border: none;border-collapse:collapse;}"
) as demo:

    # Init the chat model
    char_bot = Roleplay()
    base_model_name = 'TheBloke/MythoLogic-13B-GPTQ'
    char_bot.init_tokenizer(base_model_name)
    char_bot.quant_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type='nf4',
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=False
                    )
    char_bot.init_base_model(base_model_name, char_bot.quant_config)
    char_bot.load_adaptor('')

    with gr.Tab("Character Configuration"):
        char_conf = {}

        def update_char_list():
            char_list = []
            try:
                file_list = os.listdir('chars')
                for file in file_list:
                    if file.startswith('char') and file.endswith('.char'):
                        char_list.append(file[5:-5])
            except:
                char_list = []
            return char_list

        base_model_name_dropdown = gr.Dropdown(
            ['TheBloke/MythoLogic-13B-GPTQ'],
            label="Base Model Name",
            value='TheBloke/MythoLogic-13B-GPTQ',
            interactive=True,
            visible=True,
            allow_custom_value=True
        )
        char_list = update_char_list()

        char_list_dropdown = gr.Dropdown(
            char_list,
            label="Character list",
            value=None,
            interactive=True,
            visible=True,
            scale=5,
            allow_custom_value=True
        )

        char_name_textbox = gr.Textbox(
            label='Character Name',
            interactive=True,
            visible=True
        )

        char_avatar_imgbox = gr.Image(
            value=None,
            label='Character Avatar',
            height=200,
            sources=['upload']
        )

        gr.Interface(
            fn=None,
            inputs=char_avatar_imgbox,
            outputs=None
        )

        char_desc_textbox = gr.Textbox(
            label='Short description for the character',
            lines=4,
            max_lines=8,
            interactive=True,
            visible=True
        )

        char_persona_textbox = gr.Textbox(
            label='Character Persona',
            lines=10,
            max_lines=12,
            interactive=True,
            visible=True
        )

        char_init_msg_textbox = gr.Textbox(
            label='Initial Message',
            lines=2,
            max_lines=4,
            interactive=True,
            visible=True
        )

        char_add_prompt_textbox = gr.Textbox(
            label='Additional Prompt',
            lines=4,
            max_lines=8,
            interactive=True,
            visible=True
        )

        with gr.Group():

            char_knowledgebase_label_textbox = gr.Textbox(
                value='You can upload knowledge base file for the character',
                interactive=False
            )
            with gr.Row(equal_height=True):
                char_knowledgebase_file = gr.File(
                    label='Knowledgebase file',
                    file_count='single',
                    interactive=True,
                    scale=1
                )

                char_knowbase_textbox = gr.Textbox(
                    value='',
                    label='Knowledge base content',
                    interactive=False,
                    lines=4,
                    max_lines=10,
                    scale=4
                )

        char_gen_button = gr.Button(
            'Generate the Character'
        )

    def chat_with_char(message, history, init_msg):
        # if char_bot.init_conv_flag:
        #     history = []
        #     char_bot.init_conv_flag = False

        if history==[]:
            history.append({'role': 'user', 'content': None})
            if init_msg =='':
                history.append({'role': 'character', 'content': None})
            else:
                history.append({'role': 'character', 'content': init_msg})

        messages = [(history[i]['content'], history[i + 1]['content']) for i in range(0, len(history) - 1, 2)]
        response = char_bot.generate(history, message)
        # prompt = char_bot.gen_prompt(char_bot.char_name, char_bot.user_name, char_bot.persona, str(messages))
        # response = char_bot.generate(message)#prompt)

        history.append({'role': 'user', 'content':message})
        history.append({'role': 'character', 'content':response})
        messages = [(history[i]['content'], history[i+1]['content']) for i in range (0, len(history) - 1, 2)]

        return messages, gr.update(value=history), gr.update(value="", autofocus=True)

    with gr.Tab("Chat with Characters"):
        char_list = update_char_list()
        with gr.Row():
            curr_char_list_dropdown = gr.Dropdown(
                char_list,
                label="Character list",
                value=None,
                interactive=True,
                visible=True,
                scale=5,
                allow_custom_value=True
            )

            curr_char_list_refresh_btn = gr.Button(
                value='🔄',
                scale=1,
                interactive=True
            )

        with gr.Row(equal_height=True):
            curr_char_avatar_imgbox = gr.Image(
                label='Character Avatar',
                scale=1,
                sources=['upload'],
                interactive=False
            )
            with gr.Column(scale=2):
                curr_char_name_textbox = gr.Textbox(
                    label='Character Name',
                    interactive=False,
                    max_lines=1
                )
                curr_char_init_msg_textbox = gr.Textbox(
                    label='Initial Message',
                    interactive=False,
                    lines=5,
                    max_lines=5
                )

            curr_char_desc_textbox = gr.Textbox(
                label='Brief description of the character',
                interactive=False,
                scale=2,
                lines=8,
                max_lines=8
            )

        with gr.Row(equal_height=True):
            curr_char_persona_textbox = gr.Textbox(
                label='Character Persona',
                interactive=False,
                lines=12,
                max_lines=12
            )
            curr_char_add_prompt_textbox = gr.Textbox(
                label = 'Additional Prompt',
                interactive=False,
                lines=12,
                max_lines=12
            )
            curr_char_knowledgebase_textbox = gr.Textbox(
                label='Knowledge base',
                lines=12,
                max_lines=12,
                interactive=False
            )
        with gr.Blocks() as chatbox:
            chatbox_chatbot = gr.Chatbot(
                label='Chat with Character',
                height=600,
                bubble_full_width=False
            )
            chatbox_state = gr.State([])
            with gr.Row():
                chatbox_textbox = gr.Textbox(show_label=False, placeholder='Type in the text and press enter')
            chatbox_textbox.submit(
                chat_with_char,
                [chatbox_textbox, chatbox_state, curr_char_init_msg_textbox],
                [chatbox_chatbot, chatbox_state, chatbox_textbox]
            )
        # chatbox_chatbot = gr.Chatbot(label='Character')
        # chatbox_textbox = gr.Textbox(label='Type in')
        # chatbox_ci = gr.ChatInterface(chat_with_char, chatbot=chatbox_chatbot, textbox=chatbox_textbox)



    def dump_character(name, desc, persona, init_msg, knowledgebase_path, avatar_img, add_prompt):
        char_info = {
            'name':name,
            'desc':desc,
            'persona':persona,
            'init_msg':init_msg,
            'knowbase_path':knowledgebase_path,
            'additional_prompt': add_prompt
        }

        if os.path.exists(os.path.join('chars', f'char-{name}.char')):
            raise gr.Warning("Character already exists, updating it")
        os.makedirs('chars', exist_ok=True)
        np.save(os.path.join('chars', f'char-{name}-avatar.npy'), avatar_img)
        with open(os.path.join('chars', f'char-{name}.char'), 'wb') as f:
            pickle.dump(char_info, f)

        if knowledgebase_path:
            shutil.copy(knowledgebase_path, os.path.join('chars', f'char-{name}-kb.txt'))

        char_list = update_char_list()
        return gr.update(interactive=True), gr.update(choices=char_list)


    def update_char_info(char_name):
        if not os.path.exists(os.path.join('chars', f'char-{char_name}.char')):
            raise gr.Error("Character does not exist")
        else:
            if os.path.exists(os.path.join('chars', f'char-{char_name}.char')):
                with open(os.path.join('chars', f'char-{char_name}.char'), 'rb') as f:
                    char_info = pickle.load(f)
                char_avatar = np.load(os.path.join('chars', f'char-{char_name}-avatar.npy'))
            else:
                char_avatar = None
            if os.path.exists(os.path.join('chars', f'char-{char_name}-kb.txt')):
                with open(os.path.join('chars', f'char-{char_name}-kb.txt'), 'rt') as f:
                    knowbase_content = f.read()
            else:
                knowbase_content = ''

        return gr.update(value=char_info['name']), \
               gr.update(value=char_info['desc']), \
               gr.update(value=char_info['persona']), \
               gr.update(value=char_info['init_msg']), \
               gr.update(value=char_avatar), \
               gr.update(value=knowbase_content), gr.update(value=char_info['additional_prompt'])

    def show_knowledgebase(knowbase_path):
        with open(knowbase_path, 'rt') as f:
            content = f.read()
        return gr.update(value=content)


    def load_char_info(char_name):
        char_avatar = None
        init_msg = ''
        char_desc = ''
        char_persona = ''
        char_knowbase = ''
        try:
            if os.path.exists(os.path.join('chars', f'char-{char_name}.char')):
                with open(os.path.join('chars', f'char-{char_name}.char'), 'rb') as f:
                    char_info = pickle.load(f)
                    init_msg = char_info['init_msg']
                    char_desc = char_info['desc']
                    char_persona = char_info['persona']
                    char_add_prompt = char_info['additional_prompt']

                    char_bot.char_name = char_name
                    char_bot.persona = char_persona
                    char_bot.user_name = 'user'
                    char_bot.set_persona()
                    char_bot.set_knowbase(char_persona)
                    char_bot.build_conv_chain()
                    char_bot.init_conv_flag = True

            if os.path.exists(os.path.join('chars', f'char-{char_name}-avatar.npy')):
                char_avatar = np.load(os.path.join('chars', f'char-{char_name}-avatar.npy'))
                avatar = Image.fromarray(char_avatar)
                avatar.save('avatar.jpg')
            else:
                try:
                    shutil.copy('imgs/default_avatar.jpg', 'avatar.jpg')
                except:
                    pass

            if os.path.exists(os.path.join('chars', f'char-{char_name}-kb.txt')):
                with open(os.path.join('chars', f'char-{char_name}-kb.txt'), 'rt') as f:
                    char_knowbase = f.read()

            char_list = update_char_list()
            return gr.update(value=char_avatar), gr.update(value=char_name), gr.update(value=init_msg), \
                   gr.update(value=char_desc), gr.update(value=char_persona), gr.update(value=char_knowbase), \
                   gr.update(value=[[(None),(init_msg)]], avatar_images=('imgs/user.jpg', 'avatar.jpg')),\
                   gr.update(value=[(None, init_msg)]), gr.update(choices = char_list), \
                   gr.update(value=char_add_prompt, visible=True if char_add_prompt!='' else False)
        except:
            raise gr.Error("Character loading failed")

    char_gen_button.click(
        dump_character,
        inputs=[char_name_textbox, char_desc_textbox, char_persona_textbox,
                char_init_msg_textbox, char_knowledgebase_file,
                char_avatar_imgbox, char_add_prompt_textbox],
        outputs=[char_gen_button, char_list_dropdown]
    )

    char_knowledgebase_file.change(
        show_knowledgebase,
        inputs=[char_knowledgebase_file],
        outputs=[char_knowbase_textbox]
    )

    char_list_dropdown.change(
        update_char_info,
        inputs=[char_list_dropdown],
        outputs=[char_name_textbox, char_desc_textbox, char_persona_textbox,
                 char_init_msg_textbox, char_avatar_imgbox, char_knowbase_textbox, char_add_prompt_textbox]
    )

    def updat_curr_char_list():
        char_list = update_char_list()
        return gr.update(choices=char_list)

    curr_char_list_refresh_btn.click(
        updat_curr_char_list,
        inputs=[],
        outputs=[curr_char_list_dropdown]
    )

    curr_char_list_dropdown.change(
        load_char_info,
        inputs=[curr_char_list_dropdown],
        outputs=[curr_char_avatar_imgbox, curr_char_name_textbox,
                 curr_char_init_msg_textbox, curr_char_desc_textbox,
                 curr_char_persona_textbox, curr_char_knowledgebase_textbox,
                 chatbox_chatbot, chatbox_state, curr_char_list_dropdown,
                 curr_char_add_prompt_textbox]
    )


demo.queue()
demo.launch(share=True)