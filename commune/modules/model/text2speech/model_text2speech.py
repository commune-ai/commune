import commune as c
import torch
import torchaudio
from seamless_communication.models.inference import Translator

import gradio as gr
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
translator = Translator(
    model_name_or_card="seamlessM4T_medium",
    vocoder_name_or_card="vocoder_36langs",
    device=device,
    dtype=torch.float16 if "cuda" in device.type else torch.float32,
)


class ModelText2speech(c.Module):
    def __init__(self, config = None, **kwargs):
        self.set_config(config, kwargs=kwargs)

    def call(self, x:int = 1, y:int = 2) -> int:
        c.print(self.config.sup)
        c.print(self.config, 'This is the config, it is a Munch object')
        return x + y

    def text2speech(self, text, target_lang='eng', src_lang="eng", output_file="output.wav"):
        translated_text, wav, sr = translator.predict(text, 't2st', target_lang, src_lang)
        
        torchaudio.save(
            output_file,
            wav[0],
            sr,
        )

        c.print("Audio file created: ", output_file);
        return translated_text

    def speech2text(self, target_lang='eng', src_lang="eng", inputFile="audio.wav"):
        translated_text, _, _ = translator.predict(inputFile, 's2tt', target_lang, src_lang)
        c.print("Result: ", translated_text)
        return translated_text

    def text2text(self, text, target_lang='eng', src_lang="eng"):
        translated_text, _, _ = translator.predict(text, "t2tt", target_lang, src_lang)
        return translated_text
    
    def speech2speech(self, target_lang='eng', src_lang="eng", output_file="output.wav", input_file="audio.wav"):
        translated_text, wav, sr = translator.predict(input_file, "s2st", target_lang, src_lang)
        
        torchaudio.save(
            output_file,
            wav[0],
            sr,
        )

        c.print("Audio file created: ", output_file);
        return translated_text

    def gradio(self):
        demo.queue().launch()


DESCRIPTION = """# Translation Test
"""

TASK_NAMES = [
    "S2ST (Speech to Speech translation)",
    "S2TT (Speech to Text translation)",
    "T2ST (Text to Speech translation)",
    "T2TT (Text to Text translation)",
    "ASR (Automatic Speech Recognition)",
]

# Language dict
language_code_to_name = {
    "afr": "Afrikaans",
    "amh": "Amharic",
    "arb": "Modern Standard Arabic",
    "ary": "Moroccan Arabic",
    "arz": "Egyptian Arabic",
    "asm": "Assamese",
    "ast": "Asturian",
    "azj": "North Azerbaijani",
    "bel": "Belarusian",
    "ben": "Bengali",
    "bos": "Bosnian",
    "bul": "Bulgarian",
    "cat": "Catalan",
    "ceb": "Cebuano",
    "ces": "Czech",
    "ckb": "Central Kurdish",
    "cmn": "Mandarin Chinese",
    "cym": "Welsh",
    "dan": "Danish",
    "deu": "German",
    "ell": "Greek",
    "eng": "English",
    "est": "Estonian",
    "eus": "Basque",
    "fin": "Finnish",
    "fra": "French",
    "gaz": "West Central Oromo",
    "gle": "Irish",
    "glg": "Galician",
    "guj": "Gujarati",
    "heb": "Hebrew",
    "hin": "Hindi",
    "hrv": "Croatian",
    "hun": "Hungarian",
    "hye": "Armenian",
    "ibo": "Igbo",
    "ind": "Indonesian",
    "isl": "Icelandic",
    "ita": "Italian",
    "jav": "Javanese",
    "jpn": "Japanese",
    "kam": "Kamba",
    "kan": "Kannada",
    "kat": "Georgian",
    "kaz": "Kazakh",
    "kea": "Kabuverdianu",
    "khk": "Halh Mongolian",
    "khm": "Khmer",
    "kir": "Kyrgyz",
    "kor": "Korean",
    "lao": "Lao",
    "lit": "Lithuanian",
    "ltz": "Luxembourgish",
    "lug": "Ganda",
    "luo": "Luo",
    "lvs": "Standard Latvian",
    "mai": "Maithili",
    "mal": "Malayalam",
    "mar": "Marathi",
    "mkd": "Macedonian",
    "mlt": "Maltese",
    "mni": "Meitei",
    "mya": "Burmese",
    "nld": "Dutch",
    "nno": "Norwegian Nynorsk",
    "nob": "Norwegian Bokm\u00e5l",
    "npi": "Nepali",
    "nya": "Nyanja",
    "oci": "Occitan",
    "ory": "Odia",
    "pan": "Punjabi",
    "pbt": "Southern Pashto",
    "pes": "Western Persian",
    "pol": "Polish",
    "por": "Portuguese",
    "ron": "Romanian",
    "rus": "Russian",
    "slk": "Slovak",
    "slv": "Slovenian",
    "sna": "Shona",
    "snd": "Sindhi",
    "som": "Somali",
    "spa": "Spanish",
    "srp": "Serbian",
    "swe": "Swedish",
    "swh": "Swahili",
    "tam": "Tamil",
    "tel": "Telugu",
    "tgk": "Tajik",
    "tgl": "Tagalog",
    "tha": "Thai",
    "tur": "Turkish",
    "ukr": "Ukrainian",
    "urd": "Urdu",
    "uzn": "Northern Uzbek",
    "vie": "Vietnamese",
    "xho": "Xhosa",
    "yor": "Yoruba",
    "yue": "Cantonese",
    "zlm": "Colloquial Malay",
    "zsm": "Standard Malay",
    "zul": "Zulu",
}
LANGUAGE_NAME_TO_CODE = {v: k for k, v in language_code_to_name.items()}

# Source langs: S2ST / S2TT / ASR don't need source lang
# T2TT / T2ST use this
text_source_language_codes = [
    "afr",
    "amh",
    "arb",
    "ary",
    "arz",
    "asm",
    "azj",
    "bel",
    "ben",
    "bos",
    "bul",
    "cat",
    "ceb",
    "ces",
    "ckb",
    "cmn",
    "cym",
    "dan",
    "deu",
    "ell",
    "eng",
    "est",
    "eus",
    "fin",
    "fra",
    "gaz",
    "gle",
    "glg",
    "guj",
    "heb",
    "hin",
    "hrv",
    "hun",
    "hye",
    "ibo",
    "ind",
    "isl",
    "ita",
    "jav",
    "jpn",
    "kan",
    "kat",
    "kaz",
    "khk",
    "khm",
    "kir",
    "kor",
    "lao",
    "lit",
    "lug",
    "luo",
    "lvs",
    "mai",
    "mal",
    "mar",
    "mkd",
    "mlt",
    "mni",
    "mya",
    "nld",
    "nno",
    "nob",
    "npi",
    "nya",
    "ory",
    "pan",
    "pbt",
    "pes",
    "pol",
    "por",
    "ron",
    "rus",
    "slk",
    "slv",
    "sna",
    "snd",
    "som",
    "spa",
    "srp",
    "swe",
    "swh",
    "tam",
    "tel",
    "tgk",
    "tgl",
    "tha",
    "tur",
    "ukr",
    "urd",
    "uzn",
    "vie",
    "yor",
    "yue",
    "zsm",
    "zul",
]
TEXT_SOURCE_LANGUAGE_NAMES = sorted(
    [language_code_to_name[code] for code in text_source_language_codes]
)

# Target langs:
# S2ST / T2ST
s2st_target_language_codes = [
    "eng",
    "arb",
    "ben",
    "cat",
    "ces",
    "cmn",
    "cym",
    "dan",
    "deu",
    "est",
    "fin",
    "fra",
    "hin",
    "ind",
    "ita",
    "jpn",
    "kor",
    "mlt",
    "nld",
    "pes",
    "pol",
    "por",
    "ron",
    "rus",
    "slk",
    "spa",
    "swe",
    "swh",
    "tel",
    "tgl",
    "tha",
    "tur",
    "ukr",
    "urd",
    "uzn",
    "vie",
]
S2ST_TARGET_LANGUAGE_NAMES = sorted(
    [language_code_to_name[code] for code in s2st_target_language_codes]
)
# S2TT / ASR
S2TT_TARGET_LANGUAGE_NAMES = TEXT_SOURCE_LANGUAGE_NAMES
# T2TT
T2TT_TARGET_LANGUAGE_NAMES = TEXT_SOURCE_LANGUAGE_NAMES

AUDIO_SAMPLE_RATE = 16000.0
MAX_INPUT_AUDIO_LENGTH = 60  # in seconds
DEFAULT_TARGET_LANGUAGE = "French"


def predict(
    task_name: str,
    audio_source: str,
    input_audio_mic: str | None,
    input_audio_file: str | None,
    input_text: str | None,
    source_language: str | None,
    target_language: str,
) -> tuple[tuple[int, np.ndarray] | None, str]:
    task_name = task_name.split()[0]
    source_language_code = (
        LANGUAGE_NAME_TO_CODE[source_language] if source_language else None
    )
    target_language_code = LANGUAGE_NAME_TO_CODE[target_language]

    if task_name in ["S2ST", "S2TT", "ASR"]:
        if audio_source == "microphone":
            input_data = input_audio_mic
        else:
            input_data = input_audio_file

        arr, org_sr = torchaudio.load(input_data)
        new_arr = torchaudio.functional.resample(
            arr, orig_freq=org_sr, new_freq=AUDIO_SAMPLE_RATE
        )
        max_length = int(MAX_INPUT_AUDIO_LENGTH * AUDIO_SAMPLE_RATE)
        if new_arr.shape[1] > max_length:
            new_arr = new_arr[:, :max_length]
            gr.Warning(
                f"Input audio is too long. Only the first {MAX_INPUT_AUDIO_LENGTH} seconds is used."
            )
        torchaudio.save(input_data, new_arr, sample_rate=int(AUDIO_SAMPLE_RATE))
    else:
        input_data = input_text
    text_out, wav, sr = translator.predict(
        input=input_data,
        task_str=task_name,
        tgt_lang=target_language_code,
        src_lang=source_language_code,
        ngram_filtering=True,
    )
    if task_name in ["S2ST", "T2ST"]:
        return (sr, wav.cpu().detach().numpy()), text_out
    else:
        return None, text_out


def process_s2st_example(
    input_audio_file: str, target_language: str
) -> tuple[tuple[int, np.ndarray] | None, str]:
    return predict(
        task_name="S2ST",
        audio_source="file",
        input_audio_mic=None,
        input_audio_file=input_audio_file,
        input_text=None,
        source_language=None,
        target_language=target_language,
    )


def process_s2tt_example(
    input_audio_file: str, target_language: str
) -> tuple[tuple[int, np.ndarray] | None, str]:
    return predict(
        task_name="S2TT",
        audio_source="file",
        input_audio_mic=None,
        input_audio_file=input_audio_file,
        input_text=None,
        source_language=None,
        target_language=target_language,
    )


def process_t2st_example(
    input_text: str, source_language: str, target_language: str
) -> tuple[tuple[int, np.ndarray] | None, str]:
    return predict(
        task_name="T2ST",
        audio_source="",
        input_audio_mic=None,
        input_audio_file=None,
        input_text=input_text,
        source_language=source_language,
        target_language=target_language,
    )


def process_t2tt_example(
    input_text: str, source_language: str, target_language: str
) -> tuple[tuple[int, np.ndarray] | None, str]:
    return predict(
        task_name="T2TT",
        audio_source="",
        input_audio_mic=None,
        input_audio_file=None,
        input_text=input_text,
        source_language=source_language,
        target_language=target_language,
    )


def process_asr_example(
    input_audio_file: str, target_language: str
) -> tuple[tuple[int, np.ndarray] | None, str]:
    return predict(
        task_name="ASR",
        audio_source="file",
        input_audio_mic=None,
        input_audio_file=input_audio_file,
        input_text=None,
        source_language=None,
        target_language=target_language,
    )


def update_audio_ui(audio_source: str) -> tuple[dict, dict]:
    mic = audio_source == "microphone"
    return (
        gr.update(visible=mic, value=None),  # input_audio_mic
        gr.update(visible=not mic, value=None),  # input_audio_file
    )


def update_input_ui(task_name: str) -> tuple[dict, dict, dict, dict]:
    task_name = task_name.split()[0]
    if task_name == "S2ST":
        return (
            gr.update(visible=True),  # audio_box
            gr.update(visible=False),  # input_text
            gr.update(visible=False),  # source_language
            gr.update(
                visible=True,
                choices=S2ST_TARGET_LANGUAGE_NAMES,
                value=DEFAULT_TARGET_LANGUAGE,
            ),  # target_language
        )
    elif task_name == "S2TT":
        return (
            gr.update(visible=True),  # audio_box
            gr.update(visible=False),  # input_text
            gr.update(visible=False),  # source_language
            gr.update(
                visible=True,
                choices=S2TT_TARGET_LANGUAGE_NAMES,
                value=DEFAULT_TARGET_LANGUAGE,
            ),  # target_language
        )
    elif task_name == "T2ST":
        return (
            gr.update(visible=False),  # audio_box
            gr.update(visible=True),  # input_text
            gr.update(visible=True),  # source_language
            gr.update(
                visible=True,
                choices=S2ST_TARGET_LANGUAGE_NAMES,
                value=DEFAULT_TARGET_LANGUAGE,
            ),  # target_language
        )
    elif task_name == "T2TT":
        return (
            gr.update(visible=False),  # audio_box
            gr.update(visible=True),  # input_text
            gr.update(visible=True),  # source_language
            gr.update(
                visible=True,
                choices=T2TT_TARGET_LANGUAGE_NAMES,
                value=DEFAULT_TARGET_LANGUAGE,
            ),  # target_language
        )
    elif task_name == "ASR":
        return (
            gr.update(visible=True),  # audio_box
            gr.update(visible=False),  # input_text
            gr.update(visible=False),  # source_language
            gr.update(
                visible=True,
                choices=S2TT_TARGET_LANGUAGE_NAMES,
                value=DEFAULT_TARGET_LANGUAGE,
            ),  # target_language
        )
    else:
        raise ValueError(f"Unknown task: {task_name}")


def update_output_ui(task_name: str) -> tuple[dict, dict]:
    task_name = task_name.split()[0]
    if task_name in ["S2ST", "T2ST"]:
        return (
            gr.update(visible=True, value=None),  # output_audio
            gr.update(value=None),  # output_text
        )
    elif task_name in ["S2TT", "T2TT", "ASR"]:
        return (
            gr.update(visible=False, value=None),  # output_audio
            gr.update(value=None),  # output_text
        )
    else:
        raise ValueError(f"Unknown task: {task_name}")


def update_example_ui(task_name: str) -> tuple[dict, dict, dict, dict, dict]:
    task_name = task_name.split()[0]
    return (
        gr.update(visible=task_name == "S2ST"),  # s2st_example_row
        gr.update(visible=task_name == "S2TT"),  # s2tt_example_row
        gr.update(visible=task_name == "T2ST"),  # t2st_example_row
        gr.update(visible=task_name == "T2TT"),  # t2tt_example_row
        gr.update(visible=task_name == "ASR"),  # asr_example_row
    )


css = """
h1 {
  text-align: center;
}

.contain {
  max-width: 730px;
  margin: auto;
  padding-top: 1.5rem;
}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Group():
        task_name = gr.Dropdown(
            label="Task",
            choices=TASK_NAMES,
            value=TASK_NAMES[0],
        )
        with gr.Row():
            source_language = gr.Dropdown(
                label="Source language",
                choices=TEXT_SOURCE_LANGUAGE_NAMES,
                value="English",
                visible=False,
            )
            target_language = gr.Dropdown(
                label="Target language",
                choices=S2ST_TARGET_LANGUAGE_NAMES,
                value=DEFAULT_TARGET_LANGUAGE,
            )
        with gr.Row() as audio_box:
            audio_source = gr.Radio(
                label="Audio source",
                choices=["file", "microphone"],
                value="file",
            )
            input_audio_mic = gr.Audio(
                label="Input speech",
                type="filepath",
                source="microphone",
                visible=False,
            )
            input_audio_file = gr.Audio(
                label="Input speech",
                type="filepath",
                source="upload",
                visible=True,
            )
        input_text = gr.Textbox(label="Input text", visible=False)
        btn = gr.Button("Translate")
        with gr.Column():
            output_audio = gr.Audio(
                label="Translated speech",
                autoplay=False,
                streaming=False,
                type="numpy",
            )
            output_text = gr.Textbox(label="Translated text")

    # with gr.Row(visible=True) as s2st_example_row:
    #     s2st_examples = gr.Examples(
    #         examples=[
    #             ["assets/sample_input.mp3", "French"],
    #             ["assets/sample_input.mp3", "Mandarin Chinese"],
    #             ["assets/sample_input_2.mp3", "Hindi"],
    #             ["assets/sample_input_2.mp3", "Spanish"],
    #         ],
    #         inputs=[input_audio_file, target_language],
    #         outputs=[output_audio, output_text],
    #         fn=process_s2st_example,
    #     )
    # with gr.Row(visible=False) as s2tt_example_row:
    #     s2tt_examples = gr.Examples(
    #         examples=[
    #             ["assets/sample_input.mp3", "French"],
    #             ["assets/sample_input.mp3", "Mandarin Chinese"],
    #             ["assets/sample_input_2.mp3", "Hindi"],
    #             ["assets/sample_input_2.mp3", "Spanish"],
    #         ],
    #         inputs=[input_audio_file, target_language],
    #         outputs=[output_audio, output_text],
    #         fn=process_s2tt_example,
    #     )
    with gr.Row(visible=False) as t2st_example_row:
        t2st_examples = gr.Examples(
            examples=[
                ["My favorite animal is the elephant.", "English", "French"],
                ["My favorite animal is the elephant.", "English", "Mandarin Chinese"],
                [
                    "Meta AI's Seamless M4T model is democratising spoken communication across language barriers",
                    "English",
                    "Hindi",
                ],
                [
                    "Meta AI's Seamless M4T model is democratising spoken communication across language barriers",
                    "English",
                    "Spanish",
                ],
            ],
            inputs=[input_text, source_language, target_language],
            outputs=[output_audio, output_text],
            fn=process_t2st_example,
        )
    with gr.Row(visible=False) as t2tt_example_row:
        t2tt_examples = gr.Examples(
            examples=[
                ["My favorite animal is the elephant.", "English", "French"],
                ["My favorite animal is the elephant.", "English", "Mandarin Chinese"],
                [
                    "Meta AI's Seamless M4T model is democratising spoken communication across language barriers",
                    "English",
                    "Hindi",
                ],
                [
                    "Meta AI's Seamless M4T model is democratising spoken communication across language barriers",
                    "English",
                    "Spanish",
                ],
            ],
            inputs=[input_text, source_language, target_language],
            outputs=[output_audio, output_text],
            fn=process_t2tt_example,
        )
    with gr.Row(visible=False) as asr_example_row:
        asr_examples = gr.Examples(
            examples=[
                ["assets/sample_input.mp3", "English"],
                ["assets/sample_input_2.mp3", "English"],
            ],
            inputs=[input_audio_file, target_language],
            outputs=[output_audio, output_text],
            fn=process_asr_example,
        )

    audio_source.change(
        fn=update_audio_ui,
        inputs=audio_source,
        outputs=[
            input_audio_mic,
            input_audio_file,
        ],
        queue=False,
        api_name=False,
    )
    task_name.change(
        fn=update_input_ui,
        inputs=task_name,
        outputs=[
            audio_box,
            input_text,
            source_language,
            target_language,
        ],
        queue=False,
        api_name=False,
    ).then(
        fn=update_output_ui,
        inputs=task_name,
        outputs=[output_audio, output_text],
        queue=False,
        api_name=False,
    ).then(
        fn=update_example_ui,
        inputs=task_name,
        outputs=[
            # s2st_example_row,
            # s2tt_example_row,
            t2st_example_row,
            t2tt_example_row,
            asr_example_row,
        ],
        queue=False,
        api_name=False,
    )

    btn.click(
        fn=predict,
        inputs=[
            task_name,
            audio_source,
            input_audio_mic,
            input_audio_file,
            input_text,
            source_language,
            target_language,
        ],
        outputs=[output_audio, output_text],
        api_name="run",
    )

if __name__ == "__main__":
    demo.queue().launch()
