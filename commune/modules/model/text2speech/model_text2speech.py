import commune as c
import torch
import torchaudio
from seamless_communication.models.inference import Translator

class ModelText2speech(c.Module):
    def __init__(self, config = None, **kwargs):
        self.set_config(config, kwargs=kwargs)
        self.translator = Translator("seamlessM4T_large", "vocoder_36langs", torch.device("cpu"), torch.float32)
        self.src_lang = 'eng'

    def call(self, x:int = 1, y:int = 2) -> int:
        c.print(self.config.sup)
        c.print(self.config, 'This is the config, it is a Munch object')
        return x + y

    def text2speech(self, text, src_format='t2st', target_lang='eng', src_lang="eng", output_file="output.wav"):
        translated_text, wav, sr = self.translator.predict(text, src_format, target_lang, src_lang)
        
        torchaudio.save(
            output_file,
            wav[0],
            sr,
        )

        c.print("Audio file created: ", output_file);
        return translated_text

    def speech2text(self, src_format='s2tt', target_lang='eng', src_lang="eng", inputFile="audio.wav"):
        translated_text, _, _ = self.translator.predict(inputFile, src_format, target_lang, src_lang)
        c.print("Result: ", translated_text)
        return translated_text

    def text2text(self, text, src_format='t2tt', target_lang='eng', src_lang="eng"):
        translated_text, _, _ = self.translator.predict(text, src_format, target_lang, src_lang)
        return translated_text
    