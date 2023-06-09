
import transformer

from transformers import AutoProcessor, SpeechT5ForTextToSpeech


class Text2Speech(c.Module):
        
    def __init__(self, model:str = "microsoft/speecht5_tts"):
        self.processor = AutoProcessor.from_pretrained(model)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(model)
        
    def forward(self, text):
        input_ids = self.processor(text, return_tensors="pt").input_ids
        logits = self.model(input_ids).logits
        return logits