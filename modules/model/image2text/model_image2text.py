import commune as c
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration



class ModelImage2text(c.Module):
    def __init__(self, config = None, **kwargs):
        self.set_config(config, kwargs=kwargs)
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


    def call(self, x:int = 1, y:int = 2) -> int:
        c.print(self.config.sup)
        c.print(self.config, 'This is the config, it is a Munch object')
        return x + y

    def image2text(self, img_url='https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'):
        raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

        # conditional image captioning
        # text = "a photography of"
        # inputs = processor(raw_image, text, return_tensors="pt")

        # out = model.generate(**inputs)
        # print(processor.decode(out[0], skip_special_tokens=True))
        # >>> a photography of a woman and her dog

        # unconditional image captioning
        inputs = self.processor(raw_image, return_tensors="pt")

        out = self.model.generate(**inputs)
        print(self.processor.decode(out[0], skip_special_tokens=True))

    