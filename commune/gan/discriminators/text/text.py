import commune
import torch
import torch.nn as nn
from typing import Union, List
from transformers import AutoTokenizer, ElectraForPreTraining


DEFAULT_TOKENIZER = "google/electra-base-generator"


class TextDiscriminator(commune.Module):

    def __init__(self,  
                 discriminator : str="google/electra-base-generator", 
                 tokenizer : str= None
                 ) -> None:
        super(TextDiscriminator, self).__init__()
        self.discriminator = ElectraForPreTraining.from_pretrained(discriminator)
        self.tokenizer = AutoTokenizer.from_pretrained(discriminator if tokenizer == None else tokenizer)

    def set_tokenizer(self, tokenizer : str):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    
    def set_discriminator(self, discriminator : str):
        self.discriminator = ElectraForPreTraining.from_pretrained(discriminator)

    def encode_text(self, x : Union[List[str], List[List[str]], str], *args, **kwargs):
        if self.tokenizer == None:
            self.tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER)

        return self.tokenizer.batch_encode_plus(x, return_tensors="pt", padding=True, truncation=True)

    def tokenize_text(self, x : Union[List[str], str]):
            return self.tokenizer.tokenize(x, add_special_tokens=True, padding=True)

    def forward(self, x : Union[List[str], str, torch.Tensor], *args, **kwargs):
        if not isinstance(x, torch.Tensor):
            x = self.encode_text(x)
        return self.discriminator(**x)[0]

    __call__ = forward
    
    @classmethod
    def test_encoding(cls):
        model = cls()
        print(model.encode_text(["The quick brown fox jumps over the lazy dog", "this is another example"]))
            # return isinstance(self, torch.Tensor) 
            
    @classmethod
    def test_tokenize(cls):
        model = cls()
        print(model.tokenize_text(["The quick brown fox jumps over the lazy dog", "The quick brown fox jumps over the lazy dog"], ))
        # return isinstance(self, torch.Tensor) 

    @classmethod
    def test_forward(cls):
        with torch.no_grad():
            model = cls()
            print(torch.round(( 
                        torch.sign(
                                model(
                                        ["The quick brown fox jumps over the lazy dog", 
                                        "The quick brown fox jumps over the lazy dog"]
                                    )
                                    ) + 1
                            ) / 2 ))

# generative adversarial networks
if __name__ == "__main__":
    TextDiscriminator.run()