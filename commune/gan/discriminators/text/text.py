import commune
import torch
import torch.nn as nn
from typing import Union, List
from transformers import AutoTokenizer, ElectraForMultipleChoice


DEFAULT_TOKENIZER = "google/electra-base-generator"


class TextDiscriminator(nn.Module, commune.Module):

    def __init__(self,  
                 discriminator : str="google/electra-base-generator", 
                 tokenizer : str= None
                 ) -> None:
        super(TextDiscriminator, self).__init__()
        self.discriminator = ElectraForMultipleChoice.from_pretrained(discriminator)
        self.tokenizer = AutoTokenizer.from_pretrained(discriminator if tokenizer == None else tokenizer)

    def set_tokenizer(self, tokenizer : str):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    
    def set_discriminator(self, discriminator : str):
        self.discriminator = ElectraForMultipleChoice.from_pretrained(discriminator)

    def encode_text(self, 
                    prompt : Union[List[str], List[List[str]], str]=None, 
                    labels : Union[List[str], List[List[str]], str]=None,
                    **prams : dict[str, any]):
        
        if self.tokenizer == None:
            self.tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER)
        
        return self.tokenizer(prompt, labels, **prams)
    
    def tokenize_text(self, x : Union[List[str], str]):
            return self.tokenizer.tokenize(x, add_special_tokens=True, padding=True)



    def forward(self, 
                prompt : Union[List[str], str], 
                output : Union[List[str], str], 
                labels : Union[List[str], str], 
                **tokenizer_parameters : dict[str, any]):
        assert not isinstance(inputs, labels), "These must be the same type"
        
        inputs = self.encode_text(inputs, labels, **tokenizer_parameters)
        inputs["labels"] = labels

        return self.discriminator(**inputs)[0]

    __call__ = forward
    
    @classmethod
    def test_encoding(cls):
        model = cls()
        prams = dict(return_tensors= 'pt', max_length=512, truncation=True, padding='max_length')
        output = model.encode_text(["what was the fox doing ", "this is another example"], ["The quick brown fox jumps over the lazy dog", "this is another example"], **prams)
        print(output)

            # return isinstance(self, torch.Tensor) 
            
    @classmethod
    def test_tokenize(cls):
        model = cls()
        print(model.tokenize_text(["The quick brown fox jumps over the lazy dog", "The quick brown fox jumps over the lazy dog"], ))
        # return isinstance(self, torch.Tensor) 

    @classmethod
    def test_forward(cls):
        prams = dict(return_tensors= 'pt', max_length=512, truncation=True, padding='max_length')
        # with torch.no_grad():
        #     model = cls()
        #     print(torch.round(( 
        #                 torch.sign(
        #                         model(
        #                                 x=["What was the fox doing to the lazy dog"],
        #                                 labels=["The quick brown fox jumps over the lazy dog",
        #                                         "The quick brown fox forked over the lazy dog"]
        #                                 **prams
        #                             )
        #                             ) + 1
        #                     ) / 2 ))

# generative adversarial networks
if __name__ == "__main__":
    TextDiscriminator.run()