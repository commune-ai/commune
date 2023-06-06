
import commune
from typing import Union, List
from transformers import AutoTokenizer, ElectraForMaskedLM
import torch
from datasets import load_dataset

class TextGenerator(commune.Module):
    
    def __init__(self,
                 generator : str="google/electra-small-generator",
                 tokenizer : Union[str, None]=None,
                 dataset   : Union[List[str], str, None]="squad-v2"
                ) -> None:
        super(TextGenerator, self).__init__()
        self.generator = ElectraForMaskedLM.from_pretrained(generator)
        self.tokenizer = AutoTokenizer.from_pretrained(generator if tokenizer == None else tokenizer )
        self.dataset = load_dataset(dataset)

    def set_generator(self, generator : str) -> None:
        self.generator = ElectraForMaskedLM.from_pretrained(generator)

    def set_tokenizer(self, tokenizer : str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    def tokenize_text(self, x : Union[List[str], str]) -> torch.Tensor:
        return self.tokenizer(x, return_tensors="pt")

    def mask_text(self, x):
        ...



    def forward(self, x : Union[List[str], str, torch.Tensor]):
        if isinstance(x, torch.Tensor):
            x : torch.Tensor = self.tokenize_text(x)
        
        return self.generator(**x)
    
    __call__ = forward

    @classmethod
    def test_tokenizer(cls):
        model = cls(
            "google/electra-small-generator"
            )

        print(model.tokenize_text("Hello world how are [MASK]"))
    
    @classmethod
    def test_model(cls):
        model = cls(
            "google/electra-small-generator"
            )
        inputs = model.tokenize_text("Hello world how are [MASK]")
        with torch.no_grad():
            logits = model(inputs).logits
        
        mask_token_index = (inputs.input_ids == model.tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
        predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
        
        labels = model.tokenizer("Hello world how are you", return_tensors="pt")["input_ids"]
        labels = torch.where(inputs.input_ids == model.tokenizer.mask_token_id, labels, -100)
        outputs = model.generator(**inputs, labels=labels)
        print(round(outputs.loss.item(), 2), model.tokenizer.decode(predicted_token_id))

if __name__ == "__main__":
    TextGenerator.run()