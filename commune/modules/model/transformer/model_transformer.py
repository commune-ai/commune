from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import commune as c

class TransformerModel(c.Module):
    def __init__(self, config = None, **kwargs):
        self.set_config(config, kwargs=kwargs)

        self.config.model = self.config.shortcuts.get(self.config.model, self.config.model)

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.config.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=self.config.trust_remote_code,
            device_map=self.config.device_map,
            max_memory = self.config.max_memory
        )
        if self.config.half:
            self.pipeline.model = self.pipeline.model.half()




    def generate_text(self, 
                prompt, 
                 max_length=200, 
                 do_sample=False, 
                 top_k=10, 
                 num_return_sequences=1,
                 **kwargs):
        sequences = self.pipeline(
            prompt,
            max_length=max_length,
            do_sample=do_sample,
            top_k=top_k,
            num_return_sequences=num_return_sequences,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        return {'input': prompt,
                'output': [seq["generated_text"] for seq in sequences]}

    talk = chat = forward = generate_text


    def test(self):
        prompt = "Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.\nDaniel: Hello, Girafatron!\nGirafatron:"
        generated_texts = self.generate_text(prompt)
        for text in generated_texts:
            print(f"Result: {text}")


            
    @classmethod
    def serve(cls,
            model: str,
            tag = None,
            refresh = True,    
            **kwargs
            ):
        
        config = cls.get_config(kwargs=kwargs)
        config.tag = tag
        config.model = model
        c.print(config)
        c.serve(module=cls.module_path(),
                name= f'model.{model}',
                tag = tag,
                kwargs={'config': config},
                refresh = refresh,
                verbose=True, **kwargs)
        