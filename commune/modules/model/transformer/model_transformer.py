from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import transformers
import torch
import commune as c

class TransformerModel(c.Module):
    def __init__(self, config = None, **kwargs):
        self.set_config(config, kwargs=kwargs)

        self.config.model = self.config.shortcuts.get(self.config.model, self.config.model)

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model)
        self.model = AutoModelForCausalLM.from_pretrained(self.config.model, 
                                                        torch_dtype=torch.bfloat16, 
                                                        trust_remote_code=self.config.trust_remote_code,
                                                         device_map=self.config.device_map, 
                                                         max_memory = self.config.max_memory)

        if self.config.half:
            self.model = self.model.half()





    def generate(self, 
                prompt, 
                max_past_tokens = 256,
                max_new_tokens=10, 
                do_sample=False, 
                top_k=10, 
                early_stopping=True,
                truncation=True,
                padding = False,
                 **kwargs):


        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=max_past_tokens, truncation=truncation, padding=padding)


        outputs = self.model.generate(
            input_ids = inputs['input_ids'],
            attention_mask = inputs['attention_mask'],
            eos_token_id=self.model.config.eos_token_id,
            early_stopping=early_stopping,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample, 
            top_k=top_k, 
            **kwargs
        )

        output = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)

        return output

    talk = chat = text = generate


    @classmethod
    def test(cls, **kwargs):
        self = cls(**kwargs)
        prompt = "Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.\nDaniel: Hello, Girafatron!\nGirafatron:"
        return self.generate(prompt)
        
    @classmethod
    def serve(cls,
            model: str = None,
            tag = None,
            refresh = True,    
            **kwargs
            ):

        
        config = cls.get_config(kwargs=kwargs)
        config.tag = tag
        config.model = model if model is not None else config.model
        c.serve(module=cls.module_path(),
                name= f'model.{config.model}',
                tag = tag,
                kwargs={'config': config},
                refresh = refresh,
                verbose=True, **kwargs)


    @classmethod
    def fleet(cls, model=None, n:int = 1, **kwargs):
        return [cls(model=model, tag=i, **kwargs) for i in range(n)]
        

    @classmethod
    def text_generator(cls, text: str, module:str = 'model.wizardcoder' , max_tokens=2000, max_length=20, early_stopping:bool = False, timeout=20, **kwargs):
        model = c.connect(module)
        for i in range(max_tokens//max_length):
            # c.print(f'input: {text}')
            output =  model.talk(text, max_length=max_length, early_stopping=early_stopping, **kwargs)
            # c.print(output)

            if 'output' not in output:
                c.print({'error': f'output not in output, please consider increasing the timeout > {timeout}'}, color='red')
                break
            output_text  = output['output']
            yield output_text[len(text):]
            text = output_text
            if output['eos']:
                break

          