from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import transformers
import torch
import commune as c

class TransformerModel(c.Module):
    def __init__(self, config = None, **kwargs):
        self.set_config(config, kwargs=kwargs)

        self.config.model = self.config.shortcuts.get(self.config.model, self.config.model)

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model)
        self.eos_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.config.model, 
                                                        torch_dtype=torch.bfloat16, 
                                                        trust_remote_code=self.config.trust_remote_code,
                                                         device_map=self.config.device_map, 
                                                         max_memory = self.config.max_memory)

        if self.config.half:
            self.model = self.model.half()





    def generate(self, 
                prompt, 
                 max_length=200, 
                 do_sample=False, 
                 top_k=10, 
                 num_return_sequences=1,
                 early_stopping=True,
                 **kwargs):

        generation_config = dict(

        )

        inputs = self.tokenizer(prompt, return_tensors="pt")

        c.print('INPUT_KEYS',inputs.keys())
        outputs = self.model.generate(
            input_ids = inputs['input_ids'],
            attention_mask = inputs['attention_mask'],
            max_length=max_length,
            eos_token_id=self.model.config.eos_token_id,
            early_stopping=early_stopping,
            max_new_tokens=max_length,
            do_sample=do_sample, 
            top_k=top_k, 
            **kwargs
        )

        output = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)

        return output

    text = generate




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
        c.serve(module=cls.module_path(),
                name= f'model.{model}',
                tag = tag,
                kwargs={'config': config},
                refresh = refresh,
                verbose=True, **kwargs)
        

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

          