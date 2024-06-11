from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import transformers
import torch
import commune as c


class GPTQ(c.Module):
    def __init__(self, config = None, **kwargs):
        self.set_config(config, kwargs=kwargs)

        from transformers import AutoTokenizer, pipeline, logging
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model)

        self.model = AutoGPTQForCausalLM.from_quantized(self.config.name,
                model_basename=self.config.basename,
                use_safetensors=True,
                trust_remote_code=True,
                use_triton=self.config.use_triton,
                quantize_config=None)




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


        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            eos_token_id=self.model.config.eos_token_id,
            pad_token=self.model.config.pad_token_id,
            early_stopping=early_stopping,
            max_new_tokens=max_length,
            do_sample=do_sample, 
            top_k=top_k, 
            **kwargs
        )

        sequences = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)

        return sequences


    talk = chat = forward = generate


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

    @classmethod
    def talk(cls, text, module = 'model.lazarus30b', verbose:bool= True , *args, **kwargs):
        text_generator = cls.text_generator(text, module, **kwargs)
        output_text = ''
        for text in text_generator:
            if verbose:
                print(text, end='')
            output_text += text
        print('\n')

        # return output_text

    @classmethod
    def install(cls):
        c.cmd('pip install auto-gptq', verbose=True)
        c.cmd('pip install auto-gptq', verbose=True)

          