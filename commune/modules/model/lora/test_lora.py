from commune.modules.model.lora.lora import LoraModel
import time

######## Example with 'togethercomputer/LLaMA-2-7B-32K' model and 'UNIST-Eunchan/NLP-Paper-to-QA-Generation' dataset ########
stime = time.time()
adaptor = LoraModel('togethercomputer/LLaMA-2-7B-32K')
print(f"Base model loading time: {time.time()-stime}")
def prep_data(example):
    example['prediction'] = example['question'] + ' ->: ' + example['answer']
    return example

# adaptor.train('UNIST-Eunchan/NLP-Paper-to-QA-Generation', './together-llama2-7b-paper2qa-lora-1', prep_data)

stime = time.time()
adaptor.load_adaptor('')
print(f'LoRA adaptor switching time: {time.time()-stime}')
adaptor.generate('How does their model learn using mostly raw data? ->: ')

stime = time.time()
adaptor.load_adaptor('./together-llama2-7b-paper2qa-lora-1')
print(f'LoRA adaptor initial loading time: {time.time()-stime}')
adaptor.generate('How does their model learn using mostly raw data? ->: ')

######## Example with 'togethercomputer/LLaMA-2-7B-32K' model and 'Abirate/english_quotes' dataset ########
# stime = time.time()
# adaptor = LoraModel('togethercomputer/LLaMA-2-7B-32K')
# print(f"Base model loading time: {time.time()-stime}")
# def prep_data(example):
#     example['prediction'] = example['quote'] + ' ->: ' + str(example['tags'])
#     return example
#
# # adaptor.train('Abirate/english_quotes', './together-llama2-7b-eng-quotes-lora-1', prep_data)
#
# stime = time.time()
# adaptor.load_adaptor('./together-llama2-7b-eng-quotes-lora-1')
# print(f'LoRA switching time: {time.time()-stime}')
# adaptor.generate('Be yourself; everyone else is already taken. ->: ')