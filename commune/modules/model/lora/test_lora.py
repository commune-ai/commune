from commune.modules.model.lora.lora import LoraModel

# adaptor = LoraModel('TheBloke/Llama-2-7b-chat-fp16')
adaptor = LoraModel('togethercomputer/LLaMA-2-7B-32K')

# adaptor.train('Abirate/english_quotes', './bloke-llama2-7b-abirate-eng-quote-lora-1')
# adaptor.train('UNIST-Eunchan/NLP-Paper-to-QA-Generation', './together-llama2-7b-paper2qa-lora-1')
# adaptor.train('Abirate/english_quotes', './together-llama2-7b-eng-quotes-lora-1')
#
# # adaptor.load('')
# # adaptor.generate('How do I use the OpenAI API?')
# adaptor.load('./bloke-llama2-7b-paper2qa-lora-1')
# adaptor.generate('Be yourself; everyone else is already taken. ->: ')
# print('######################################')
adaptor.load('./together-llama2-7b-paper2qa-lora-1')
adaptor.generate('What is this paper about? ->: ')
# adaptor.load('./together-llama2-7b-eng-quotes-lora-1')
# adaptor.generate('Be yourself; everyone else is already taken. ->: ')
# print('######################################')
# adaptor.load('')
# adaptor.generate('How do I use the OpenAI API?')
