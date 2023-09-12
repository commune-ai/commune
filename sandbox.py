# # import commune as c
# # hf = c.module('hf')
# # tokenizer_class = c.import_object('commune.model.transformer.llama.LlamaTokenizer')
# # path = hf.get_model_snapshot('llama')c 
# # tokenizer = tokenizer_class.from_pretrained(path)

# # print(tokenizer.encode('hello world'))
import commune as c

servers = c.servers('vali::var')

daddy_key = 'module'

balance = c.get_balance(daddy_key)

stake_per_server = balance / len(servers)

for server in servers:
    c.stake(key=daddy_key, module_key=server , amount=stake_per_server)
    print(f'staked {stake_per_server} to {server}')
