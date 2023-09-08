# # import commune as c
# # hf = c.module('hf')
# # tokenizer_class = c.import_object('commune.model.transformer.llama.LlamaTokenizer')
# # path = hf.get_model_snapshot('llama')c 
# # tokenizer = tokenizer_class.from_pretrained(path)

# # print(tokenizer.encode('hello world'))
import commune as c

servers = c.servers('vali::alah')

daddy_key = 'model.openai'

balance = c.get_balance(daddy_key)

stake_per_server = balance / len(servers)

for server in servers:
    c.stake(key=daddy_key, module_key=server , amount=stake_per_server)
    print(f'staked {stake_per_server} to {server}')

# import commune as c

# mems = []
# for i, l in enumerate(c.get_text('~/commune/data/mems.txt').split("'mnemonic': '")[1:]):
#     mems.append(l.split("'")[0])
# for i in range(len(mems)):
#     c.print(c.add_key(f'fam.{i}', mnemonic=mems[i], refresh=True))
