# # # import commune as c
# # # hf = c.module('hf')
# # # tokenizer_class = c.import_object('commune.model.transformer.llama.LlamaTokenizer')
# # # path = hf.get_model_snapshot('llama')c 
# # # tokenizer = tokenizer_class.from_pretrained(path)

# # # print(tokenizer.encode('hello world'))
# import commune as c

# servers = c.servers('vali::var')


# daddy_key = 'module'
# ratio = 0.5

# def unstake_ratio(key='module', ratio:float=0.5):
#     s = c.module('subspace')()
#     staketo = s.my_staketo(fmt='j')[daddy_key]
#     c.print(staketo)
#     unstake_amount = {a: int(s*ratio) for a, s in staketo}
#     c.print(unstake_amount, 'unstake_amount')
#     for a, s in unstake_amount.items():
#         c.unstake(key=daddy_key, module_key=a, amount=s)

#     return {"success": True, "unstake_amount": unstake_amount}

# # def stake_across(key='module', ratio:float=0.5):
# #     s = c.module('subspace')()
# #     staketo = s.my_balance(fmt='j')[daddy_key]
# #     c.print(staketo)
# #     stake_amount = {a: int(s*ratio) for a, s in staketo}
# #     c.print(stake_amount, 'stake_amount')
# #     for a, s in stake_amount.items():
# #         c.stake(key=daddy_key, module_key=a, amount=s)

# #     return {"success": True, "stake_amount": stake_amount}

# s = c.module('subspace')()
# daddy_key = 'module'
# replace_key = 'storage'
# vali_key = 'vali.text'
# modules = c.my_modules(vali_key)
# balance = c.get_balance(daddy_key)
# stake_per_module = int(balance/len(modules))
# for m in modules:
#     c.stake(key=daddy_key, module_key=m['key'], amount=stake_per_module)
import commune as c
import base58

def is_ss58_address(address):
    try:
        # Decode the address using base58

        decoded_bytes = base58.b58decode_check(address)
        # Check if the decoded bytes have a valid length for SS58 addresses
        if len(decoded_bytes) == 32:
            return True
    except Exception as e:
        c.print(f'Error: {e}')
        pass
    
    return False

# Example usage:
address_to_check = b"5HVWVezXpJyfP66tQ8r38fd5fM3AuM9K5NtjHesdyRvvMuN5"
if is_ss58_address(address_to_check):
    print("Valid SS58 address")
else:
    print("Not a valid SS58 address")
