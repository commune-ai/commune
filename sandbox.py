import commune as c
c.module('model.openai')

bt = c.module('bittensor')
reged = bt.reged()
wallets = [bt.get_wallet(r) for r in reged]
wallet = c.choice(wallets)

metagraph = bt.get_metagraph()
d = bittensor.text_prompting_pool(keypair=wallet.hotkey, metagraph=cbt.get_metagraph())
response = d.forward(roles=['system', 'assistant'], messages=['you are chat bot', 'what is the whether'], timeout=6)
response = range(self.metagraph.n
c.print(response)
