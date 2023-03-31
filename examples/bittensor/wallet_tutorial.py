
import commune
import bittensor

# bittensor way to create a wallet
wallet = bittensor.wallet(name='default', hotkey='default')

wallet.create()
bittensor_module  = commune.get_module('bittensor')(wallet=wallet)

bittensor_module.register()