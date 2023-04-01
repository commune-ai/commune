import commune
wallet = f"{coldkey}.{hotkey}"

module = commune.get_commune('bittensor')
moduel(wallet=wallet).register(dev_ids = [0,1,2])