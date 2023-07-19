import random
import time
from typing import Optional, List

import bittensor


def rate_limit_sleep(rate_limit: Optional[int], extra: int = 1):
    if rate_limit is None:
        return
    blocks = random.randint(rate_limit, rate_limit + extra)
    time.sleep(blocks * bittensor.__blocktime__)


def update_weights(
    subtensor: bittensor.Subtensor,
    neurons: List[bittensor.NeuronInfoLite],
    netuid: int,
    rate_limit: Optional[int],
):
    bittensor.__version_as_int__ = {
        1: 1011,
        3: 401,
        11: 1011,
        21:1011,
    }[netuid]

    n_hotkeys = {neuron.hotkey: i for (i, neuron) in enumerate(neurons)}

    validator_uids = [neuron.uid for neuron in neurons]
    validator_weights = [neuron.consensus for neuron in neurons]

    validators = {
        1: [("test", "test1")],
        3: [("test", "test1")],
        11: [("test", "test1")],
        21: [("test", "test1")],
    }

    for name, hotkeys in {
        "test": {"test1": 0}
    }.items():
        for hotkey, _ in hotkeys.items():
            wallet = bittensor.wallet(name=name, hotkey=hotkey)
            if wallet.hotkey.ss58_address not in n_hotkeys:
                continue

            uid = n_hotkeys[wallet.hotkey.ss58_address]
            uids, weights, extra = [uid], [1.0], 1

            is_validator = (name, hotkey) in validators.get(netuid, [])
            if is_validator:
                uids, weights, extra = validator_uids, validator_weights, 10
                weights = [w * random.uniform(0.9, 1.1) for w in weights]

            subtensor.set_weights(
                wallet=wallet,  # type:ignore
                netuid=netuid,
                uids=uids,
                weights=weights,
                version_key=bittensor.__version_as_int__,
            )
            rate_limit_sleep(rate_limit, extra)

def main():
    subtensor = bittensor.subtensor()
    print("subtensor :")
    print(subtensor)
    rate_limit = subtensor.tx_rate_limit()
    print("rate_limit :")
    print(subtensor)
    subnets = subtensor.get_subnets()
    print("subnets :")
    print(subnets)

    def update_all():
        for netuid in subnets:
            print(netuid)
            neurons = subtensor.neurons_lite(netuid=netuid)
            print("updating...", netuid, rate_limit)
            update_weights(subtensor, neurons, netuid, rate_limit)

    blocks_per_epoch = 300
    while True:
        current_block = subtensor.block
        end_block = current_block + blocks_per_epoch

        print("block_numbers : ", current_block, end_block)

        update_all()

        print("updated")

        while end_block >= current_block:
            time.sleep(12)
            current_block = subtensor.block
            print(end_block - current_block)


if __name__ == "__main__":
    main()
